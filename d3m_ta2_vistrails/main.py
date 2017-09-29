from concurrent import futures
import grpc
import json
import logging
import multiprocessing
import os
from Queue import Empty, Queue
import sys
import threading
import time
import uuid
import vistrails.core.application
from vistrails.core.db.action import create_action
import vistrails.core.db.io
from vistrails.core.db.locator import BaseLocator, UntitledLocator
from vistrails.core.vistrail.controller import VistrailController

from d3m_ta2_vistrails import __version__
from d3m_ta2_vistrails.common import SCORES_FROM_SCHEMA
import d3m_ta2_vistrails.proto.core_pb2 as pb_core
import d3m_ta2_vistrails.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_vistrails.proto.dataflow_ext_pb2 as pb_dataflow
import d3m_ta2_vistrails.proto.dataflow_ext_pb2_grpc as pb_dataflow_grpc
from d3m_ta2_vistrails.train import train
from d3m_ta2_vistrails.utils import Observable, synchronized


logger = logging.getLogger(__name__)


vistrails_app = None


vistrails_lock = threading.RLock()


class Session(Observable):
    def __init__(self, id):
        Observable.__init__(self)
        self.id = id
        self.pipelines = {}
        self.training = False
        self.pipelines_training = set()

    def check_status(self):
        if self.training and not self.pipelines_training:
            self.notify('done_training')
            self.training = False
            logger.info("Session %s: training done", self.id)


class Pipeline(object):
    trained = False
    metrics = []
    train_run_module = None
    test_run_module = None

    def __init__(self, train_run_module, test_run_module):
        self.id = str(uuid.uuid4())
        self.train_run_module = train_run_module
        self.test_run_module = test_run_module
        self.scores = {}


class D3mTa2(object):
    def __init__(self, storage_root,
                 logs_root=None, executables_root=None, results_root=None):
        global vistrails_app
        if vistrails_app is None:
            vistrails_app = vistrails.core.application.init(
                options_dict={
                    # Don't try to install missing dependencies
                    'installBundles': False,
                    # Don't enable all packages on start
                    'loadPackages': False,
                    # Enable packages automatically when they are required
                    'enablePackagesSilently': True,
                    # Load additional packages from there
                    'userPackageDir': os.path.join(os.getcwd(), 'userpackages'),
                },
                args=[])

        self.storage = storage_root
        if not os.path.exists(self.storage):
            os.mkdir(self.storage)
        if not os.path.exists(os.path.join(self.storage, 'workflows')):
            os.mkdir(os.path.join(self.storage, 'workflows'))
        self.sessions = {}
        self._next_session = 0
        self.executor = futures.ThreadPoolExecutor(max_workers=10)
        self._run_queue = Queue()
        self._run_thread = threading.Thread(target=self._pipeline_running_thread)
        self._run_thread.setDaemon(True)
        self._run_thread.start()

    def run_search(self, dataset, problem):
        # Read problem
        with open(problem) as fp:
            problem_json = json.load(fp)
        if len(problem_json['datasets']) > 1:
            logger.error("Problem schema lists multiple datasets!")
            sys.exit(1)
        if problem_json['datasets'] != [dataset]:
            logger.error("Configuration and problem disagree on dataset! "
                         "Using configuration.")
        task = problem_json['taskType']
        metric = problem_json['metric']
        if metric not in SCORES_FROM_SCHEMA:
            logger.error("Unknown metric %r", metric)
            sys.exit(1)
        metric = SCORES_FROM_SCHEMA[metric]
        logger.info("Dataset: %s, task: %s, metric: %s",
                    dataset, task, metric)

        # Create pipelines
        session = Session('commandline')
        self.sessions[session.id] = session
        queue = Queue()
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self.build_pipelines(session.id, dataset,
                                 [metric])
            while queue.get(True)[0] != 'done_training':
                pass

        logger.info("Generated %d pipelines",
                    sum(1 for pipeline in session.pipelines.itervalues()
                        if pipeline.trained))

        # TODO: Export pipelines

    def run_test(self, dataset, executable_files):
        raise NotImplementedError  # TODO: Test executable mode

    def run_server(self, port=None):
        if not port:
            port = 50051
        core_rpc = CoreService(self)
        dataflow_rpc = DataflowService(self)
        server = grpc.server(self.executor)
        pb_core_grpc.add_CoreServicer_to_server(
            core_rpc, server)
        pb_dataflow_grpc.add_DataflowExtServicer_to_server(
            dataflow_rpc, server)
        server.add_insecure_port('[::]:%d' % port)
        logger.info("Started gRPC server on port %d", port)
        server.start()
        while True:
            time.sleep(60)

    def new_session(self):
        session = '%d' % self._next_session
        self._next_session += 1
        self.sessions[session] = Session(session)
        return session

    def finish_session(self, session_id):
        session = self.sessions.pop(session_id)
        session.notify('finish_session')

    def get_workflow(self, session_id, pipeline_id):
        if pipeline_id not in self.sessions[session_id].pipelines:
            raise KeyError("No such pipeline ID for session")

        filename = os.path.join(self.storage,
                                'workflows',
                                pipeline_id + '.vt')

        # copied from VistrailsApplicationInterface#open_vistrail()
        locator = BaseLocator.from_url(filename)
        loaded_objs = vistrails.core.db.io.load_vistrail(locator)
        controller = VistrailController(loaded_objs[0], locator,
                                        *loaded_objs[1:])
        controller.select_latest_version()
        return controller

    def build_pipelines(self, session_id, dataset, metrics):
        self.executor.submit(self._build_pipelines,
                             session_id, dataset, metrics)

    def _build_pipelines(self, session_id, dataset, metrics):
        session = self.sessions[session_id]
        logger.info("Creating pipelines...")
        session.training = True
        for template_iter in self.TEMPLATES:
            for template in template_iter:
                logger.info("Creating pipeline from %r", template)
                if isinstance(template, (list, tuple)):
                    func, args = template[0], template[1:]
                    tpl_func = lambda s: func(s, *args)
                else:
                    tpl_func = template
                try:
                    pipeline = self._build_pipeline_from_template(session,
                                                                  tpl_func)
                    pipeline.metrics = metrics
                except Exception:
                    logger.exception("Error building pipeline from %r",
                                     template)
                else:
                    session.pipelines_training.add(pipeline.id)
                    self._run_queue.put((session, pipeline, dataset))
        logger.info("Pipeline creation completed")
        session.check_status()

    @synchronized(vistrails_lock)
    def _build_pipeline_from_template(self, session, template):
        # Create workflow from a template
        controller, pipeline = template(self)

        # Save it to disk
        locator = BaseLocator.from_url(os.path.join(self.storage,
                                                    'workflows',
                                                    pipeline.id + '.vt'))
        controller.flush_delayed_actions()
        controller.write_vistrail(locator)

        # Add it to the database
        with session.lock:
            session.pipelines[pipeline.id] = pipeline
        session.notify('new_pipeline', pipeline_id=pipeline.id)

        return pipeline

    def _pipeline_running_thread(self):
        MAX_RUNNING_PROCESSES = 2
        running_pipelines = {}
        msg_queue = multiprocessing.Queue()
        while True:
            # Wait for a process to be done
            remove = []
            for session, pipeline, proc in running_pipelines.itervalues():
                if not proc.is_alive():
                    logger.info("Pipeline training process done, returned %d",
                                proc.exitcode)
                    if proc.exitcode == 0:
                        pipeline.trained = True
                        session.notify('training_success', pipeline_id=pipeline.id)
                    else:
                        session.notify('training_error', pipeline_id=pipeline.id)
                    session.pipelines_training.discard(pipeline.id)
                    session.check_status()
                    remove.append(pipeline.id)
            for id in remove:
                del running_pipelines[id]

            if len(running_pipelines) < MAX_RUNNING_PROCESSES:
                try:
                    session, pipeline, dataset = self._run_queue.get(False)
                except Empty:
                    pass
                else:
                    logger.info("Running training pipeline for %s", pipeline.id)
                    filename = os.path.join(self.storage, 'workflows',
                                            pipeline.id + '.vt')
                    proc = multiprocessing.Process(target=train,
                                                   args=(filename, pipeline,
                                                         dataset, msg_queue))
                    proc.start()
                    running_pipelines[pipeline.id] = session, pipeline, proc
                    session.notify('training_start', pipeline_id=pipeline.id)

            try:
                pipeline_id, msg, arg = msg_queue.get(timeout=3)
            except Empty:
                pass
            else:
                session, pipeline, proc = running_pipelines[pipeline_id]
                if msg == 'progress':
                    # TODO: Report progress
                    logger.info("Training pipeline %s: %.0f%%",
                                pipeline_id, arg * 100)
                elif msg == 'scores':
                    pipeline.scores = arg
                else:
                    logger.error("Unexpected message from training process %s",
                                 msg)

    def _new_controller(self):
        # Copied from VistrailsApplicationInterface#open_vistrail()
        locator = UntitledLocator()
        loaded_objs = vistrails.core.db.io.load_vistrail(locator)
        controller = VistrailController(loaded_objs[0], locator,
                                        *loaded_objs[1:])
        controller.select_latest_version()
        return controller

    def _load_template(self, name):
        # Copied from VistrailsApplicationInterface#open_vistrail()
        locator = BaseLocator.from_url(
            os.path.join(os.path.dirname(__file__),
                         '..',
                         'pipelines',
                         name))
        loaded_objs = vistrails.core.db.io.load_vistrail(locator)
        controller = VistrailController(loaded_objs[0], locator,
                                        *loaded_objs[1:])
        controller.select_latest_version()
        return controller

    @staticmethod
    def _replace_module(controller, ops, old_module_id, new_module):
        ops.append(('add', new_module))
        pipeline = controller.current_pipeline
        up_list = pipeline.graph.inverse_adjacency_list[old_module_id]
        for up_mod_id, up_conn_id in up_list:
            up_conn = pipeline.connections[up_conn_id]
            # Remove old connection
            ops.append(('delete', up_conn))
            # Add new connection
            new_up_conn = controller.create_connection(
                pipeline.modules[up_conn.source.moduleId], up_conn.source.name,
                new_module, up_conn.destination.name)
            ops.append(('add', new_up_conn))

        down_list = pipeline.graph.adjacency_list[old_module_id]
        assert len(down_list) <= 1
        for down_mod_id, down_conn_id in down_list:
            down_conn = pipeline.connections[down_conn_id]
            # Remove old connection
            ops.append(('delete', down_conn))
            # Add new connection
            new_down_conn = controller.create_connection(
                new_module, down_conn.source.name,
                pipeline.modules[down_conn.destination.moduleId],
                down_conn.destination.name)
            ops.append(('add', new_down_conn))

        ops.append(('delete', pipeline.modules[old_module_id]))

    @staticmethod
    def _get_module(pipeline, label):
        for module in pipeline.module_list:
            if '__desc__' in module.db_annotations_key_index:
                name = module.get_annotation_by_key('__desc__').value
                if name == label:
                    return module
        return None

    def _classification_template(self, classifier_name):
        controller = self._load_template('classification.xml')
        ops = []

        # Replace the classifier module
        module = self._get_module(controller.current_pipeline, 'Classifier')
        if module is not None:
            new_module = controller.create_module(
                'org.vistrails.vistrails.sklearn',
                classifier_name,
                namespace='classifiers')
            self._replace_module(controller, ops,
                                 module.id, new_module)
        else:
            raise ValueError("Couldn't find Classifier module in "
                             "classification template")

        action = create_action(ops)
        controller.add_new_action(action)
        version = controller.perform_action(action)
        controller.change_selected_version(version)
        return controller, Pipeline(train_run_module='classifier-sink',
                                    test_run_module='test_targets')

    TEMPLATES = [
        [(_classification_template, 'LinearSVC'),
         (_classification_template, 'KNeighborsClassifier'),
         (_classification_template, 'DecisionTreeClassifier')]
    ]


class CoreService(pb_core_grpc.CoreServicer):
    grpc2metric = dict((k, v) for v, k in pb_core.Metric.items())
    metric2grpc = dict(pb_core.Metric.items())

    def __init__(self, app):
        self._app = app

    def StartSession(self, request, context):
        version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]
        session_id = self._app.new_session()
        logger.info("Session started: %s (protocol version %s)",
                    session_id, version)
        return pb_core.SessionResponse(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK)
            ),
            user_agent='vistrails_ta2 %s' % __version__,
            version=version,
            context=pb_core.SessionContext(session_id=session_id),
        )

    def EndSession(self, request, context):
        if request.session_id in self._app.sessions:
            status = pb_core.OK
            self._app.finish_session(request.session_id)
            logger.info("Session terminated: %s", request.session_id)
        elif request.session_id < self._app._next_session:
            status = pb_core.SESSION_ENDED
        else:
            status = pb_core.SESSION_UNKNOWN
        return pb_core.Response(
            status=pb_core.Status(code=status),
        )

    def CreatePipelines(self, request, context):
        sessioncontext = request.context
        if sessioncontext.session_id not in self._app.sessions:
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
                )
            )
            return
        train_features = request.train_features
        task = request.task
        assert task == pb_core.CLASSIFICATION
        task_subtype = request.task_subtype
        task_description = request.task_description
        output = request.output
        metrics = request.metrics
        if any(m not in self.grpc2metric for m in metrics):
            logger.warning("Got metrics that we don't know about: %s",
                           ", ".join(m for m in metrics
                                     if m not in self.grpc2metric))
        metrics = [self.grpc2metric[m] for m in metrics
                   if m in self.grpc2metric]
        target_features = request.target_features
        max_pipelines = request.max_pipelines

        # FIXME: Handle the actual list of training features
        dataset = set(feat.data_uri for feat in train_features)
        if len(dataset) != 1:
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.INVALID_ARGUMENT,
                        details="Please only use a single training dataset",
                    ),
                )
            )
            return
        dataset, = dataset
        if dataset.startswith('file:///'):
            dataset = dataset[7:]

        logger.info("Got CreatePipelines request, session=%s, metrics=%s, "
                    "dataset=%s",
                    sessioncontext.session_id, ", ".join(metrics), dataset)

        queue = Queue()
        session = self._app.sessions[sessioncontext.session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.build_pipelines(sessioncontext.session_id, dataset,
                                      metrics)

            for msg in self._pipelinecreateresult_stream(context, queue,
                                                         session):
                yield msg

    def GetCreatePipelineResults(self, request, context):
        sessioncontext = request.context
        if sessioncontext.session_id not in self._app.sessions:
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
                ),
            )
            return
        pipeline_ids = request.pipeline_ids

        logger.info("Got GetCreatePipelineResults request, session=%s",
                    sessioncontext.session_id)

        queue = Queue()
        session = self._app.sessions[sessioncontext.session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            for msg in self._pipelinecreateresult_stream(context, queue,
                                                         session,
                                                         pipeline_ids):
                yield msg

    def _pipelinecreateresult_stream(self, context, queue,
                                     session, pipeline_ids=None):
        if pipeline_ids is None:
            pipeline_filter = lambda p_id: True
        else:
            pipeline_filter = lambda p_id, s=set(pipeline_ids): p_id in s

        while True:
            if not context.is_active():
                logger.info("Client closed CreatePipelines stream")
                break
            event, kwargs = queue.get()
            if event == 'finish_session':
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(code=pb_core.SESSION_ENDED),
                    )
                )
                break
            elif event == 'new_pipeline':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(code=pb_core.OK),
                    ),
                    progress_info=pb_core.SUBMITTED,
                    pipeline_id=pipeline_id,
                    pipeline_info=pb_core.Pipeline(
                        # FIXME: OutputType
                        output=pb_core.CLASS_LABEL,
                    ),
                )
            elif event == 'training_start':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(code=pb_core.OK),
                    ),
                    progress_info=pb_core.RUNNING,
                    pipeline_id=pipeline_id,
                )
            elif event == 'training_success':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                pipeline = session.pipelines[pipeline_id]
                scores = [
                    pb_core.Score(
                        metric=self.metric2grpc[m],
                        value=s,
                    )
                    for m, s in pipeline.scores.iteritems()
                    if m in self.metric2grpc
                ]
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(code=pb_core.OK),
                    ),
                    progress_info=pb_core.COMPLETED,
                    pipeline_id=pipeline_id,
                    pipeline_info=pb_core.Pipeline(
                        output=pb_core.CLASS_LABEL,
                        scores=scores,
                    ),
                )
            elif event == 'training_error':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(
                            code=pb_core.ABORTED,
                            details="Pipeline execution failed",
                        ),
                    ),
                    pipeline_id=pipeline_id,
                )
            elif event == 'done_training':
                break
            else:
                logger.error("Unexpected notification event %s",
                             event)

    def ExecutePipeline(self, request, context):
        sessioncontext = request.context
        if sessioncontext.session_id not in self._app.sessions:
            yield pb_core.PipelineExecuteResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
                ),
            )
            return
        pipeline_id = request.pipeline_id

        logger.info("Got ExecutePipeline request, session=%s",
                    sessioncontext.session_id)

        # TODO: ExecutePipeline
        yield pb_core.PipelineExecuteResult(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK),
            ),
            progress_info=pb_core.COMPLETED,
            pipeline_id=pipeline_id,
        )

    def ListPipelines(self, request, context):
        sessioncontext = request.context
        if sessioncontext.session_id not in self._app.sessions:
            return pb_core.PipelineListResult(
                status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
            )
        session = self._app.sessions[sessioncontext.session_id]
        with session.lock:
            pipelines = session.pipelines.keys()
        return pb_core.PipelineListResult(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK),
            ),
            pipeline_ids=pipelines,
        )

    def GetExecutePipelineResults(self, request, context):
        raise NotImplementedError  # TODO: GetExecutePipelineResults

    def ExportPipeline(self, request, context):
        raise NotImplementedError  # TODO: ExportPipeline


class DataflowService(pb_dataflow_grpc.DataflowExtServicer):
    def __init__(self, app):
        self._app = app

    @synchronized(vistrails_lock)
    def DescribeDataflow(self, request, context):
        sessioncontext = request.context
        assert sessioncontext.session_id in self._app.sessions
        pipeline_id = request.pipeline_id

        # Build description from VisTrails workflow
        controller = self._app.get_workflow(sessioncontext.session_id,
                                            pipeline_id)
        vt_pipeline = controller.current_pipeline
        modules = []
        for vt_module in vt_pipeline.module_list:
            functions = dict((func.name, func.params[0].strValue)
                             for func in vt_module.functions
                             if len(func.params) == 1)
            inputs = []
            for port in vt_module.destinationPorts():
                port_desc = dict(name=port.name,
                                 type=port.sigstring)
                if port.name in functions:
                    port_desc['value'] = functions[port.name]
                port_desc = pb_dataflow.DataflowDescription.Input(**port_desc)
                inputs.append(port_desc)
            outputs = [
                pb_dataflow.DataflowDescription.Output(
                    name=port.name,
                    type=port.sigstring,
                )
                for port in vt_module.sourcePorts()
            ]

            modules.append(pb_dataflow.DataflowDescription.Module(
                id='%d' % vt_module.id,
                type=vt_module.module_descriptor.sigstring,
                label=vt_module.module_descriptor.name,
                inputs=inputs,
                outputs=outputs,
            ))

        connections = []
        for vt_connection in vt_pipeline.connection_list:
            connections.append(pb_dataflow.DataflowDescription.Connection(
                from_module_id='%d' % vt_connection.source.moduleId,
                from_output_name=vt_connection.source.name,
                to_module_id='%d' % vt_connection.destination.moduleId,
                to_input_name=vt_connection.destination.name,
            ))

        return pb_dataflow.DataflowDescription(
            pipeline_id=pipeline_id,
            modules=modules,
            connections=connections,
        )

    @synchronized(vistrails_lock)
    def GetDataflowResults(self, request, context):
        sessioncontext = request.context
        assert sessioncontext.session_id in self._app.sessions
        pipeline_id = request.pipeline_id

        # TODO: GetDataflowResults
        if False:
            yield


def main_search():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    if len(sys.argv) != 2:
        sys.stderr.write(
            "Invalid usage, use:\n"
            "    ta2_search <config_file.json>\n"
            "        Run the TA2 system standalone, solving the given problem "
            "as per official schemas\n")
        sys.exit(1)
    else:
        with open(sys.argv[1]) as config_file:
            config = json.load(config_file)
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'],
            logs_root=config['pipeline_logs_root'],
            executables_root=config['executables_root'])
        ta2.run_search(
            dataset=os.path.dirname(config['dataset_schema']),
            problem=config['problem_schema'])

def main_serve():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    if len(sys.argv) not in (2, 3):
        sys.stderr.write(
            "Invalid usage, use:\n"
            "    ta2_serve <config_file.json> [port_number]\n"
            "        Runs in server mode, waiting for a TA3 to connect on the "
            "given port\n"
            "        (default: 50051)\n")
        sys.exit(1)
    else:
        with open(sys.argv[1]) as config_file:
            config = json.load(config_file)
        port = None
        if len(sys.argv) == 3:
            port = int(sys.argv[2])
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'],
            logs_root=config['pipeline_logs_root'],
            executables_root=config['executables_root'])
        ta2.run_server(port)


def main_test():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    if len(sys.argv) != 3:
        sys.exit(1)
    else:
        with open(sys.argv[1]) as config_file:
            config = json.load(config_file)
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'],
            results_root=config['results_path'])
        ta2.run_test(
            dataset=config['test_data_root'],
            executable_files=sys.argv[2])
