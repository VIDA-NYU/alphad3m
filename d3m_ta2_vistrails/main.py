from concurrent import futures
import grpc
import json
import logging
import os
from Queue import Queue
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
import d3m_ta2_vistrails.proto.core_pb2 as pb_core
import d3m_ta2_vistrails.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_vistrails.proto.dataflow_ext_pb2 as pb_dataflow
import d3m_ta2_vistrails.proto.dataflow_ext_pb2_grpc as pb_dataflow_grpc
from d3m_ta2_vistrails.utils import Observable, synchronized


logger = logging.getLogger(__name__)


vistrails_app = vistrails.core.application.init(
    options_dict={
        # Don't try to install missing dependencies
        'installBundles': False,
        # Don't enable all packages on start
        'loadPackages': False,
        # Enable packages automatically when they are required
        'enablePackagesSilently': True,
    },
    args=[])


vistrails_lock = threading.RLock()


class Session(Observable):
    def __init__(self):
        Observable.__init__(self)
        self.pipelines = {}


class Pipeline(object):
    pass


class D3mTa2(object):
    def __init__(self, storage_root,
                 logs_root=None, executables_root=None, results_root=None):
        self.storage = storage_root
        if not os.path.exists(self.storage):
            os.mkdir(self.storage)
        if not os.path.exists(os.path.join(self.storage, 'workflows')):
            os.mkdir(os.path.join(self.storage, 'workflows'))
        self.sessions = {}
        self._next_session = 0
        self.executor = futures.ThreadPoolExecutor(max_workers=10)

    def run_search(self, dataset, problem):
        raise NotImplementedError  # TODO: Standalone TA2 mode

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
        server.start()
        while True:
            time.sleep(60)

    def new_session(self):
        session = '%d' % self._next_session
        self._next_session += 1
        self.sessions[session] = Session()
        return session

    def finish_session(self, session_id):
        session = self.sessions.pop(session_id)
        session.notify('finish_session')

    def get_pipeline(self, session_id, pipeline_id):
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

    def build_pipelines(self, session_id):
        self.executor.submit(self._build_pipelines, session_id)

    def _build_pipelines(self, session_id):
        session = self.sessions[session_id]
        logger.info("Creating pipelines...")
        for template_iter in self.TEMPLATES:
            for template in template_iter:
                logger.info("Creating pipeline from %r", template)
                if isinstance(template, (list, tuple)):
                    func, args = template[0], template[1:]
                    tpl_func = lambda s: func(s, *args)
                else:
                    tpl_func = template
                try:
                    pipeline_id, pipeline = \
                        self._build_pipeline_from_template(session, tpl_func)
                except Exception:
                    logger.exception("Error building pipeline from %r",
                                     template)
                else:
                    pass  # TODO: Run the pipeline to get scores
        logger.info("Pipeline creation completed")
        session.notify('done')

    @synchronized(vistrails_lock)
    def _build_pipeline_from_template(self, session, template):
        pipeline_id = str(uuid.uuid4())

        # Create workflow from a template
        controller = template(self)

        # Save it to disk
        locator = BaseLocator.from_url(os.path.join(self.storage,
                                                    'workflows',
                                                    pipeline_id + '.vt'))
        controller.flush_delayed_actions()
        controller.write_vistrail(locator)

        # Add it to the database
        with session.lock:
            session.pipelines[pipeline_id] = pipeline = Pipeline()
        session.notify('new_pipeline', pipeline_id=pipeline_id)

        return pipeline_id, pipeline

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

        # Set the correct input files
        # TODO: Input files
        module = self._get_module(controller.current_pipeline, 'Train')
        if module is not None:
            ops.extend(controller.update_function_ops(
                module, 'name', ['TODO']))
        module = self._get_module(controller.current_pipeline, 'Target')
        if module is not None:
            ops.extend(controller.update_function_ops(
                module, 'name', ['TODO']))

        action = create_action(ops)
        controller.add_new_action(action)
        version = controller.perform_action(action)
        controller.change_selected_version(version)
        return controller

    TEMPLATES = [
        [(_classification_template, 'LinearSVC'),
         (_classification_template, 'KNeighborsClassifier'),
         (_classification_template, 'DecisionTreeClassifier')]
    ]


class CoreService(pb_core_grpc.CoreServicer):
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
        assert output == pb_core.FILE
        metrics = request.metrics
        target_features = request.target_features
        max_pipelines = request.max_pipelines

        logger.info("Got CreatePipelines request, session=%s",
                    sessioncontext.session_id)

        queue = Queue()
        session = self._app.sessions[sessioncontext.session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.build_pipelines(sessioncontext.session_id)

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
                    yield pb_core.PipelineCreateResult(
                        response_info=pb_core.Response(
                            status=pb_core.Status(code=pb_core.OK),
                        ),
                        progress_info=pb_core.SUBMITTED,
                        pipeline_id=pipeline_id,
                        pipeline_info=pb_core.Pipeline(
                            output=pb_core.FILE,  # FIXME: OutputType
                        ),
                    )
                elif event == 'ran':
                    pipeline_id = kwargs['pipeline_id']
                    scores = kwargs['scores']
                    yield pb_core.PipelineCreateResult(
                        response_info=pb_core.Response(
                            status=pb_core.Status(code=pb_core.OK),
                        ),
                        progress_info=pb_core.COMPLETED,
                        pipeline_id=pipeline_id,
                        pipeline_info=pb_core.Pipeline(
                            output=pb_core.FILE,
                            scores=scores,
                        ),
                    )
                elif event == 'done':
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

    def GetCreatePipelineResults(self, request, context):
        raise NotImplementedError  # TODO: GetCreatePipelineResults

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
        controller = self._app.get_pipeline(sessioncontext.session_id,
                                            pipeline_id)
        vt_pipeline = controller.current_pipeline
        modules = []
        for vt_module in vt_pipeline.module_list:
            functions = dict((func.name, func.params[0].strValue)
                             for func in vt_module.functions
                             if len(func.param) == 1)
            inputs = []
            for port in vt_module.destinationPorts():
                port = pb_dataflow.DataflowDescription.Input(
                    name=port.name,
                    type=port.sigstring,
                )
                if port.name in functions:
                    port.value.CopyFrom(functions[port.name])
                inputs.append(port)
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
        yield pb_dataflow.ModuleResult(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK),
            ),
            module_id='module1',
            status=pb_dataflow.ModuleResult.RUNNING,
            progress=0.5,
        )
        yield pb_dataflow.ModuleResult(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK),
            ),
            module_id='module1',
            status=pb_dataflow.ModuleResult.DONE,
            progress=1.0,
            outputs=[
                pb_dataflow.ModuleOutput(
                    output_name='result',
                    value='42',
                ),
            ],
            execution_time=60.0,
        )


def main_search():
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) == 2:
        with open(sys.argv[1]) as config_file:
            config = json.load(config_file)
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'],
            logs_root=config['pipeline_logs_root'],
            executables_root=config['executables_root'])
        ta2.run_search(
            dataset=config['training_data_root'],
            problem=config['problem_schema'])
    elif len(sys.argv) in (3, 4) and sys.argv[1] == 'serve':
        with open(sys.argv[2]) as config_file:
            config = json.load(config_file)
        port = None
        if len(sys.argv) == 4:
            port = sys.argv[3]
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'],
            logs_root=config['pipeline_logs_root'],
            executables_root=config['executables_root'])
        ta2.run_server(port)
    elif len(sys.argv) == 4 and sys.argv[1] == 'test':
        with open(sys.argv[2]) as config_file:
            config = json.load(config_file)
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'],
            results_root=config['results_path'])
        ta2.run_test(
            dataset=config['test_data_root'],
            executable_files=sys.argv[3])
    else:
        sys.stderr.write(
            "Invalid usage, either use:\n"
            "    1 argument: <config_file.json>\n"
            "        Runs the system standalone, solving the given problem as "
            "per official schemas\n\n"
            "    3 arguments: \"serve\" <config_file.json> [port_number]\n"
            "        Runs in server mode, waiting for a TA3 to connect on the "
            "given port\n"
            "        (default: 50051)\n")
        sys.exit(2)
