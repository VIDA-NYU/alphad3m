from concurrent import futures
import grpc
import json
import logging
import multiprocessing
import os
from Queue import Empty, Queue
import shutil
import stat
import sys
import threading
import time
import vistrails.core.application
from vistrails.core.db.action import create_action
import vistrails.core.db.io
from vistrails.core.db.locator import BaseLocator, UntitledLocator
from vistrails.core.modules.module_registry import get_module_registry
from vistrails.core.vistrail.controller import VistrailController

from d3m_ta2_vistrails import __version__
from d3m_ta2_vistrails.common import SCORES_FROM_SCHEMA, \
    SCORES_RANKING_ORDER, TASKS_FROM_SCHEMA
from d3m_ta2_vistrails.names import name
import d3m_ta2_vistrails.proto.core_pb2 as pb_core
import d3m_ta2_vistrails.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_vistrails.proto.dataflow_ext_pb2 as pb_dataflow
import d3m_ta2_vistrails.proto.dataflow_ext_pb2_grpc as pb_dataflow_grpc
from d3m_ta2_vistrails.test import test
from d3m_ta2_vistrails.train import train
from d3m_ta2_vistrails.utils import Observable, synchronized


logger = logging.getLogger(__name__)


vistrails_app = None


vistrails_lock = threading.RLock()


class Session(Observable):
    def __init__(self, id, logs_dir, problem_id):
        Observable.__init__(self)
        self.id = id
        self._logs_dir = logs_dir
        self._problem_id = problem_id
        self.pipelines = {}
        self.training = False
        self.pipelines_training = set()

    def check_status(self):
        if self.training and not self.pipelines_training:
            self.training = False
            logger.info("Session %s: training done", self.id)

            # Rank pipelines
            pipelines = [pipeline for pipeline in self.pipelines.itervalues()
                         if pipeline.trained]
            metric = None
            for pipeline in pipelines:
                if pipeline.metrics:
                    metric = pipeline.metrics[0]
                    break
            if metric:
                logger.info("Ranking %d pipelines using %s...",
                            len(pipelines), metric)
                def rank(pipeline):
                    order = SCORES_RANKING_ORDER[metric]
                    if metric not in pipeline.scores:
                        return 9.0e99
                    return pipeline.scores.get(metric, 0) * order
                for i, pipeline in enumerate(sorted(pipelines, key=rank)):
                    pipeline.rank = i + 1
                    logger.info("  %d: %s", i + 1, pipeline.id)

            self.write_logs()
            self.notify('done_training')

    def write_logs(self):
        written = 0
        for pipeline in self.pipelines.itervalues():
            if not pipeline.trained:
                continue
            filename = os.path.join(self._logs_dir, pipeline.id + '.json')
            obj = {
                'problem_id': self._problem_id,
                'pipeline_rank': pipeline.rank,
                'name': pipeline.id,
                'primitives': pipeline.primitives,
            }
            with open(filename, 'w') as fp:
                json.dump(obj, fp)
            written += 1
        logger.info("Wrote %d log files", written)


class Pipeline(object):
    trained = False
    metrics = []

    def __init__(self, primitives=None):
        self.id = name()
        self.scores = {}
        self.rank = 0
        self.primitives = []
        if primitives is not None:
            self.primitives = list(primitives)


class D3mTa2(object):
    def __init__(self, storage_root,
                 logs_root=None, executables_root=None):
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
                    'userPackageDir': os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        '../userpackages'),
                },
                args=[])

        self.problem_id = 'problem_id_unset'
        self.storage = os.path.abspath(storage_root)
        if not os.path.exists(self.storage):
            os.makedirs(self.storage)
        if not os.path.exists(os.path.join(self.storage, 'workflows')):
            os.makedirs(os.path.join(self.storage, 'workflows'))
        if not os.path.exists(os.path.join(self.storage, 'persist')):
            os.makedirs(os.path.join(self.storage, 'persist'))
        if logs_root is not None:
            self.logs_root = os.path.abspath(logs_root)
        else:
            self.logs_root = None
        if self.logs_root and not os.path.exists(self.logs_root):
            os.makedirs(self.logs_root)
        if executables_root:
            self.executables_root = os.path.abspath(executables_root)
        else:
            self.executables_root = None
        if self.executables_root and not os.path.exists(self.executables_root):
            os.makedirs(self.executables_root)
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
        self.problem_id = problem_json['problemId']
        if len(problem_json['datasets']) > 1:
            logger.error("Problem schema lists multiple datasets!")
            sys.exit(1)
        if problem_json['datasets'] != [dataset]:
            logger.error("Configuration and problem disagree on dataset! "
                         "Using configuration.\n"
                         "%r != %r", dataset, problem_json['datasets'])
        task = problem_json['taskType']
        if task not in TASKS_FROM_SCHEMA:
            logger.error("Unknown task %r", task)
            sys.exit(1)
        task = TASKS_FROM_SCHEMA[task]
        if task not in ('CLASSIFICATION', 'REGRESSION'):  # TODO
            logger.error("Unsupported task %s requested", task)
            sys.exit(1)
        metric = problem_json['metric']
        if metric not in SCORES_FROM_SCHEMA:
            logger.error("Unknown metric %r", metric)
            sys.exit(1)
        metric = SCORES_FROM_SCHEMA[metric]
        logger.info("Dataset: %s, task: %s, metric: %s",
                    dataset, task, metric)

        # Create pipelines
        session = Session('commandline', self.logs_root, self.problem_id)
        self.sessions[session.id] = session
        queue = Queue()
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self.build_pipelines(session.id, task, dataset,
                                 [metric])
            while queue.get(True)[0] != 'done_training':
                pass

        logger.info("Generated %d pipelines",
                    sum(1 for pipeline in session.pipelines.itervalues()
                        if pipeline.trained))

        for pipeline in session.pipelines.itervalues():
            if pipeline.trained:
                self.write_executable(pipeline)

    def run_test(self, dataset, pipeline_id, results_path):
        vt_file = os.path.join(self.storage,
                               'workflows',
                               pipeline_id + '.vt')
        persist_dir = os.path.join(self.storage, 'persist',
                                   pipeline_id)
        logger.info("About to run test")
        test(vt_file, dataset, persist_dir, results_path)

    def run_server(self, problem_id, port=None):
        self.problem_id = problem_id
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
        self.sessions[session] = Session(session,
                                         self.logs_root, self.problem_id)
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

    def build_pipelines(self, session_id, task, dataset, metrics):
        self.executor.submit(self._build_pipelines,
                             session_id, task, dataset, metrics)

    def _build_pipelines(self, session_id, task, dataset, metrics):
        session = self.sessions[session_id]
        logger.info("Creating pipelines...")
        session.training = True
        for template in self.TEMPLATES.get(task, []):
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
                logger.info("Created pipeline %s", pipeline.id)
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
                        session.notify('training_success',
                                       pipeline_id=pipeline.id)
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
                    persist_dir = os.path.join(self.storage, 'persist',
                                               pipeline.id)
                    if not os.path.exists(persist_dir):
                        os.makedirs(persist_dir)
                    shutil.copyfile(
                        os.path.join(dataset, 'dataSchema.json'),
                        os.path.join(persist_dir, 'dataSchema.json'))
                    proc = multiprocessing.Process(target=train,
                                                   args=(filename, pipeline,
                                                         dataset, persist_dir,
                                                         msg_queue))
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

    def write_executable(self, pipeline, filename=None):
        if filename is None:
            filename = os.path.join(self.executables_root, pipeline.id)
        with open(filename, 'w') as fp:
            fp.write('#!/bin/sh\n\n'
                     'echo "Running pipeline {pipeline_id}..." >&2\n'
                     '{python} -c '
                     '"from d3m_ta2_vistrails.main import main_test; '
                     'main_test()" {pipeline_id} "$@"\n'.format(
                         pipeline_id=pipeline.id,
                         python=sys.executable))
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)
        logger.info("Wrote executable %s", filename)

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
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
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

    def _classification_template(self, classifier_name, primitive):
        from vistrails.core.modules.utils import parse_descriptor_string

        controller = self._load_template('classification.xml')
        ops = []

        # Replace the classifier module
        module = self._get_module(controller.current_pipeline, 'Classifier')
        if module is None:
            raise ValueError("Couldn't find Classifier module in "
                             "classification template")

        mod_identifier, mod_name, mod_namespace = parse_descriptor_string(
            classifier_name,
            'org.vistrails.vistrails.sklearn')
        new_module = controller.create_module(
            mod_identifier,
            mod_name,
            namespace=mod_namespace)
        self._replace_module(controller, ops,
                             module.id, new_module)

        primitives = [
            'dsbox.datapreprocessing.cleaner.Encoder',
            'dsbox.datapreprocessing.cleaner.Imputation'
        ] + [primitive]

        action = create_action(ops)
        controller.add_new_action(action)
        version = controller.perform_action(action)
        controller.change_selected_version(version)
        return controller, Pipeline(primitives)

    TEMPLATES = {
        'CLASSIFICATION': [
            (_classification_template, 'classifiers|LinearSVC',
             'sklearn.svm.classes.LinearSVC'),
            (_classification_template, 'classifiers|KNeighborsClassifier',
             'sklearn.neighbors.classification.KNeighborsClassifier'),
            (_classification_template, 'classifiers|DecisionTreeClassifier',
             'sklearn.tree.tree.DecisionTreeClassifier'),
            (_classification_template, 'classifiers|MultinomialNB',
             'sklearn.naive_bayes.MultinomialNB'),
            (_classification_template, 'classifiers|RandomForestClassifier',
             'sklearn.ensemble.forest.RandomForestClassifier'),
            (_classification_template, 'classifiers|LogisticRegression',
             'sklearn.linear_model.logistic.LogisticRegression'),
        ],
        'REGRESSION': [
            (_classification_template, 'regressors|LinearRegression',
             'sklearn.linear_model.base.LinearRegression'),
            (_classification_template, 'regressors|BayesianRidge',
             'sklearn.linear_model.bayes.BayesianRidge'),
            (_classification_template, 'regressors|LassoCV',
             'sklearn.linear_model.coordinate_descent.LassoCV'),
            (_classification_template, 'regressors|Ridge',
             'sklearn.linear_model.ridge.Ridge'),
            (_classification_template, 'regressors|Lars',
             'sklearn.linear_model.least_angle.Lars'),
        ],
    }


class CoreService(pb_core_grpc.CoreServicer):
    grpc2metric = dict((k, v) for v, k in pb_core.Metric.items()
                       if k != pb_core.METRIC_UNDEFINED)
    metric2grpc = dict(pb_core.Metric.items())
    grpc2task = dict((k, v) for v, k in pb_core.TaskType.items()
                     if k != pb_core.TASK_TYPE_UNDEFINED)

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
        if task not in self.grpc2task:
            logger.error("Got unknown task %r", task)
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.INVALID_ARGUMENT,
                        details="Unknown task",
                    ),
                ),
            )
            return
        task = self.grpc2task[task]
        if task not in ('CLASSIFICATION', 'REGRESSION'):  # TODO
            logger.error("Unsupported task %s requested", task)
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.UNIMPLEMENTED,
                        details="Only CLASSIFICATION and REGRESSION are "
                                "supported for now",
                    ),
                ),
            )
            return
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
        if not metrics:
            logger.error("Didn't get any metrics we know")
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.INVALID_ARGUMENT,
                        details="Didn't get any metrics we know",
                    ),
                ),
            )
            return
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
                ),
            )
            return
        dataset, = dataset
        if dataset.startswith('file:///'):
            dataset = dataset[7:]

        logger.info("Got CreatePipelines request, session=%s, task=%s, "
                    "dataset=%s, metrics=%s, ",
                    sessioncontext.session_id, task,
                    dataset, ", ".join(metrics))

        queue = Queue()
        session = self._app.sessions[sessioncontext.session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.build_pipelines(sessioncontext.session_id, task, dataset,
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
                    progress_info=pb_core.ERRORED,
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
        sessioncontext = request.context
        if sessioncontext.session_id not in self._app.sessions:
            return pb_core.Response(
                status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
            )
        session = self._app.sessions[sessioncontext.session_id]
        with session.lock:
            if request.pipeline_id not in session.pipelines:
                return pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.INVALID_ARGUMENT,
                        details="No such pipeline"),
                )
            pipeline = session.pipelines[request.pipeline_id]
            if not pipeline.trained:
                return pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.UNAVAILABLE,
                        details="This pipeline is not trained yet"),
                )
            uri = request.pipeline_exec_uri
            if uri.startswith('file:///'):
                uri = uri[7:]
            self._app.write_executable(pipeline, filename=uri)
        return pb_core.Response(
            status=pb_core.Status(code=pb_core.OK)
        )


class DataflowService(pb_dataflow_grpc.DataflowExtServicer):
    def __init__(self, app):
        self._app = app

    @synchronized(vistrails_lock)
    def DescribeDataflow(self, request, context):
        sessioncontext = request.context
        assert sessioncontext.session_id in self._app.sessions
        pipeline_id = request.pipeline_id

        # We want to hide Persist modules, which are pass-through modules used
        # to transfer state between train & test
        registry = get_module_registry()
        descr_persist = registry.get_descriptor_by_name(
            'org.vistrails.vistrails.persist',
            'Persist')
        persist_modules = {}

        # Build description from VisTrails workflow
        controller = self._app.get_workflow(sessioncontext.session_id,
                                            pipeline_id)
        vt_pipeline = controller.current_pipeline
        modules = []
        for vt_module in vt_pipeline.module_list:
            if vt_module.module_descriptor == descr_persist:
                persist_modules[vt_module.id] = None
                continue

            functions = dict((func.name, func.params[0].strValue)
                             for func in vt_module.functions
                             if len(func.params) == 1)
            inputs = []
            for port in vt_module.destinationPorts():
                port_desc = dict(name=port.name,
                                 type=port.sigstring)
                if port.name in functions:
                    port_desc['value'] = functions[port.name]
                elif port.optional:
                    continue  # Skip unset optional ports
                port_desc = pb_dataflow.DataflowDescription.Input(**port_desc)
                inputs.append(port_desc)
            outputs = [
                pb_dataflow.DataflowDescription.Output(
                    name=port.name,
                    type=port.sigstring,
                )
                for port in vt_module.sourcePorts()
            ]

            label = vt_module.module_descriptor.name
            if '__desc__' in vt_module.db_annotations_key_index:
                label = vt_module.get_annotation_by_key('__desc__').value

            modules.append(pb_dataflow.DataflowDescription.Module(
                id='%d' % vt_module.id,
                type=vt_module.module_descriptor.sigstring,
                label=label,
                inputs=inputs,
                outputs=outputs,
            ))

        # Find upstream connections of Persist modules
        for vt_connection in vt_pipeline.connection_list:
            dest_mod_id = vt_connection.destination.moduleId
            if dest_mod_id in persist_modules:
                persist_modules[dest_mod_id] = vt_connection.source

        connections = []
        for vt_connection in vt_pipeline.connection_list:
            if vt_connection.destination.moduleId in persist_modules:
                continue

            # Connect over the Persist modules
            source = vt_connection.source
            if source.moduleId in persist_modules:
                source = persist_modules[source.moduleId]

            connections.append(pb_dataflow.DataflowDescription.Connection(
                from_module_id='%d' % source.moduleId,
                from_output_name=source.name,
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

    if len(sys.argv) not in (1, 2):
        sys.stderr.write(
            "Invalid usage, use:\n"
            "    ta2_serve [port_number]\n"
            "        Runs in server mode, waiting for a TA3 to connect on the "
            "given port\n"
            "        (default: 50051)\n"
            "        The configuration file is read from $CONFIG_JSON_PATH\n")
        sys.exit(1)
    elif 'CONFIG_JSON_PATH' not in os.environ:
        sys.stderr.write("CONFIG_JSON_PATH is not set!\n")
        sys.exit(1)
    else:
        with open(os.environ['CONFIG_JSON_PATH']) as config_file:
            config = json.load(config_file)
        port = None
        if len(sys.argv) == 2:
            port = int(sys.argv[1])
        with open(config['problem_schema']) as fp:
            problem_id = json.load(fp)['problemId']
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'],
            logs_root=config['pipeline_logs_root'],
            executables_root=config['executables_root'])
        ta2.run_server(problem_id, port)


def main_test():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    if len(sys.argv) != 3:
        sys.exit(1)
    else:
        with open(sys.argv[2]) as config_file:
            config = json.load(config_file)
        ta2 = D3mTa2(
            storage_root=config['temp_storage_root'])
        ta2.run_test(
            dataset=config['test_data_root'],
            pipeline_id=sys.argv[1],
            results_path=config['results_path'])
