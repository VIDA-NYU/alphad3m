"""The D3mTa2 class, that creates pipelines, train, and run them.

We use multiprocessing to run training in separate processes, sending messages
back to this process via a Queue.
"""

from concurrent import futures
import grpc
import json
import logging
import multiprocessing
import os
from queue import Empty, Queue
import shutil
import stat
import sys
import threading
import time

from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA, SCORES_RANKING_ORDER, \
    TASKS_FROM_SCHEMA
from d3m_ta2_nyu import grpc_server
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.dataflow_ext_pb2_grpc as pb_dataflow_grpc
from d3m_ta2_nyu.test import test
from d3m_ta2_nyu.train import train
from d3m_ta2_nyu.utils import Observable
from d3m_ta2_nyu.workflow import database


logger = logging.getLogger(__name__)


engine, db_session = database.connect()


class Session(Observable):
    """A session, in the GRPC meaning.

    This is a TA3 session in which pipelines are created.
    """
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
            pipelines = [pipeline for pipeline in self.pipelines.values()
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
        for pipeline in self.pipelines.values():
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
                    sum(1 for pipeline in session.pipelines.values()
                        if pipeline.trained))

        for pipeline in session.pipelines.values():
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
        core_rpc = grpc_server.CoreService(self)
        dataflow_rpc = grpc_server.DataflowService(self)
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

    # Runs in a background thread
    def _pipeline_running_thread(self):
        MAX_RUNNING_PROCESSES = 2
        running_pipelines = {}
        msg_queue = multiprocessing.Queue()
        while True:
            # Wait for a process to be done
            remove = []
            for session, pipeline, proc in running_pipelines.values():
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
                     '"from d3m_ta2_nyu.main import main_test; '
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
