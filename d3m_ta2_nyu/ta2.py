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
from sqlalchemy import select
from sqlalchemy.orm import joinedload
import stat
import sys
import threading
import time
import uuid

from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA, TASKS_FROM_SCHEMA
from d3m_ta2_nyu import grpc_server
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.dataflow_ext_pb2_grpc as pb_dataflow_grpc
from d3m_ta2_nyu.test import test
from d3m_ta2_nyu.train import train
from d3m_ta2_nyu.utils import Observable
from d3m_ta2_nyu.workflow import database


MAX_RUNNING_PROCESSES = 2


logger = logging.getLogger(__name__)


class Session(Observable):
    """A session, in the GRPC meaning.

    This is a TA3 session in which pipelines are created.
    """
    def __init__(self, logs_dir, problem, DBSession):
        Observable.__init__(self)
        self.id = uuid.uuid4()
        self._logs_dir = logs_dir
        self.problem = problem
        self.pipelines = set()
        self.training = False
        self.pipelines_training = set()
        self.metrics = []
        self.DBSession = DBSession

    def check_status(self):
        if self.training and not self.pipelines_training:
            self.training = False
            logger.info("Session %s: training done", self.id)

            self.write_logs()
            self.notify('done_training')

    def write_logs(self):
        if not self.metrics:
            logger.error("Can't write logs for session, no metric is set!")
            return
        metric = self.metrics[0]

        try:
            with open(os.path.join(self.problem, 'problemDoc.json')) as fp:
                problem_id = json.load(fp)['about']['problemID']
        except (IOError, KeyError):
            logger.error("Error reading problemID from problem JSON")
            problem_id = 'problem_id_unset'

        written = 0
        db = self.DBSession()
        try:
            q = (
                db.query(database.CrossValidation)
                .options(joinedload(database.CrossValidation.pipeline)
                         .joinedload(database.Pipeline.modules))
                .filter(database.Pipeline.id.in_(self.pipelines))
                .filter(database.Pipeline.trained != 0)
                .order_by(
                    select([database.CrossValidationScore.value])
                    .where(database.CrossValidationScore.cross_validation_id ==
                           database.CrossValidation.id)
                    .where(database.CrossValidationScore.metric == metric)
                    .as_scalar()
                    .desc()
                )
            ).all()
            for i, crossval in enumerate(q):
                pipeline = crossval.pipeline
                filename = os.path.join(self._logs_dir, pipeline.id + '.json')
                obj = {
                    'problem_id': problem_id,
                    'pipeline_rank': i + 1,
                    'name': pipeline.id,
                    'primitives': [module.module_name
                                   for module in pipeline.modules],
                }
                with open(filename, 'w') as fp:
                    json.dump(obj, fp)
                written += 1
        finally:
            db.close()


class D3mTa2(object):
    def __init__(self, storage_root,
                 logs_root=None, executables_root=None):
        self.problem_id = 'problem_id_unset'
        self.problem = None
        self.storage = os.path.abspath(storage_root)
        if not os.path.exists(self.storage):
            os.makedirs(self.storage)
        if not os.path.exists(os.path.join(self.storage, 'workflows')):
            os.makedirs(os.path.join(self.storage, 'workflows'))
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

        self.db_filename = os.path.join(self.storage, 'db.sqlite3')
        self.dbengine, self.DBSession = database.connect(self.db_filename)

        self.sessions = {}
        self.executor = futures.ThreadPoolExecutor(max_workers=10)
        self._run_queue = Queue()
        self._run_thread = threading.Thread(
            target=self._pipeline_running_thread)
        self._run_thread.setDaemon(True)
        self._run_thread.start()

    def run_search(self, dataset, problem):
        # Read problem
        self.problem = problem
        with open(os.path.join(self.problem, 'problemDoc.json')) as fp:
            problem_json = json.load(fp)
        self.problem_id = problem_json['about']['problemID']
        task = problem_json['about']['taskType']
        if task not in TASKS_FROM_SCHEMA:
            logger.error("Unknown task %r", task)
            sys.exit(1)
        task = TASKS_FROM_SCHEMA[task]
        if task not in ('CLASSIFICATION', 'REGRESSION'):  # TODO
            logger.error("Unsupported task %s requested", task)
            sys.exit(1)
        metrics = []
        for metric in problem_json['inputs']['performanceMetrics']:
            metric = metric['metric']
            try:
                metric = SCORES_FROM_SCHEMA[metric]
            except KeyError:
                logger.error("Unknown metric %r", metric)
                sys.exit(1)
            metrics.append(metric)
        logger.info("Dataset: %s, task: %s, metrics: %s",
                    dataset, task, ", ".join(metrics))

        # Create pipelines
        session = Session(self.logs_root, self.problem, self.DBSession)
        session.metrics = metrics
        self.sessions[session.id] = session
        queue = Queue()
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self.build_pipelines(session.id, task, dataset,
                                 metrics)
            while queue.get(True)[0] != 'done_training':
                pass

        db = self.DBSession()
        try:
            pipelines = (
                db.query(database.Pipeline)
                .filter(database.Pipeline.trained)
                .options(joinedload(database.Pipeline.modules),
                         joinedload(database.Pipeline.connections))
            ).all()

            logger.info("Generated %d pipelines",
                        len(pipelines))

            for pipeline in pipelines:
                self.write_executable(pipeline)
        finally:
            db.close()

    def run_test(self, dataset, problem, pipeline_id, results_path):
        logger.info("About to run test")
        self.problem = problem
        with open(os.path.join(self.problem, 'problemDoc.json')) as fp:
            problem_json = json.load(fp)
        self.problem_id = problem_json['about']['problemID']
        test(pipeline_id, dataset, problem, results_path,
             db_filename=self.db_filename)

    def run_server(self, problem, port=None):
        self.problem = problem
        with open(os.path.join(self.problem, 'problemDoc.json')) as fp:
            problem_json = json.load(fp)
        self.problem_id = problem_json['about']['problemID']
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
        session = Session(self.logs_root, self.problem, self.DBSession)
        self.sessions[session.id] = session
        return session.id

    def finish_session(self, session_id):
        session = self.sessions.pop(session_id)
        session.notify('finish_session')

    def get_workflow(self, session_id, pipeline_id):
        if pipeline_id not in self.sessions[session_id].pipelines:
            raise KeyError("No such pipeline ID for session")

        db = self.DBSession()
        try:
            return (
                db.query(database.Pipeline)
                .filter(database.Pipeline.id == pipeline_id)
                .options(joinedload(database.Pipeline.modules),
                         joinedload(database.Pipeline.connections))
            ).one_or_none()
        finally:
            db.close()

    def build_pipelines(self, session_id, task, dataset, metrics):
        if not metrics:
            raise ValueError("no metrics")
        self.executor.submit(self._build_pipelines,
                             session_id, task, dataset, metrics)

    # Runs in a worker thread from executor
    def _build_pipelines(self, session_id, task, dataset, metrics):
        session = self.sessions[session_id]
        with session.lock:
            if session.metrics != metrics:
                if session.metrics:
                    old = 'from %s ' % ', '.join(session.metrics)
                else:
                    old = ''
                session.metrics = metrics
                logger.info("Set metrics to %s %s(for session %s)",
                            metrics, old, session_id)

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
                pipeline_id = self._build_pipeline_from_template(session,
                                                                 tpl_func)
            except Exception:
                logger.exception("Error building pipeline from %r",
                                 template)
            else:
                logger.info("Created pipeline %s", pipeline_id)
                session.pipelines_training.add(pipeline_id)
                self._run_queue.put((session, pipeline_id, dataset))
        logger.info("Pipeline creation completed")
        session.check_status()

    def _build_pipeline_from_template(self, session, template):
        # Create workflow from a template
        pipeline_id = template(self)

        # Add it to the session
        with session.lock:
            session.pipelines.add(pipeline_id)
        session.notify('new_pipeline', pipeline_id=pipeline_id)

        return pipeline_id

    # Runs in a background thread
    def _pipeline_running_thread(self):
        running_pipelines = {}
        msg_queue = multiprocessing.Queue()
        while True:
            # Wait for a process to be done
            remove = []
            for pipeline_id, (session, proc) in running_pipelines.items():
                if not proc.is_alive():
                    logger.info("Pipeline training process done, returned %d "
                                "(pipeline: %s)",
                                proc.exitcode, pipeline_id)
                    if proc.exitcode == 0:
                        session.notify('training_success',
                                       pipeline_id=pipeline_id)
                    else:
                        session.notify('training_error',
                                       pipeline_id=pipeline_id)
                    session.pipelines_training.discard(pipeline_id)
                    session.check_status()
                    remove.append(pipeline_id)
            for id in remove:
                del running_pipelines[id]

            if len(running_pipelines) < MAX_RUNNING_PROCESSES:
                try:
                    session, pipeline_id, dataset = self._run_queue.get(False)
                except Empty:
                    pass
                else:
                    logger.info("Running training pipeline for %s",
                                pipeline_id)
                    proc = multiprocessing.Process(
                        target=train,
                        args=(pipeline_id, session.metrics,
                              dataset, session.problem, msg_queue),
                        kwargs={'db_filename': self.db_filename})
                    proc.start()
                    running_pipelines[pipeline_id] = session, proc
                    session.notify('training_start', pipeline_id=pipeline_id)

            try:
                pipeline_id, msg, arg = msg_queue.get(timeout=3)
            except Empty:
                pass
            else:
                session, proc = running_pipelines[pipeline_id]
                if msg == 'progress':
                    # TODO: Report progress
                    logger.info("Training pipeline %s: %.0f%%",
                                pipeline_id, arg * 100)
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

    def _classification_template(self, classifier):
        db = self.DBSession()

        pipeline = database.Pipeline(
            origin="classification_template(classifier=%s, problemID=%r)" % (
                classifier, self.problem_id))

        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def connect(from_module, to_module,
                    from_output='data', to_input='data'):
            db.add(database.PipelineConnection(from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        try:
            data = make_module('data', '0.0', 'data')
            targets = make_module('data', '0.0', 'targets')
            imputer = make_module(
                'primitives', '0.0',
                'dsbox.datapreprocessing.cleaner.KNNImputation')
            encoder = make_module(
                'primitives', '0.0',
                'dsbox.datapreprocessing.cleaner.Encoder')
            classifier = make_module('sklearn-builtin', '0.0', classifier)

            connect(data, imputer)
            connect(imputer, encoder)
            connect(encoder, classifier)
            connect(targets, classifier, 'targets', 'targets')

            db.add(pipeline)
            db.commit()
            return pipeline.id
        finally:
            db.close()

    TEMPLATES = {
        'CLASSIFICATION': [
            (_classification_template,
             'sklearn.svm.classes.LinearSVC'),
            (_classification_template,
             'sklearn.neighbors.classification.KNeighborsClassifier'),
            (_classification_template,
             'sklearn.tree.tree.DecisionTreeClassifier'),
            (_classification_template,
             'sklearn.naive_bayes.MultinomialNB'),
            (_classification_template,
             'sklearn.ensemble.forest.RandomForestClassifier'),
            (_classification_template,
             'sklearn.linear_model.logistic.LogisticRegression'),
        ],
        'REGRESSION': [
            (_classification_template,
             'sklearn.linear_model.base.LinearRegression'),
            (_classification_template,
             'sklearn.linear_model.bayes.BayesianRidge'),
            (_classification_template,
             'sklearn.linear_model.coordinate_descent.LassoCV'),
            (_classification_template,
             'sklearn.linear_model.ridge.Ridge'),
            (_classification_template,
             'sklearn.linear_model.least_angle.Lars'),
        ],
    }
