"""The D3mTa2 class, that creates pipelines, train, and run them.

We use multiprocessing to run training in separate processes, sending messages
back to this process via a Queue.
"""

from concurrent import futures
import grpc
import itertools
import json
import logging
import multiprocessing
import os
import pickle
from queue import Empty, Queue
from sqlalchemy import select
from sqlalchemy.orm import aliased, joinedload
import stat
import subprocess
import sys
import threading
import time
import uuid

from . import __version__

from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA, SCORES_RANKING_ORDER, \
    TASKS_FROM_SCHEMA
from d3m_ta2_nyu.d3mds import D3MDataset
from d3m_ta2_nyu import grpc_server
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.dataflow_ext_pb2_grpc as pb_dataflow_grpc
from d3m_ta2_nyu.test import test
from d3m_ta2_nyu.utils import Observable
from d3m_ta2_nyu.workflow import database


MAX_RUNNING_PROCESSES = 1

TUNE_PIPELINES_COUNT = 3


logger = logging.getLogger(__name__)


class Session(Observable):
    """A session, in the GRPC meaning.

    This is a TA3 session in which pipelines are created.
    """
    def __init__(self, logs_dir, problem, problem_id, DBSession):
        Observable.__init__(self)
        self.id = uuid.uuid4()
        self._logs_dir = logs_dir
        self.DBSession = DBSession
        self.problem = problem
        self.problem_id = problem_id
        self.metrics = []

        # All the pipelines that belong to this session
        self.pipelines = set()
        # The pipelines currently in the queue for training
        self.pipelines_training = set()
        # The pipelines in the queue for hyperparameter tuning
        self.pipelines_tuning = set()
        # Pipelines already tuned, and pipelines created through tuning
        self.tuned_pipelines = set()
        # Flag indicating we started training & tuning, and a 'done_training'
        # signal should be sent once both pipelines_training and
        # pipelines_tuning are empty
        self.working = False

    def add_training_pipeline(self, pipeline_id):
        with self.lock:
            self.working = True
            self.pipelines.add(pipeline_id)
            self.pipelines_training.add(pipeline_id)

    def pipeline_training_done(self, pipeline_id):
        with self.lock:
            self.pipelines_training.discard(pipeline_id)
            self.check_status()

    def get_top_pipelines(self, db, metric, limit=None):
        # SELECT pipelines.*
        # FROM pipelines
        # WHERE (
        #     SELECT COUNT(runs.id)
        #     FROM runs
        #     WHERE runs.pipeline_id = pipelines.id AND
        #         runs.special = 0 AND
        #         runs.type = 'TRAIN'
        # ) != 0
        # ORDER BY (
        #     SELECT cross_validation_scores.value
        #     FROM cross_validation_scores
        #     INNER JOIN cross_validation ON cross_validations.id =
        #         cross_validation_scores.cross_validation_id
        #     WHERE cross_validation_scores.metric = 'F1_MACRO' AND
        #         cross_validations.pipeline_id = pipelines.id
        # ) DESC;
        pipeline = aliased(database.Pipeline)
        crossval_score = (
            select([database.CrossValidationScore.value])
            .where(database.CrossValidationScore.cross_validation_id ==
                   database.CrossValidation.id)
            .where(database.CrossValidationScore.metric == metric)
            .where(database.CrossValidation.pipeline_id == pipeline.id)
            .as_scalar()
        )
        if SCORES_RANKING_ORDER[metric] == -1:
            crossval_score_order = crossval_score.desc()
        else:
            crossval_score_order = crossval_score.asc()
        q = (
            db.query(pipeline, crossval_score)
            .filter(pipeline.id.in_(self.pipelines))
            .filter(pipeline.trained)
            .options(joinedload(pipeline.modules),
                     joinedload(pipeline.connections))
            .order_by(crossval_score_order)
        )
        if limit is not None:
            q = q.limit(limit)
        return q.all()

    def check_status(self):
        with self.lock:
            # We are already done
            if not self.working:
                return
            # If pipelines are still in the queue
            if self.pipelines_training or self.pipelines_tuning:
                return

            # If we are out of pipelines to train, maybe submit pipelines for
            # tuning
            logger.info("Session %s: training done", self.id)

            db = self.DBSession()
            tune = []
            try:
                top_pipelines = self.get_top_pipelines(
                    db, self.metrics[0],
                    TUNE_PIPELINES_COUNT)
                for pipeline, _ in top_pipelines:
                    if pipeline.id not in self.tuned_pipelines:
                        tune.append(pipeline.id)
            finally:
                db.close()

            tune = []  # TODO: Submit
            if tune:
                # Found some pipelines to tune, do that
                logger.info("Found %d pipelines to tune:", len(tune))
                for pipeline_id in tune:
                    logger.info("    %s", pipeline_id)
                    # TODO: Submit
                    self.pipelines_tuning.add(pipeline_id)
                return
            logger.info("Found no pipeline to tune")

            # Session is done (but new pipelines might be added later)
            self.working = False

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
            top_pipelines = self.get_top_pipelines(db, metric)
            logger.info("Writing logs for %d pipelines", len(top_pipelines))
            for i, (pipeline, score) in enumerate(top_pipelines):
                logger.info("    %d) %s %s=%s",
                            i + 1, pipeline.id, metric, score)
                filename = os.path.join(self._logs_dir,
                                        str(pipeline.id) + '.json')
                obj = {
                    'problem_id': problem_id,
                    'pipeline_rank': i + 1,
                    'name': str(pipeline.id),
                    'primitives': [
                        module.name
                        for module in pipeline.modules
                        if module.package in ('primitives', 'sklearn-builtin')
                    ],
                }
                with open(filename, 'w') as fp:
                    json.dump(obj, fp)
                written += 1
        finally:
            db.close()


class D3mTa2(object):
    def __init__(self, storage_root,
                 logs_root=None, executables_root=None):
        if 'TA2_DEBUG_BE_FAST' in os.environ:
            logger.warning("**************************************************"
                           "*****")
            logger.warning("***   DEBUG mode is on, will try fewer pipelines  "
                           "  ***")
            logger.warning("*** If this is not wanted, unset $TA2_DEBUG_BE_FAS"
                           "T ***")
            logger.warning("**************************************************"
                           "*****")
        self.default_problem_id = 'problem_id_unset'
        self.default_problem = None
        self.storage = os.path.abspath(storage_root)
        if not os.path.exists(self.storage):
            os.makedirs(self.storage)
        self.predictions_root = os.path.join(self.storage, 'tmp_predictions')
        if not os.path.exists(self.predictions_root):
            os.mkdir(self.predictions_root)
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

        logger.info("TA2 started, version=%s", __version__)

    def run_search(self, dataset, problem):
        """Run the search phase: create pipelines, train and score them.

        This is called by the ``ta2_search`` executable, it is part of the
        evaluation.
        """
        # Read problem
        with open(os.path.join(problem, 'problemDoc.json')) as fp:
            problem_json = json.load(fp)
        problem_id = problem_json['about']['problemID']
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
        session = Session(self.logs_root, problem, problem_id, self.DBSession)
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
            pipelines = session.get_top_pipelines(db, metrics[0])

            for pipeline, score in itertools.islice(pipelines, 20):
                self.write_executable(pipeline)
        finally:
            db.close()

    def run_pipeline(self, pipeline_id, dataset, problem, metric=None):
        """Train and score a single pipeline.

        This is used to test the pipeline synthesis code.
        """
        # Read problem
        with open(os.path.join(problem, 'problemDoc.json')) as fp:
            problem_json = json.load(fp)
        problem_id = problem_json['about']['problemID']

        if metric is not None:
            try:
                metric = SCORES_FROM_SCHEMA[metric]
            except KeyError:
                raise ValueError("Unknown metric %r" % metric)
        else:
            if not problem_json['inputs']['performanceMetrics']:
                raise ValueError("Problem has no metric")
            metric = problem_json['inputs']['performanceMetrics'][0]['metric']
            try:
                metric = SCORES_FROM_SCHEMA[metric]
            except KeyError:
                raise ValueError("Unknown metric %r", metric)

        logger.info("Running single pipeline, dataset: %s, metric: %s",
                    dataset, metric)

        # Create session
        session = Session(self.logs_root, problem, problem_id, self.DBSession)
        session.metrics = [metric]
        self.sessions[session.id] = session
        queue = Queue()
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            session.add_training_pipeline(pipeline_id)
            self._run_queue.put((session, pipeline_id, dataset))
            while queue.get(True)[0] != 'done_training':
                pass

        db = self.DBSession()
        try:
            # Find most recent cross-validation
            crossval_id = (
                select([database.CrossValidation.id])
                .where(database.CrossValidation.pipeline_id == pipeline_id)
                .order_by(database.CrossValidation.date.desc())
            ).as_scalar()
            # Get scores from that cross-validation
            scores = (
                db.query(database.CrossValidationScore)
                .filter(database.CrossValidationScore.cross_validation_id ==
                        crossval_id)
            ).all()
            for score in scores:
                if score.metric == metric:
                    logger.info("Evaluation result: %s -> %r",
                                metric, score.value)
                    return score.value
            logger.info("Didn't get the requested metric from "
                        "cross-validation")
            return None
        finally:
            db.close()

    def run_test(self, dataset, problem, pipeline_id, results_root):
        """Run a previously trained pipeline.

        This is called by the generated executables, it is part of the
        evaluation.
        """
        logger.info("About to run test")
        with open(os.path.join(problem, 'problemDoc.json')) as fp:
            problem_json = json.load(fp)
        problem_id = problem_json['about']['problemID']
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        results_path = os.path.join(
            results_root,
            problem_json['expectedOutputs']['predictionsFile'])
        test(pipeline_id, dataset, problem, results_path,
             db_filename=self.db_filename)

    def run_server(self, problem, port=None):
        """Spin up the gRPC server to receive requests from a TA3 system.

        This is called by the ``ta2_serve`` executable. It is part of the
        TA2+TA3 evaluation.
        """
        self.default_problem = problem
        with open(os.path.join(problem, 'problemDoc.json')) as fp:
            problem_json = json.load(fp)
        self.default_problem_id = problem_json['about']['problemID']
        if not port:
            port = 45042
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
        if self.default_problem is None:
            logger.error("Creating a session but no default problem is set!")
        session = Session(self.logs_root, self.default_problem,
                          self.default_problem_id, self.DBSession)
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

    def get_pipeline_scores(self, session_id, pipeline_id):
        if pipeline_id not in self.sessions[session_id].pipelines:
            raise KeyError("No such pipeline ID for session")

        db = self.DBSession()
        try:
            # Find most recent cross-validation
            crossval_id = (
                select([database.CrossValidation.id])
                .where(database.CrossValidation.pipeline_id == pipeline_id)
                .order_by(database.CrossValidation.date.desc())
            ).as_scalar()
            # Get scores from that cross-validation
            scores = (
                db.query(database.CrossValidationScore)
                .filter(database.CrossValidationScore.cross_validation_id ==
                        crossval_id)
            ).all()
            return {score.metric: score.value for score in scores}
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
            # Force working=True so we get 'done_training' even if no pipeline
            # gets created
            session.working = True
            template_name = task
            if 'TA2_DEBUG_BE_FAST' in os.environ:
                template_name = 'DEBUG_' + task
            for template in self.TEMPLATES.get(template_name, []):
                logger.info("Creating pipeline from %r", template)
                if isinstance(template, (list, tuple)):
                    func, args = template[0], template[1:]
                    tpl_func = lambda s, **kw: func(s, *args, **kw)
                else:
                    tpl_func = template
                try:
                    self._build_pipeline_from_template(session, tpl_func,
                                                       dataset)
                except Exception:
                    logger.exception("Error building pipeline from %r",
                                     template)
            logger.info("Pipeline creation completed")
            session.check_status()

    def _build_pipeline_from_template(self, session, template, dataset):
        # Create workflow from a template
        pipeline_id = template(self, dataset=dataset)

        # Add it to the session
        session.add_training_pipeline(pipeline_id)

        logger.info("Created pipeline %s", pipeline_id)
        self._run_queue.put((session, pipeline_id, dataset))
        session.notify('new_pipeline', pipeline_id=pipeline_id)

    # Runs in a background thread
    def _pipeline_running_thread(self):
        running_pipelines = {}
        msg_queue = multiprocessing.Queue()
        while True:
            # Wait for a process to be done
            remove = []
            for pipeline_id, (session, proc) in running_pipelines.items():
                if proc.poll() is not None:
                    logger.info("Pipeline training process done, returned %d "
                                "(pipeline: %s)",
                                proc.returncode, pipeline_id)
                    if proc.returncode == 0:
                        results = os.path.join(self.predictions_root,
                                               '%s.csv' % pipeline_id)
                        session.notify('training_success',
                                       pipeline_id=pipeline_id,
                                       predict_result=results)
                    else:
                        session.notify('training_error',
                                       pipeline_id=pipeline_id)
                    session.pipeline_training_done(pipeline_id)
                    remove.append(pipeline_id)
            for id in remove:
                del running_pipelines[id]

            if len(running_pipelines) < MAX_RUNNING_PROCESSES:
                try:
                    session, pipeline_id, dataset = self._run_queue.get(False)
                except Empty:
                    pass
                else:
                    logger.info("Running training pipeline for %s "
                                "(session %s has %d pipelines left to train)",
                                pipeline_id, session.id,
                                len(session.pipelines_training))
                    results = os.path.join(self.predictions_root,
                                           '%s.csv' % pipeline_id)
                    # FIXME: Can't use multiprocessing here because of gRPC bug
                    # https://github.com/grpc/grpc/issues/12455
                    # If changing this back, update poll() and returncode to
                    # is_alive() and exitcode above.
                    #proc = multiprocessing.Process(
                    #    target=train,
                    #    args=(pipeline_id, session.metrics,
                    #          dataset, session.problem, results, msg_queue),
                    #    kwargs={'db_filename': self.db_filename})
                    #proc.start()
                    proc = subprocess.Popen(
                        [sys.executable,
                         '-c',
                         'import uuid; from d3m_ta2_nyu.train_and_tune import tune; '
                         'tune(uuid.UUID(hex=%r), %r, %r, %r, %r, '
                         'None, db_filename=%r)' % (
                             pipeline_id.hex, session.metrics,
                             dataset, session.problem, results,
                             self.db_filename,
                         )]
                    )
                    running_pipelines[pipeline_id] = session, proc
                    session.notify('training_start', pipeline_id=pipeline_id)

            try:
                pipeline_id, msg, arg = msg_queue.get(timeout=3)
            except Empty:
                pass
            else:
                if msg == 'progress':
                    # TODO: Report progress
                    logger.info("Training pipeline %s: %.0f%%",
                                pipeline_id, arg * 100)
                else:
                    logger.error("Unexpected message from training process %s",
                                 msg)

    def write_executable(self, pipeline, filename=None):
        if not filename:
            filename = os.path.join(self.executables_root, str(pipeline.id))
        with open(filename, 'w') as fp:
            fp.write('#!/bin/sh\n\n'
                     'echo "Running pipeline {pipeline_id}..." >&2\n'
                     '{python} -c '
                     '"from d3m_ta2_nyu.main import main_test; '
                     'main_test()" {pipeline_id} "$@"\n'.format(
                         pipeline_id=str(pipeline.id),
                         python=sys.executable))
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)
        logger.info("Wrote executable %s", filename)

    def _classification_template(self, imputer_cat, imputer_num, encoder,
                                 classifier, dataset):
        db = self.DBSession()

        pipeline = database.Pipeline(
            origin="classification_template(imputer_cat=%s, imputer_num=%s, "
                   "encoder=%s, classifier=%s)" % (
                       imputer_cat, imputer_num, encoder, classifier))

        ds = D3MDataset(dataset)
        columns = ds.get_learning_data_columns()
        # colType one of: integer, real, string, boolean, categorical, dateTime
        categorical = [c['colName']
                       for c in columns
                       if ('attribute' in c['role'] and
                           c['colType'] not in ['integer', 'real'])]
        numerical = [c['colName']
                     for c in columns
                     if ('attribute' in c['role'] and
                         c['colType'] in ['integer', 'real'])]

        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data','0.0', name)

        def make_primitive_module(name):
            if name.startswith('sklearn'):
                return make_module('sklearn-builtin', '0.0', name)
            else:
                return make_module('primitives', '0.0', name)

        def connect(from_module, to_module,
                    from_output='data', to_input='data'):
            db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        try:
            data = make_data_module('data')
            targets = make_data_module('targets')

            # If we have to split the data for imputation
            if (categorical and numerical and
                    (imputer_cat or imputer_num or encoder)):
                # Split the data
                data_cat = make_data_module('get_columns')
                db.add(database.PipelineParameter(
                    pipeline=pipeline, module=data_cat,
                    name='columns', value=pickle.dumps(categorical),
                ))
                connect(data, data_cat)

                data_num = make_data_module('get_columns')
                db.add(database.PipelineParameter(
                    pipeline=pipeline, module=data_num,
                    name='columns', value=pickle.dumps(numerical),
                ))
                connect(data, data_num)

                # Add imputers
                if imputer_cat:
                    imputer_cat = make_primitive_module(imputer_cat)
                    connect(data_cat, imputer_cat)
                    data_cat = imputer_cat
                if imputer_num:
                    imputer_num = make_primitive_module(imputer_num)
                    connect(data_num, imputer_num)
                    data_num = imputer_num

                # Add encoder
                if encoder:
                    encoder = make_primitive_module(encoder)
                    connect(data_cat, encoder)
                    data_cat = encoder

                # Merge data
                data = make_data_module('merge_columns')
                connect(data_cat, data)
                connect(data_num, data)
            # If we don't have to split
            else:
                if categorical and (imputer_cat or encoder):
                    if imputer_cat:
                        imputer = make_primitive_module(imputer_cat)
                        connect(data, imputer)
                        data = imputer
                    if encoder:
                        encoder = make_primitive_module(encoder)
                        connect(data, encoder)
                        data = encoder
                elif numerical and imputer_num:
                    imputer = make_primitive_module(imputer_num)
                    connect(data, imputer)
                    data = imputer

            classifier = make_primitive_module(classifier)
            connect(data, classifier)
            connect(targets, classifier, 'targets', 'targets')

            db.add(pipeline)
            db.commit()
            return pipeline.id
        finally:
            db.close()

    TEMPLATES = {
        'CLASSIFICATION': list(itertools.product(
            [_classification_template],
            # Imputer for categorical data
            [None],
            # Imputer for numerical data
            [
                'dsbox.datapreprocessing.cleaner.KNNImputation',
                'sklearn.preprocessing.Imputer',
            ],
            # Encoder for categorical data
            [
                'sklearn.preprocessing.LabelBinarizer',
            ],
            # Classifier
            [
                'sklearn.svm.classes.LinearSVC',
                'sklearn.neighbors.classification.KNeighborsClassifier',
                'sklearn.naive_bayes.MultinomialNB',
                'sklearn.ensemble.forest.RandomForestClassifier',
                'sklearn.linear_model.logistic.LogisticRegression'
            ],
        )),
        'DEBUG_CLASSIFICATION': list(itertools.product(
            [_classification_template],
            # Imputer for categorical data
            [None],
            # Imputer for numerical data
            [
                'sklearn.preprocessing.Imputer',
            ],
            # Encoder for categorical data
            [
                'sklearn.preprocessing.LabelBinarizer',
            ],
            # Classifier
            [
                'sklearn.svm.classes.LinearSVC',
                'sklearn.neighbors.classification.KNeighborsClassifier',
            ],
        )),
        'REGRESSION': list(itertools.product(
            [_classification_template],
            # Imputer for categorical data
            [None],
            # Imputer for numerical data
            [
                'dsbox.datapreprocessing.cleaner.KNNImputation',
                'sklearn.preprocessing.Imputer',
            ],
            # Encoder for categorical data
            [
                'sklearn.preprocessing.LabelBinarizer',
            ],
            # Classifier
            [
                'sklearn.linear_model.base.LinearRegression',
                'sklearn.linear_model.bayes.BayesianRidge',
                'sklearn.linear_model.coordinate_descent.LassoCV',
                'sklearn.linear_model.ridge.Ridge',
                'sklearn.linear_model.least_angle.Lars',
            ],
        )),
        'DEBUG_REGRESSION': list(itertools.product(
            [_classification_template],
            # Imputer for categorical data
            [None],
            # Imputer for numerical data
            [
                'dsbox.datapreprocessing.cleaner.KNNImputation',
                'sklearn.preprocessing.Imputer',
            ],
            # Encoder for categorical data
            [
                'sklearn.preprocessing.LabelBinarizer',
            ],
            # Classifier
            [
                'sklearn.linear_model.base.LinearRegression',
            ],
        )),
    }
