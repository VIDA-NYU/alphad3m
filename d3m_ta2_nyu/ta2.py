"""The D3mTa2 class, that creates pipelines, train, and run them.

We use multiprocessing to run training in separate processes, sending messages
back to this process via a Queue.
"""

from concurrent import futures
import grpc
import itertools
import json
import logging
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

from d3m.container import Dataset

from d3m_ta2_nyu import __version__
from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA, SCORES_RANKING_ORDER, \
    TASKS_FROM_SCHEMA
from d3m_ta2_nyu.multiprocessing import Receiver, run_process
from d3m_ta2_nyu import grpc_server
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.dataflow_ext_pb2_grpc as pb_dataflow_grpc
from d3m_ta2_nyu.test import test
from d3m_ta2_nyu.utils import Observable
from d3m_ta2_nyu.workflow import database


MAX_RUNNING_PROCESSES = 1

TUNE_PIPELINES_COUNT = 3
TUNE_PIPELINES_COUNT_DEBUG = 1

TRAIN_PIPELINES_COUNT = 10
TRAIN_PIPELINES_COUNT_DEBUG = 5


logger = logging.getLogger(__name__)


class Session(Observable):
    """A session, in the GRPC meaning.

    This is a TA3 session in which pipelines are created.
    """
    def __init__(self, ta2, logs_dir, problem, DBSession):
        Observable.__init__(self)
        self.id = uuid.uuid4()
        self._ta2 = ta2
        self._logs_dir = logs_dir
        self.DBSession = DBSession
        self.problem = problem
        self.metrics = []

        # Should tuning be triggered when we are done with current pipelines?
        self._tune_when_ready = False

        # All the pipelines that belong to this session
        self.pipelines = set()
        # The pipelines currently in the queue for scoring
        self.pipelines_scoring = set()
        # The pipelines in the queue for hyperparameter tuning
        self.pipelines_tuning = set()
        # Pipelines already tuned, and pipelines created through tuning
        self.tuned_pipelines = set()
        # Pipelines in the queue for training
        self.pipelines_training = set()
        # Flag indicating we started scoring, tuning & training, and a
        # 'done_training' signal should be sent once no pipeline is pending
        self.working = False

        # Read metrics from problem
        for metric in self.problem['inputs']['performanceMetrics']:
            metric = metric['metric']
            try:
                metric = SCORES_FROM_SCHEMA[metric]
            except KeyError:
                logger.error("Unknown metric %r", metric)
                raise ValueError("Unknown metric %r" % metric)
            self.metrics.append(metric)

        self._targets = None
        self._features = None

    @property
    def problem_id(self):
        return self.problem['about']['problemID']

    @property
    def targets(self):
        if self._targets is not None:
            return set(self._targets)
        else:
            # Read targets from problem
            targets = set()
            assert len(self.problem['inputs']['data']) == 1
            for target in self.problem['inputs']['data'][0]['targets']:
                targets.add((target['resID'], target['colName']))
            return targets

    @targets.setter
    def targets(self, value):
        if value is None:
            self._targets = None
        elif isinstance(value, (set, list)):
            if not value:
                raise ValueError("Can't set targets to empty set")
            self._targets = set(value)
        else:
            raise TypeError("targets should be a set or None")

    @property
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        if value is None:
            self._features = None
        elif isinstance(value, (set, list)):
            if not value:
                raise ValueError("Can't set features to empty set")
            self._features = set(value)
        else:
            raise TypeError("features should be a set or None")

    def tune_when_ready(self):
        self._tune_when_ready = True
        self.check_status()

    def add_scoring_pipeline(self, pipeline_id):
        with self.lock:
            self.working = True
            self.pipelines.add(pipeline_id)
            self.pipelines_scoring.add(pipeline_id)

    def pipeline_scoring_done(self, pipeline_id):
        with self.lock:
            self.pipelines_scoring.discard(pipeline_id)
            self.check_status()

    def pipeline_tuning_done(self, old_pipeline_id, new_pipeline_id=None):
        with self.lock:
            self.pipelines_tuning.discard(old_pipeline_id)
            self.tuned_pipelines.add(old_pipeline_id)
            if new_pipeline_id is not None:
                self.pipelines.add(new_pipeline_id)
                self.tuned_pipelines.add(new_pipeline_id)
            self.check_status()

    def pipeline_training_done(self, pipeline_id):
        with self.lock:
            self.pipelines_training.discard(pipeline_id)
            self.check_status()

    def get_top_pipelines(self, db, metric, limit=None, only_trained=True):
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
            .options(joinedload(pipeline.modules),
                     joinedload(pipeline.connections))
            .order_by(crossval_score_order)
        )
        if only_trained:
            q = q.filter(pipeline.trained)
        if limit is not None:
            q = q.limit(limit)
        return q.all()

    def check_status(self):
        with self.lock:
            # Session is not to be finished automatically
            if not self._tune_when_ready:
                return
            # We are already done
            if not self.working:
                return
            # If pipelines are still in the queue
            if (self.pipelines_scoring or self.pipelines_tuning or
                    self.pipelines_training):
                return

            db = self.DBSession()

            # If we are out of pipelines to score, maybe submit pipelines for
            # tuning
            logger.info("Session %s: scoring done", self.id)

            tune = []
            try:
                tune_nb = TUNE_PIPELINES_COUNT
                if 'TA2_DEBUG_BE_FAST' in os.environ:
                    tune_nb = TUNE_PIPELINES_COUNT_DEBUG
                if tune_nb:
                    top_pipelines = self.get_top_pipelines(
                        db, self.metrics[0],
                        tune_nb, only_trained=False)
                    for pipeline, _ in top_pipelines:
                        if pipeline.id not in self.tuned_pipelines:
                            tune.append(pipeline.id)
            finally:
                db.close()

            if tune:
                # Found some pipelines to tune, do that
                logger.warning("Found %d pipelines to tune", len(tune))
                for pipeline_id in tune:
                    logger.info("    %s", pipeline_id)
                    self._ta2._run_queue.put(
                        TuneHyperparamsJob(self, pipeline_id)
                    )
                    self.pipelines_tuning.add(pipeline_id)
                return
            logger.info("Found no pipeline to tune")

            # If we are out of pipelines to tune, maybe submit pipelines for
            # training
            logger.info("Session %s: tuning done", self.id)

            train = []
            try:
                train_nb = TRAIN_PIPELINES_COUNT
                if 'TA2_DEBUG_BE_FAST' in os.environ:
                    train_nb = TRAIN_PIPELINES_COUNT_DEBUG
                if train_nb:
                    top_pipelines = self.get_top_pipelines(
                        db, self.metrics[0],
                        train_nb, only_trained=False)
                    for pipeline, _ in top_pipelines:
                        if not pipeline.trained:
                            train.append(pipeline.id)
            finally:
                db.close()

            if train:
                # Found some pipelines to train, do that
                logger.warning("Found %d pipelines to train", len(train))
                for pipeline_id in train:
                    logger.info("    %s", pipeline_id)
                    self._ta2._run_queue.put(
                        TrainJob(self, pipeline_id)
                    )
                    self.pipelines_training.add(pipeline_id)
                return
            logger.info("Found no pipeline to train")

            # Session is done (but new pipelines might be added later)
            self.working = False

            self.write_logs()
            self.notify('done_training')

    def write_logs(self):
        if not self.metrics:
            logger.error("Can't write logs for session, no metric is set!")
            return
        metric = self.metrics[0]

        written = 0
        db = self.DBSession()
        try:
            top_pipelines = self.get_top_pipelines(db, metric)
            logger.warning("Writing logs for %d pipelines", len(top_pipelines))
            for i, (pipeline, score) in enumerate(top_pipelines):
                logger.info("    %d) %s %s=%s origin=%s" ,
                            i + 1, pipeline.id, metric, score, pipeline.origin)
                filename = os.path.join(self._logs_dir,
                                        str(pipeline.id) + '.json')
                obj = {
                    'problem_id': self.problem_id,
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


class Job(object):
    __id_gen = 1

    def __init__(self, session):
        self.id = Job.__id_gen
        Job.__id_gen += 1
        self.session = session
        self.msg = None

    def start(self, **kwargs):
        raise NotImplementedError

    def check(self):
        if self.msg is not None:
            try:
                while True:
                    self.message(*self.msg.recv(0))
            except Empty:
                pass

        return self.poll()

    def poll(self):
        raise NotImplementedError

    def message(self, *args):
        raise NotImplementedError


class ScoreJob(Job):
    def __init__(self, session, pipeline_id, store_results=True):
        Job.__init__(self, session)
        self.pipeline_id = pipeline_id
        self.store_results = store_results
        self.results = None

    def start(self, db_filename, predictions_root, **kwargs):
        self.predictions_root = predictions_root
        logger.info("Scoring pipeline for %s "
                    "(session %s has %d pipelines left to score)",
                    self.pipeline_id, self.session.id,
                    len(self.session.pipelines_scoring))
        if self.store_results:
            self.results = os.path.join(self.predictions_root,
                                        '%s.csv' % self.pipeline_id)
        self.msg = Receiver()
        self.proc = run_process('d3m_ta2_nyu.score.score', 'score', self.msg,
                                pipeline_id=self.pipeline_id,
                                metrics=self.session.metrics,
                                targets=self.session.targets,
                                results_path=self.results,
                                db_filename=db_filename)
        self.session.notify('scoring_start', pipeline_id=self.pipeline_id)

    def poll(self):
        if self.proc.poll() is None:
            return False
        log = logger.info if self.proc.returncode == 0 else logger.error
        log("Pipeline scoring process done, returned %d (pipeline: %s)",
            self.proc.returncode, self.pipeline_id)
        if self.proc.returncode == 0:
            self.session.notify('scoring_success',
                                pipeline_id=self.pipeline_id,
                                predict_result=self.results)
        else:
            self.session.notify('scoring_error',
                                pipeline_id=self.pipeline_id)
        self.session.pipeline_scoring_done(self.pipeline_id)
        return True

    def message(self, msg, arg):
        if msg == 'progress':
            # TODO: Report progress
            logger.info("Scoring pipeline %s: %.0f%%",
                        self.pipeline_id, arg * 100)
        else:
            logger.error("Unexpected message from scoring process %s",
                         msg)


class TrainJob(Job):
    def __init__(self, session, pipeline_id):
        Job.__init__(self, session)
        self.pipeline_id = pipeline_id

    def start(self, db_filename, predictions_root, **kwargs):
        self.predictions_root = predictions_root
        logger.info("Training pipeline for %s "
                    "(session %s has %d pipelines left to train)",
                    self.pipeline_id, self.session.id,
                    len(self.session.pipelines_training))
        self.msg = Receiver()
        self.proc = run_process('d3m_ta2_nyu.train.train', 'train', self.msg,
                                pipeline_id=self.pipeline_id,
                                db_filename=db_filename)
        self.session.notify('training_start', pipeline_id=self.pipeline_id)

    def poll(self):
        if self.proc.poll() is None:
            return False
        log = logger.info if self.proc.returncode == 0 else logger.error
        log("Pipeline training process done, returned %d (pipeline: %s)",
            self.proc.returncode, self.pipeline_id)
        if self.proc.returncode == 0:
            self.session.notify('training_success',
                                pipeline_id=self.pipeline_id)
        else:
            self.session.notify('training_error',
                                pipeline_id=self.pipeline_id)
        self.session.pipeline_training_done(self.pipeline_id)
        return True

    def message(self, msg, arg):
        if msg == 'progress':
            # TODO: Report progress
            logger.info("Training pipeline %s: %.0f%%",
                        self.pipeline_id, arg * 100)
        else:
            logger.error("Unexpected message from training process %s",
                         msg)


class TuneHyperparamsJob(Job):
    def __init__(self, session, pipeline_id, store_results=True):
        Job.__init__(self, session)
        self.pipeline_id = pipeline_id
        self.store_results = store_results
        self.results = None

    def start(self, db_filename, predictions_root, **kwargs):
        self.predictions_root = predictions_root
        logger.info("Running tuning for %s "
                    "(session %s has %d pipelines left to tune)",
                    self.pipeline_id, self.session.id,
                    len(self.session.pipelines_tuning))
        if self.store_results:
            self.results = os.path.join(self.predictions_root,
                                        '%s.csv' % self.pipeline_id)
        self.msg = Receiver()
        self.proc = run_process('d3m_ta2_nyu.train_and_tune.tune',
                                'tune', self.msg,
                                pipeline_id=self.pipeline_id,
                                metrics=self.session.metrics,
                                targets=self.session.targets,
                                results_path=self.results,
                                db_filename=db_filename)
        self.session.notify('tuning_start', pipeline_id=self.pipeline_id)

    def poll(self):
        if self.proc.poll() is None:
            return False
        log = logger.info if self.proc.returncode == 0 else logger.error
        log("Pipeline tuning process done, returned %d (pipeline: %s)",
            self.proc.returncode, self.pipeline_id)
        if self.proc.returncode == 0:
            logger.info("New pipeline: %s)", self.tuned_pipeline_id)
            self.session.notify('tuning_success',
                                old_pipeline_id=self.pipeline_id,
                                new_pipeline_id=self.tuned_pipeline_id)
            self.session.notify('scoring_success',
                                pipeline_id=self.tuned_pipeline_id,
                                predict_result=self.results)
            self.session.pipeline_tuning_done(self.pipeline_id,
                                              self.tuned_pipeline_id)
        else:
            self.session.notify('tuning_error',
                                pipeline_id=self.pipeline_id)
            self.session.pipeline_tuning_done(self.pipeline_id)
        return True

    def message(self, msg, arg):
        if msg == 'progress':
            # TODO: Report progress
            logger.info("Tuning pipeline %s: %.0f%%",
                        self.pipeline_id, arg * 100)
        elif msg == 'tuned_pipeline_id':
            self.tuned_pipeline_id = arg
        else:
            logger.error("Unexpected message from tuning process %s",
                         msg)


class D3mTa2(object):
    def __init__(self, storage_root, supporting_files=None,
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
        if 'TA2_USE_TEMPLATES' in os.environ:
            logger.warning("**************************************************"
                           "****")
            logger.warning("***      Using templates instead of generator     "
                           " ***")
            logger.warning("*** If this is not wanted, unset TA2_USE_TEMPLATES"
                           " ***")
            logger.warning("**************************************************"
                           "****")
        self.default_problem = None
        self.storage = os.path.abspath(storage_root)
        if not os.path.exists(self.storage):
            os.makedirs(self.storage)
        self.predictions_root = os.path.join(self.storage, 'tmp_predictions')
        if not os.path.exists(self.predictions_root):
            os.mkdir(self.predictions_root)
        if supporting_files is not None:
            self.supporting_files = supporting_files
        else:
            self.supporting_files = os.path.join(self.storage,
                                                 'supporting_files')
        if not os.path.exists(self.supporting_files):
            os.makedirs(self.supporting_files)
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

        self.db_filename = os.path.join(self.supporting_files, 'db.sqlite3')
        self.dbengine, self.DBSession = database.connect(self.db_filename)

        self.sessions = {}
        self.executor = futures.ThreadPoolExecutor(max_workers=16)
        self._run_queue = Queue()
        self._run_thread = threading.Thread(
            target=self._pipeline_running_thread)
        self._run_thread.setDaemon(True)
        self._run_thread.start()

        logger.warning("TA2 started, version=%s", __version__)

    def run_search(self, dataset, problem_path):
        """Run the search phase: create pipelines, score and train them.

        This is called by the ``ta2_search`` executable, it is part of the
        evaluation.
        """
        if dataset[0] == '/':
            dataset = 'file://' + dataset
        # Read problem
        with open(os.path.join(problem_path, 'problemDoc.json')) as fp:
            problem = json.load(fp)
        problem_id = problem['about']['problemID']
        task = problem['about']['taskType']
        if task not in TASKS_FROM_SCHEMA:
            logger.error("Unknown task %r", task)
            sys.exit(1)
        task = TASKS_FROM_SCHEMA[task]
        if task not in ('CLASSIFICATION', 'REGRESSION'):  # TODO
            logger.error("Unsupported task %s requested", task)
            sys.exit(148)

        # Create pipelines
        session = Session(self, self.logs_root, problem, self.DBSession)
        logger.info("Dataset: %s, task: %s, metrics: %s",
                    dataset, task, ", ".join(session.metrics))
        self.sessions[session.id] = session
        queue = Queue()
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self.build_pipelines(session.id, task, dataset, session.metrics)
            while queue.get(True)[0] != 'done_training':
                pass

        db = self.DBSession()
        try:
            pipelines = session.get_top_pipelines(db, session.metrics[0])

            for pipeline, score in itertools.islice(pipelines, 20):
                self.write_executable(pipeline)
        finally:
            db.close()

    def run_pipeline(self, session_id, pipeline_id, store_results=False):
        """Score a single pipeline.

        This is used by the pipeline synthesis code.
        """

        # Get the session
        session = self.sessions[session_id]
        metric = session.metrics[0]
        logger.info("Running single pipeline, metric: %s", metric)

        # Add the pipeline to the session, score it
        queue = Queue()
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            session.add_scoring_pipeline(pipeline_id)
            self._run_queue.put(ScoreJob(session, pipeline_id,
                                         store_results=False))
            session.notify('new_pipeline', pipeline_id=pipeline_id)
            while True:
                event, kwargs = queue.get(True)
                if event == 'done_training':
                    raise RuntimeError("Never got pipeline results")
                elif (event == 'scoring_error' and
                        kwargs['pipeline_id'] == pipeline_id):
                    return None
                elif (event == 'scoring_success' and
                        kwargs['pipeline_id'] == pipeline_id):
                    break

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

    def run_test(self, dataset, problem_path, pipeline_id, results_root):
        """Run a previously trained pipeline.

        This is called by the generated executables, it is part of the
        evaluation.
        """
        logger.info("About to run test")
        with open(os.path.join(problem_path, 'problemDoc.json')) as fp:
            problem = json.load(fp)
        if not os.path.exists(results_root):
            os.makedirs(results_root)
        results_path = os.path.join(
            results_root,
            problem['expectedOutputs']['predictionsFile'])

        # Get targets from problem
        targets = set()
        for target in problem['inputs']['data'][0]['targets']:
            targets.add((target['resID'], target['colName']))

        test(pipeline_id, dataset, targets, results_path,
             db_filename=self.db_filename)

    def run_server(self, problem_path, port=None):
        """Spin up the gRPC server to receive requests from a TA3 system.

        This is called by the ``ta2_serve`` executable. It is part of the
        TA2+TA3 evaluation.
        """
        with open(os.path.join(problem_path, 'problemDoc.json')) as fp:
            problem = json.load(fp)
        self.default_problem = problem
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

    def new_session(self, problem_path=None):
        if problem_path is None:
            if self.default_problem is None:
                logger.error("Creating a session but no default problem is "
                             "set!")
            problem = self.default_problem
        else:
            with open(os.path.join(problem_path, 'problemDoc.json')) as fp:
                problem = json.load(fp)

        session = Session(self, self.logs_root, problem,
                          self.DBSession)
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

    def build_pipelines(self, session_id, task, dataset, metrics,
                        targets=None, features=None):
        if not metrics:
            raise ValueError("no metrics")
        if 'TA2_USE_TEMPLATES' in os.environ:
            self.executor.submit(self._build_pipelines_from_templates,
                                 session_id, task, dataset, metrics,
                                 targets, features)
        else:
            self.executor.submit(self._build_pipelines_with_generator,
                                 session_id, task, dataset, metrics,
                                 targets, features)

    # Runs in a worker thread from executor
    def _build_pipelines_with_generator(self, session_id, task, dataset,
                                        metrics, targets, features):
        """Generates pipelines for the session, using the generator process.
        """
        # Start AlphaD3M process
        session = self.sessions[session_id]
        with session.lock:
            session.targets = targets
            session.features = features
            if session.metrics != metrics:
                if session.metrics:
                    old = 'from %s ' % ', '.join(session.metrics)
                else:
                    old = ''
                session.metrics = metrics
                logger.info("Set metrics to %s %s(for session %s)",
                            metrics, old, session_id)

            # Force working=True so we get 'done_training' even if no pipeline
            # gets created
            session.working = True

            logger.info("Starting AlphaD3M process...")
            msg_queue = Receiver()
            proc = run_process(
                'd3m_ta2_nyu.alphad3m_edit'
                '.PipelineGenerator.generate',
                'alphad3m',
                msg_queue,
                task=task,
                dataset=dataset,
                metrics=metrics,
                problem=session.problem,
                targets=session.targets,
                features=session.features,
                db_filename=self.db_filename,
            )

        # Now we wait for pipelines to be sent over the pipe
        while proc.poll() is None:
            try:
                msg, *args = msg_queue.recv(3)
            except Empty:
                continue

            if msg == 'eval':
                pipeline_id, = args
                logger.info("Got pipeline %s from generator process",
                            pipeline_id)
                score = self.run_pipeline(session_id, pipeline_id)
                logger.info("Sending score to generator process")
                msg_queue.send(score)
            else:
                raise RuntimeError("Got unknown message from generator "
                                   "process: %r" % msg)

        logger.warning("Generator process exited with %r", proc.returncode)
        session.tune_when_ready()

    # Runs in a worker thread from executor
    def _build_pipelines_from_templates(self, session_id, task, dataset,
                                        metrics, targets, features):
        """Generates pipelines for the session, using templates.
        """
        session = self.sessions[session_id]
        with session.lock:
            session.targets = targets
            session.features = features
            if session.metrics != metrics:
                if session.metrics:
                    old = 'from %s ' % ', '.join(session.metrics)
                else:
                    old = ''
                session.metrics = metrics
                logger.info("Set metrics to %s %s(for session %s)",
                            metrics, old, session_id)

            logger.info("Creating pipelines from templates...")
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
            session.tune_when_ready()
            logger.warning("Pipeline creation completed")
            session.check_status()

    def _build_pipeline_from_template(self, session, template, dataset):
        # Create workflow from a template
        pipeline_id = template(self, dataset=dataset,
                               targets=session.targets,
                               features=session.features)

        # Add it to the session
        session.add_scoring_pipeline(pipeline_id)

        logger.info("Created pipeline %s", pipeline_id)
        self._run_queue.put(ScoreJob(session, pipeline_id))
        session.notify('new_pipeline', pipeline_id=pipeline_id)

    # Runs in a background thread
    def _pipeline_running_thread(self):
        running_jobs = {}
        while True:
            # Poll jobs, remove finished ones
            remove = []
            for job in running_jobs.values():
                if job.check():
                    remove.append(job.id)
            for job_id in remove:
                del running_jobs[job_id]

            # Start new jobs if we are under the maximum
            if len(running_jobs) < MAX_RUNNING_PROCESSES:
                try:
                    job = self._run_queue.get(False)
                except Empty:
                    pass
                else:
                    job.start(db_filename=self.db_filename,
                              predictions_root=self.predictions_root)
                    running_jobs[job.id] = job

            time.sleep(3)

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

    def test_pipeline(self, session_id, pipeline_id, dataset):
        session = self.sessions[session_id]
        if pipeline_id not in session.pipelines:
            raise KeyError("No such pipeline ID for session")

        self.executor.submit(self._test_pipeline, session, pipeline_id,
                             dataset)

    def _test_pipeline(self, session, pipeline_id, dataset):
        results = os.path.join(self.predictions_root,
                               'execute-%s.csv' % uuid.uuid4())
        proc = subprocess.Popen(
            [sys.executable,
             '-c',
             'import uuid; from d3m_ta2_nyu.test import test; '
             'test(uuid.UUID(hex=%r), %r, %r, %r, db_filename=%r)' % (
                 pipeline_id.hex, dataset, session.targets, results,
                 self.db_filename,
             )
            ]
        )
        ret = proc.wait()
        session.notify('test_done',
                       pipeline_id=pipeline_id, results_path=results,
                       success=(ret == 0))

    def _classification_template(self, imputer, classifier, dataset,
                                 targets, features):
        db = self.DBSession()

        pipeline = database.Pipeline(
            origin="classification_template(imputer=%s, classifier=%s)" % (
                       imputer, classifier),
            dataset=dataset)

        dataset = Dataset.load(dataset)

        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data','0.0', name)

        def make_primitive_module(name):
            if name[0] == '.':
                name = 'd3m.primitives' + name
            return make_module('d3m', '2018.6.5', name)

        def connect(from_module, to_module,
                    from_output='produce', to_input='inputs'):
            db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        def set_hyperparams(module, **hyperparams):
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=module,
                name='hyperparams', value=pickle.dumps(hyperparams),
            ))

        try:
            #                          data
            #                            |
            #                       -Denormalize-
            #                            |
            #                     DatasetToDataframe
            #                            |
            #                        ColumnParser
            #                       /     |     \
            #                     /       |       \
            #                   /         |         \
            # Extract (attribute)  Extract (target)  |
            #         |               |              |
            #     CastToType      CastToType         |
            #         |               |              |
            #     [imputer]           |             /
            #            \            /           /
            #             [classifier]          /
            #                       |         /
            #                   ConstructPredictions
            # TODO: Use pipeline input for this
            input_data = make_data_module('dataset')
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=input_data,
                name='targets', value=pickle.dumps(targets),
            ))
            db.add(database.PipelineParameter(
                pipeline=pipeline, module=input_data,
                name='features', value=pickle.dumps(features),
            ))

            # FIXME: Denormalize?
            #step0 = make_primitive_module('.datasets.Denormalize')
            #connect(input_data, step0, from_output='dataset')

            step1 = make_primitive_module('.datasets.DatasetToDataFrame')
            connect(input_data, step1, from_output='dataset')
            #connect(step0, step1)

            step2 = make_primitive_module('.data.ColumnParser')
            connect(step1, step2)

            step3 = make_primitive_module('.data.'
                                          'ExtractColumnsBySemanticTypes')
            set_hyperparams(
                step3,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            )
            connect(step2, step3)

            step4 = make_primitive_module('.data.CastToType')
            connect(step3, step4)

            step5 = make_primitive_module(imputer)
            connect(step4, step5)

            step6 = make_primitive_module('.data.'
                                          'ExtractColumnsBySemanticTypes')
            set_hyperparams(
                step6,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Target',
                ],
            )
            connect(step2, step6)

            step7 = make_primitive_module('.data.CastToType')
            connect(step6, step7)

            step8 = make_primitive_module(classifier)
            connect(step5, step8)
            connect(step7, step8, to_input='outputs')

            step9 = make_primitive_module('.data.ConstructPredictions')
            connect(step8, step9)
            connect(step2, step9, to_input='reference')

            db.add(pipeline)
            db.commit()
            return pipeline.id
        finally:
            db.close()

    TEMPLATES = {
        'CLASSIFICATION': list(itertools.product(
            [_classification_template],
            # Imputer
            ['d3m.primitives.sklearn_wrap.SKImputer'],
            # Classifier
            [
                'd3m.primitives.sklearn_wrap.SKLinearSVC',
                'd3m.primitives.sklearn_wrap.SKKNeighborsClassifier',
                'd3m.primitives.sklearn_wrap.SKMultinomialNB',
                'd3m.primitives.sklearn_wrap.SKRandomForestClassifier',
                'd3m.primitives.sklearn_wrap.SKLogisticRegression',
            ],
        )),
        'DEBUG_CLASSIFICATION': list(itertools.product(
            [_classification_template],
            # Imputer
            ['d3m.primitives.sklearn_wrap.SKImputer'],
            # Classifier
            [
                'd3m.primitives.sklearn_wrap.SKLinearSVC',
                'd3m.primitives.sklearn_wrap.SKKNeighborsClassifier',
            ],
        )),
        'REGRESSION': list(itertools.product(
            [_classification_template],
            # Imputer
            ['d3m.primitives.sklearn_wrap.SKImputer'],
            # Classifier
            [
                'd3m.primitives.common_primitives.LinearRegression',
                'd3m.primitives.sklearn_wrap.SKDecisionTreeRegressor',
                'd3m.primitives.sklearn_wrap.SKRandomForestRegressor',
                'd3m.primitives.sklearn_wrap.SKRidge',
                'd3m.primitives.sklearn_wrap.SKSGDRegressor',
            ],
        )),
        'DEBUG_REGRESSION': list(itertools.product(
            [_classification_template],
            # Imputer
            ['d3m.primitives.sklearn_wrap.SKImputer'],
            # Classifier
            [
                'd3m.primitives.sklearn_wrap.SKRandomForestRegressor',
                'd3m.primitives.sklearn_wrap.SKSGDRegressor',
            ],
        )),
    }
