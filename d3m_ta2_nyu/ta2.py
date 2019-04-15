"""The D3mTa2 class, that creates pipelines, train, and run them.

We use multiprocessing to run training in separate processes, sending messages
back to this process via a Queue.
"""

from concurrent import futures
import datetime
import grpc
import itertools
import json
import logging
import os
import pickle
from queue import Empty, Queue
from sqlalchemy import select
from sqlalchemy.orm import aliased, joinedload, lazyload
from sqlalchemy.sql import func
import stat
import subprocess
import sys
import threading
import time
from uuid import uuid4, UUID
from d3m_ta2_nyu import __version__
from d3m_ta2_nyu.common import SCORES_FROM_SCHEMA, SCORES_RANKING_ORDER, \
    TASKS_FROM_SCHEMA, normalize_score
from d3m_ta2_nyu.multiprocessing import Receiver, run_process
from d3m_ta2_nyu import grpc_server
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
from d3m_ta2_nyu.utils import Observable, ProgressStatus
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.convert import to_d3m_json


MAX_RUNNING_PROCESSES = 1

TUNE_PIPELINES_COUNT = 0
if 'TA2_DEBUG_BE_FAST' in os.environ:
    TUNE_PIPELINES_COUNT = 0

TRAIN_PIPELINES_COUNT = 0
TRAIN_PIPELINES_COUNT_DEBUG = 5


logger = logging.getLogger(__name__)


class Session(Observable):
    """A session, in the gRPC meaning.

    This corresponds to a search in which pipelines are created.
    """
    def __init__(self, ta2, problem, DBSession, searched_pipelines_dir, scored_pipelines_dir, ranked_pipelines_dir):
        Observable.__init__(self)
        self.id = uuid4()
        self._ta2 = ta2
        self.problem = problem
        self.DBSession = DBSession
        self._searched_pipelines_dir = searched_pipelines_dir
        self._scored_pipelines_dir = scored_pipelines_dir
        self._ranked_pipelines_dir = ranked_pipelines_dir
        self.metrics = []

        self._observer = self._ta2.add_observer(self._ta2_event)

        self.start = datetime.datetime.utcnow()

        # Should tuning be triggered when we are done with current pipelines?
        self._tune_when_ready = None

        # All the pipelines that belong to this session
        self.pipelines = set()
        # The pipelines currently in the queue for scoring
        self.pipelines_scoring = set()
        # The pipelines in the queue for hyperparameter tuning
        self.pipelines_tuning = set()
        # Pipelines already tuned, and pipelines created through tuning
        self.tuned_pipelines = set()
        # Flag indicating we started scoring & tuning, and a
        # 'done_searching' signal should be sent once no pipeline is pending
        self.working = False
        # Flag allowing TA3 to stop the search early
        self.stop_requested = False

        # Read metrics from problem
        if self.problem is not None:
            for metric in self.problem['inputs']['performanceMetrics']:
                metric_name = metric['metric']
                try:
                    metric_name = SCORES_FROM_SCHEMA[metric_name]
                except KeyError:
                    logger.error("Unknown metric %r", metric_name)
                    raise ValueError("Unknown metric %r" % metric_name)

                formatted_metric = {'metric': metric_name}

                if len(metric) > 1:  # Metric has parameters
                    formatted_metric['params'] = {}
                    for param in metric.keys():
                        if param != 'metric':
                            formatted_metric['params'][param] = metric[param]

                self.metrics.append(formatted_metric)

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

    def _ta2_event(self, event, **kwargs):
        if event == 'scoring_start':
            if kwargs['pipeline_id'] in self.pipelines_scoring:
                logger.info("Scoring pipeline for %s (session %s has %d "
                            "pipelines left to score)",
                            kwargs['pipeline_id'], self.id,
                            len(self.pipelines_scoring))
                self.notify(event, **kwargs)
        elif event == 'scoring_success' or event == 'scoring_error':
            if kwargs['pipeline_id'] in self.pipelines_scoring:
                self.notify(event, **kwargs)
                self.pipeline_scoring_done(kwargs['pipeline_id'], event)

    def tune_when_ready(self, tune=None):
        if tune is None:
            tune = TUNE_PIPELINES_COUNT
        self._tune_when_ready = tune
        self.working = True
        self.check_status()

    def add_scoring_pipeline(self, pipeline_id):
        with self.lock:
            self.working = True
            self.pipelines.add(pipeline_id)
            self.pipelines_scoring.add(pipeline_id)
            self.write_searched_pipeline(pipeline_id)

    def pipeline_scoring_done(self, pipeline_id, event=None):
        with self.lock:
            self.pipelines_scoring.discard(pipeline_id)
            self.check_status()
            if event == 'scoring_success':
                self.write_scored_pipeline(pipeline_id)

    def pipeline_tuning_done(self, old_pipeline_id, new_pipeline_id=None):
        with self.lock:
            self.pipelines_tuning.discard(old_pipeline_id)
            self.tuned_pipelines.add(old_pipeline_id)
            if new_pipeline_id is not None:
                self.pipelines.add(new_pipeline_id)
                self.tuned_pipelines.add(new_pipeline_id)
                #self.write_searched_pipeline(new_pipeline_id)  # I'm not sure it should be here.
            self.check_status()

    @property
    def progress(self):
        if self._tune_when_ready is not None:
            to_tune = self._tune_when_ready - len(self.tuned_pipelines) / 2
        else:
            to_tune = 0
        return ProgressStatus(
            current=len(self.pipelines) - len(self.pipelines_scoring),
            total=len(self.pipelines) + to_tune,
        )

    def get_top_pipelines(self, db, metric, limit=None):
        pipeline = aliased(database.Pipeline)
        crossval_score = (
            select([func.avg(database.CrossValidationScore.value)])
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
            .filter(crossval_score != None)
            # FIXME: Using a joined load here results in duplicated results
            .options(lazyload(pipeline.parameters))
            .order_by(crossval_score_order)
        )
        if limit is not None:
            q = q.limit(limit)
        return q.all()

    def check_status(self):
        with self.lock:
            # Session is not to be finished automatically
            if self._tune_when_ready is None:
                return
            # We are already done
            if not self.working:
                return
            # If pipelines are still in the queue
            if self.pipelines_scoring or self.pipelines_tuning:
                return

            db = self.DBSession()

            # If we are out of pipelines to score, maybe submit pipelines for
            # tuning
            logger.info("Session %s: scoring done", self.id)

            tune = []
            if self._tune_when_ready and self.stop_requested:
                logger.info("Session stop requested, skipping tuning")
            elif self._tune_when_ready:
                top_pipelines = self.get_top_pipelines(
                    db, self.metrics[0]['metric'],
                    self._tune_when_ready)
                for pipeline, _ in top_pipelines:
                    if pipeline.id not in self.tuned_pipelines:
                        tune.append(pipeline.id)

                if tune:
                    # Found some pipelines to tune, do that
                    logger.warning("Found %d pipelines to tune", len(tune))
                    for pipeline_id in tune:
                        logger.info("    %s", pipeline_id)
                        self._ta2._run_queue.put(
                            TuneHyperparamsJob(self, pipeline_id, self.problem)
                        )
                        self.pipelines_tuning.add(pipeline_id)
                    return
                logger.info("Found no pipeline to tune")
            else:
                logger.info("No tuning requested")

            # Session is done (but new pipelines might be added later)
            self.working = False
            self.notify('done_searching')

            logger.warning("Search done")
            if self.metrics:
                metric = self.metrics[0]['metric']
                top_pipelines = self.get_top_pipelines(db, metric)
                logger.warning("Found %d pipelines", len(top_pipelines))

                for i, (pipeline, score) in enumerate(top_pipelines):
                    created = pipeline.created_date - self.start
                    logger.info("    %d) %s %s=%s origin=%s time=%.2fs",
                                i + 1, pipeline.id, metric, score,
                                pipeline.origin, created.total_seconds())

            db.close()

    def write_searched_pipeline(self, pipeline_id):
        if not self._searched_pipelines_dir:
            logger.info("Not writing log file")
            return

        db = self.DBSession()
        try:
            # Get pipeline
            pipeline = db.query(database.Pipeline).get(pipeline_id)

            logger.warning("Writing searched_pipeline JSON for pipeline %s "
                           "origin=%s",
                           pipeline_id, pipeline.origin)

            filename = os.path.join(self._searched_pipelines_dir,
                                    '%s.json' % pipeline_id)
            obj = to_d3m_json(pipeline)
            with open(filename, 'w') as fp:
                json.dump(obj, fp, indent=2)
        except Exception:
            logger.exception("Error writing searched_pipeline for %s",
                             pipeline_id)
        finally:
            db.close()

    def write_scored_pipeline(self, pipeline_id):
        if not self._scored_pipelines_dir:
            logger.info("Not writing log file")
            return

        db = self.DBSession()
        try:
            # Get pipeline
            pipeline = db.query(database.Pipeline).get(pipeline_id)

            logger.warning("Writing scored_pipeline JSON for pipeline %s "
                           "origin=%s",
                           pipeline_id, pipeline.origin)

            filename = os.path.join(self._scored_pipelines_dir,
                                    '%s.json' % pipeline_id)
            obj = to_d3m_json(pipeline)
            with open(filename, 'w') as fp:
                json.dump(obj, fp, indent=2)
        except Exception:
            logger.exception("Error writing scored_pipeline for %s",
                             pipeline_id)
        finally:
            db.close()

    def write_exported_pipeline(self, pipeline_id, rank=None):
        metric = self.metrics[0]['metric']

        db = self.DBSession()
        try:
            # Get pipeline
            pipeline = db.query(database.Pipeline).get(pipeline_id)

            if rank is None:
                # Find most recent cross-validation
                crossval_id = (
                    select([database.CrossValidation.id])
                    .where(database.CrossValidation.pipeline_id == pipeline_id)
                    .order_by(database.CrossValidation.date.desc())
                ).as_scalar()
                # Get score from that cross-validation
                score = db.query(
                    select([func.avg(database.CrossValidationScore.value)])
                    .where(
                        database.CrossValidationScore.cross_validation_id ==
                        crossval_id
                    )
                    .where(database.CrossValidationScore.metric == metric)
                    .as_scalar()
                )
                if score is None:
                    rank = 1000.0
                    logger.error("Writing pipeline JSON for pipeline %s, but "
                                 "it is not scored for %s. Rank set to %s. "
                                 "origin=%s",
                                 pipeline_id, metric, rank, pipeline.origin)
                else:
                    logger.warning("Writing pipeline JSON for pipeline %s "
                                   "%s=%s origin=%s",
                                   pipeline_id, metric, score.value,
                                   pipeline.origin)
                    rank = normalize_score(metric, score.value, 'desc')
            else:
                logger.warning("Writing pipeline JSON for pipeline %s with "
                               "provided rank %s. origin=%s",
                               pipeline_id, rank, pipeline.origin)

            filename = os.path.join(self._ranked_pipelines_dir, '%s.json' % pipeline_id)
            obj = to_d3m_json(pipeline)
            obj['pipeline_rank'] = rank
            with open(filename, 'w') as fp:
                json.dump(obj, fp, indent=2)
        finally:
            db.close()

    def close(self):
        self._ta2.remove_observer(self._observer)
        self._observer = None
        self.stop_requested = True
        self.notify('finish_session')


class Job(object):
    def __init__(self):
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
    timeout = 8 * 60

    def __init__(self, ta2, pipeline_id, dataset, metrics, problem, scoring_conf=None):
        Job.__init__(self)
        self.ta2 = ta2
        self.pipeline_id = pipeline_id
        self.dataset = dataset
        self.metrics = metrics
        self.problem = problem
        self.scoring_conf = scoring_conf

    def start(self, db_filename, **kwargs):
        self.msg = Receiver()
        self.proc = run_process('d3m_ta2_nyu.score.score', 'score', self.msg,
                                pipeline_id=self.pipeline_id,
                                dataset=self.dataset,
                                metrics=self.metrics,
                                problem=self.problem,
                                scoring_conf=self.scoring_conf,
                                db_filename=db_filename)
        self.started = time.time()
        self.ta2.notify('scoring_start',
                        pipeline_id=self.pipeline_id,
                        job_id=id(self))

    def poll(self):
        if self.proc.poll() is None:
            return False
        if self.started + self.timeout < time.time():
            logger.error("Scoring process is stuck, terminating after %d "
                         "seconds", time.time() - self.started)
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        log = logger.info if self.proc.returncode == 0 else logger.error
        log("Pipeline scoring process done, returned %d (pipeline: %s)",
            self.proc.returncode, self.pipeline_id)
        if self.proc.returncode == 0:
            self.ta2.notify('scoring_success',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self))
        else:
            self.ta2.notify('scoring_error',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self))
        return True


class TrainJob(Job):
    def __init__(self, ta2, pipeline_id, dataset, problem):
        Job.__init__(self)
        self.ta2 = ta2
        self.pipeline_id = pipeline_id
        self.dataset = dataset
        self.problem = problem

    def start(self, db_filename, **kwargs):
        logger.info("Training pipeline for %s", self.pipeline_id)
        self.msg = Receiver()
        self.proc = run_process('d3m_ta2_nyu.train.train', 'train', self.msg,
                                pipeline_id=self.pipeline_id,
                                dataset=self.dataset,
                                problem=self.problem,
                                storage_dir=self.ta2.storage_root,
                                results_path=os.path.join(self.ta2.predictions_root,
                                                          'fit_%s.csv' % UUID(int=id(self))),
                                db_filename=db_filename)
        self.ta2.notify('training_start',
                        pipeline_id=self.pipeline_id,
                        job_id=id(self))

    def poll(self):
        if self.proc.poll() is None:
            return False
        log = logger.info if self.proc.returncode == 0 else logger.error
        log("Pipeline training process done, returned %d (pipeline: %s)",
            self.proc.returncode, self.pipeline_id)
        if self.proc.returncode == 0:
            self.ta2.notify('training_success',
                            pipeline_id=self.pipeline_id,
                            results_path=os.path.join(self.ta2.predictions_root,
                                                      'fit_%s.csv' % UUID(int=id(self))),
                            job_id=id(self))
        else:
            self.ta2.notify('training_error',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self))
        return True


class TestJob(Job):
    def __init__(self, ta2, pipeline_id, dataset):
        Job.__init__(self)
        self.ta2 = ta2
        self.pipeline_id = pipeline_id
        self.dataset = dataset

    def start(self, db_filename, **kwargs):
        logger.info("Testing pipeline for %s", self.pipeline_id)
        self.msg = Receiver()
        self.proc = run_process('d3m_ta2_nyu.test.test', 'test', self.msg,
                                pipeline_id=self.pipeline_id,
                                dataset=self.dataset,
                                storage_dir=self.ta2.storage_root,
                                results_path=os.path.join(self.ta2.predictions_root,
                                                          'predictions_%s.csv' % UUID(int=id(self))),
                                db_filename=db_filename)
        self.ta2.notify('testing_start',
                        pipeline_id=self.pipeline_id,
                        job_id=id(self))

    def poll(self):
        if self.proc.poll() is None:
            return False
        log = logger.info if self.proc.returncode == 0 else logger.error
        log("Pipeline testing process done, returned %d (pipeline: %s)",
            self.proc.returncode, self.pipeline_id)
        if self.proc.returncode == 0:
            self.ta2.notify('testing_success',
                            pipeline_id=self.pipeline_id,
                            results_path=os.path.join(self.ta2.predictions_root,
                                                      'predictions_%s.csv' % UUID(int=id(self))),
                            job_id=id(self))
        else:
            self.ta2.notify('testing_error',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self))
        return True


class TuneHyperparamsJob(Job):
    def __init__(self, session, pipeline_id, problem, store_results=True):
        Job.__init__(self)
        self.session = session
        self.pipeline_id = pipeline_id
        self.problem = problem
        self.store_results = store_results
        self.results = None

    def start(self, db_filename, predictions_root, **kwargs):
        self.predictions_root = predictions_root
        logger.info("Running tuning for %s "
                    "(session %s has %d pipelines left to tune)",
                    self.pipeline_id, self.session.id,
                    len(self.session.pipelines_tuning))
        if self.store_results and self.predictions_root is not None:
            subdir = os.path.join(self.predictions_root, str(self.pipeline_id))
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            self.results = os.path.join(subdir, 'predictions.csv')
        self.msg = Receiver()
        self.proc = run_process('d3m_ta2_nyu.tune_and_score.tune',
                                'tune', self.msg,
                                pipeline_id=self.pipeline_id,
                                metrics=self.session.metrics,
                                targets=self.session.targets,
                                problem=self.problem,
                                results_path=self.results,
                                db_filename=db_filename)
        self.session.notify('tuning_start',
                            pipeline_id=self.pipeline_id,
                            job_id=id(self))

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
                                new_pipeline_id=self.tuned_pipeline_id,
                                job_id=id(self))
            self.session.notify('scoring_success',
                                pipeline_id=self.tuned_pipeline_id,
                                predict_result=self.results,
                                job_id=id(self))
            self.session.pipeline_tuning_done(self.pipeline_id,
                                              self.tuned_pipeline_id)
        else:
            self.session.notify('tuning_error',
                                pipeline_id=self.pipeline_id,
                                job_id=id(self))
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


class ThreadPoolExecutor(futures.ThreadPoolExecutor):
    def submit(self, fn, *args, **kwargs):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                logger.exception("Exception in worker thread")
                raise
        return futures.ThreadPoolExecutor.submit(self, wrapper,
                                                 *args, **kwargs)


class D3mTa2(Observable):
    def __init__(self, storage_root, pipelines_root=None,
                 predictions_root=None,
                 executables_root=None):
        Observable.__init__(self)
        if 'TA2_DEBUG_BE_FAST' in os.environ:
            logger.warning("**************************************************"
                           "*****")
            logger.warning("***   DEBUG mode is on, will try fewer pipelines  "
                           "  ***")
            logger.warning("*** If this is not wanted, unset $TA2_DEBUG_BE_FAS"
                           "T ***")
            logger.warning("**************************************************"
                           "*****")
        self.storage_root = storage_root
        self.pipelines_root = pipelines_root
        self.predictions_root = predictions_root
        self.executables_root = executables_root
        self.searched_pipelines = None
        self.scored_pipelines = None
        self.ranked_pipelines = None

        self.create_outputfolders(self.storage_root)

        if self.pipelines_root is not None:
            self.create_outputfolders(self.pipelines_root)
            self.searched_pipelines = os.path.join(self.pipelines_root, 'pipelines_searched')
            self.scored_pipelines = os.path.join(self.pipelines_root, 'pipelines_scored')
            self.ranked_pipelines = os.path.join(self.pipelines_root, 'pipelines_ranked')
            self.create_outputfolders(self.searched_pipelines)
            self.create_outputfolders(self.scored_pipelines)
            self.create_outputfolders(self.ranked_pipelines)

        if self.predictions_root is not None:
            self.create_outputfolders(self.predictions_root)

        if self.executables_root is not None:
            self.create_outputfolders(self.executables_root)

        self.db_filename = os.path.join(self.storage_root, 'db.sqlite3')

        logger.info("storage_root=%r, pipelines_root=%r, predictions_root=%r, "
                    "executables_root=%r",
                    self.storage_root,
                    self.pipelines_root,
                    self.predictions_root,
                    self.executables_root)

        self.dbengine, self.DBSession = database.connect(self.db_filename)

        self.sessions = {}
        self.executor = ThreadPoolExecutor(max_workers=16)
        self._run_queue = Queue()
        self._run_thread = threading.Thread(target=self._pipeline_running_thread)
        self._run_thread.setDaemon(True)
        self._run_thread.start()

        logger.warning("TA2 started, version=%s", __version__)

    def create_outputfolders(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def run_search(self, dataset, problem_path, timeout=None):
        """Run the search phase: create pipelines and score them.

        This is called by the ``ta2_search`` executable, it is part of the
        evaluation.
        """
        if dataset[0] == '/':
            dataset = 'file://' + dataset
        # Read problem
        with open(os.path.join(problem_path, 'problemDoc.json')) as fp:
            problem = json.load(fp)
        task = problem['about']['taskType']
        if task not in TASKS_FROM_SCHEMA:
            logger.error("Unknown task %r", task)
            sys.exit(1)
        task = TASKS_FROM_SCHEMA[task]

        # Create session
        session = Session(self, problem, self.DBSession,
                          self.searched_pipelines, self.scored_pipelines, self.ranked_pipelines)
        logger.info("Dataset: %s, task: %s, metrics: %s",
                    dataset, task, ", ".join([m['metric'] for m in session.metrics]))
        self.sessions[session.id] = session

        if timeout:
            # Save 2 minutes to finish scoring
            timeout = max(timeout - 2 * 60, 0.8 * timeout)

        # Create pipelines, NO TUNING

        with session.with_observer_queue() as queue:
            self.build_pipelines(session.id, task, dataset, session.metrics,
                                 tune=0, timeout=timeout)
            while queue.get(True)[0] != 'done_searching':
                pass

        logger.info("Tuning pipelines...")

        # Now do tuning, when we already have written out some solutions
        with session.with_observer_queue() as queue:
            session.tune_when_ready()
            while queue.get(True)[0] != 'done_searching':
                pass

    def train_top_pipelines(self, session, limit=20):
        db = self.DBSession()
        try:
            pipelines = session.get_top_pipelines(db, session.metrics[0]['metric'],
                                                  limit=limit)

            with self.with_observer_queue() as queue:
                training = {}
                for pipeline, score in itertools.islice(pipelines, limit):
                    if pipeline.trained:
                        continue
                    # TODO: pass problem/targets?
                    self._run_queue.put(
                        TrainJob(self, pipeline.id)
                    )
                    training[pipeline.id] = pipeline

                while training:
                    event, kwargs = queue.get(True)
                    if event == 'training_success':
                        pipeline_id = kwargs['pipeline_id']
                        session.write_exported_pipeline(
                            pipeline_id,
                            self.pipelines_exported_root
                        )
                        self.write_executable(training.pop(pipeline_id))
                    elif event == 'training_error':
                        pipeline_id = kwargs['pipeline_id']
                        del training[pipeline_id]
        finally:
            db.close()

    def search_score_pipeline(self, session_id, pipeline_id, dataset):
        """Score a single pipeline.

        This is used by the pipeline generator.
        """

        # Get the session
        session = self.sessions[session_id]
        metric = session.metrics[0]['metric']
        logger.info("Search process scoring single pipeline, metric: %s, "
                    "dataset: %s", metric, dataset)

        # Add the pipeline to the session, score it
        with session.with_observer_queue() as queue:
            session.add_scoring_pipeline(pipeline_id)
            self._run_queue.put(ScoreJob(self, pipeline_id, dataset,
                                         session.metrics, session.problem))
            session.notify('new_pipeline', pipeline_id=pipeline_id)
            while True:
                event, kwargs = queue.get(True)
                if event == 'done_searching':
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
            score = db.query(
                select([func.avg(database.CrossValidationScore.value)])
                .where(
                    database.CrossValidationScore.cross_validation_id ==
                    crossval_id
                )
                .where(database.CrossValidationScore.metric == metric)
                .as_scalar()
            )
            if score is not None:
                logger.info("Evaluation result: %s -> %r",
                            metric, score.value)
                return score
            else:
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

        mgs_queue = Receiver()
        # TODO: pass problem/targets?
        proc = run_process('d3m_ta2_nyu.test.test', 'test', mgs_queue,
                           pipeline_id=pipeline_id,
                           dataset=dataset,
                           targets=targets,
                           results_path=results_path,
                           db_filename=self.db_filename)
        ret = proc.wait()
        if ret != 0:
            raise subprocess.CalledProcessError(ret, 'd3m_ta2_nyu.test.test')

    def run_server(self, port=None):
        """Spin up the gRPC server to receive requests from a TA3 system.

        This is called by the ``ta2_serve`` executable. It is part of the
        TA2+TA3 evaluation.
        """
        if not port:
            port = 45042
        core_rpc = grpc_server.CoreService(self)
        server = grpc.server(self.executor)
        pb_core_grpc.add_CoreServicer_to_server(
            core_rpc, server)
        server.add_insecure_port('[::]:%d' % port)
        logger.info("Started gRPC server on port %d", port)
        server.start()
        while True:
            time.sleep(60)

    def new_session(self, problem):
        session = Session(self, problem, self.DBSession,
                          self.searched_pipelines, self.scored_pipelines, self.ranked_pipelines)
        self.sessions[session.id] = session
        return session.id

    def finish_session(self, session_id):
        session = self.sessions.pop(session_id)
        session.close()

    def stop_session(self, session_id):
        session = self.sessions[session_id]
        session.stop_requested = True

    def get_workflow(self, pipeline_id):
        db = self.DBSession()
        try:
            return (
                db.query(database.Pipeline)
                .filter(database.Pipeline.id == pipeline_id)
                .options(
                    joinedload(database.Pipeline.modules)
                        .joinedload(database.PipelineModule.connections_to),
                    joinedload(database.Pipeline.connections)
                )
            ).one_or_none()
        finally:
            db.close()

    def get_pipeline_scores(self, pipeline_id):
        db = self.DBSession()
        try:
            # Find most recent cross-validation
            crossval_id = (
                select([database.CrossValidation.id])
                .where(database.CrossValidation.pipeline_id == pipeline_id)
                .order_by(database.CrossValidation.date.desc())
            ).as_scalar()
            # Get scores from that cross-validation
            scores = db.query(
                select([func.avg(database.CrossValidationScore.value),
                        database.CrossValidationScore.metric])
                .where(
                    database.CrossValidationScore.cross_validation_id ==
                    crossval_id
                )
                .group_by(database.CrossValidationScore.metric)
            ).all()
            return {metric: value for value, metric in scores}
        finally:
            db.close()

    def score_pipeline(self, pipeline_id, metrics, dataset, problem, scoring_conf):
        job = ScoreJob(self, pipeline_id, dataset, metrics, problem, scoring_conf)
        self._run_queue.put(job)
        return id(job)

    def train_pipeline(self, pipeline_id, dataset, problem):
        job = TrainJob(self, pipeline_id, dataset, problem)
        self._run_queue.put(job)
        return id(job)

    def test_pipeline(self, pipeline_id, dataset):
        job = TestJob(self, pipeline_id, dataset)
        self._run_queue.put(job)
        return id(job)

    def build_pipelines(self, session_id, task, dataset, metrics,
                        targets=None, features=None, tune=None, timeout=None):
        if not metrics:
            raise ValueError("no metrics")
        self.executor.submit(self._build_pipelines,
                             session_id, task, dataset, metrics,
                             targets, features, timeout=timeout, tune=tune)

    def build_fixed_pipeline(self, session_id, pipeline):
        self.executor.submit(self._build_fixed_pipeline, session_id, pipeline)

    # Runs in a worker thread from executor
    def _build_fixed_pipeline(self, session_id, d3m_pipeline):
        session = self.sessions[session_id]

        db = self.DBSession()
        pipeline_database = database.Pipeline(origin='Fixed pipeline template', dataset='NA')
        # TODO Convert D3M pipeline to our database pipeline
        db.add(pipeline_database)
        db.commit()
        pipeline_id = pipeline_database.id
        db.close()

        logger.info("Created fixed pipeline %s", pipeline_id)

    # Runs in a worker thread from executor
    def _build_pipelines(self, session_id, task, dataset,
                         metrics, targets, features, tune=None,
                         timeout=None):
        """Generates pipelines for the session, using the generator process.
        """
        # Start AlphaD3M process
        session = self.sessions[session_id]
        with session.lock:
            session.targets = targets
            session.features = features
            if session.metrics != metrics:
                if session.metrics:
                    old = 'from %s ' % ', '.join([m['metric'] for m in session.metrics])
                else:
                    old = ''
                session.metrics = metrics
                logger.info("Set metrics to %s %s(for session %s)",
                            metrics, old, session_id)

            # Force working=True so we get 'done_searching' even if no pipeline
            # gets created
            session.working = True

        logger.info("Creating pipelines from templates...")
        template_name = task
        if 'TA2_DEBUG_BE_FAST' in os.environ:
            template_name = 'DEBUG_' + task
        for template in []:#self.TEMPLATES.get(template_name, []):
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

        if 'TA2_DEBUG_BE_FAST' not in os.environ:
            self._build_pipelines_from_generator(session, task, dataset,
                                                 metrics, timeout)

        session.tune_when_ready(tune)

    def _build_pipelines_from_generator(self, session, task, dataset,
                                        metrics, timeout=None):
        logger.info("Starting AlphaD3M process, timeout is %s", timeout)
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
            timeout=timeout,
            db_filename=self.db_filename,
        )

        start = time.time()
        stopped = False

        # Now we wait for pipelines to be sent over the pipe
        while proc.poll() is None:
            if not stopped:
                if session.stop_requested:
                    logger.error("Session stop requested, sending SIGTERM to "
                                 "generator process")
                    proc.terminate()
                    stopped = True

                if timeout is not None and time.time() > start + timeout:
                    logger.error("Reached search timeout (%d > %d seconds), "
                                 "sending SIGTERM to generator process",
                                 time.time() - start, timeout)
                    proc.terminate()
                    stopped = True

            try:
                msg, *args = msg_queue.recv(3)
            except Empty:
                continue

            if msg == 'eval':
                pipeline_id, = args
                logger.info("Got pipeline %s from generator process",
                            pipeline_id)
                score = self.run_pipeline(session, dataset, pipeline_id)

                logger.info("Sending score to generator process")
                try:  # Fixme, just to avoid Broken pipe error
                    msg_queue.send(score)
                except:
                    logger.error("Broken pipe")
                    return
            else:
                raise RuntimeError("Got unknown message from generator "
                                   "process: %r" % msg)

        logger.warning("Generator process exited with %r", proc.returncode)

    def run_pipeline(self, session, dataset, pipeline_id):

        """Score a single pipeline.

        This is used by the pipeline synthesis code.
        """

        # Add the pipeline to the session, score it
        with session.with_observer_queue() as queue:
            session.add_scoring_pipeline(pipeline_id)
            logger.info("Created pipeline %s", pipeline_id)
            self._run_queue.put(ScoreJob(self, pipeline_id, dataset,
                                         session.metrics, session.problem))
            session.notify('new_pipeline', pipeline_id=pipeline_id)

            while True:
                event, kwargs = queue.get(True)
                if event == 'done_searching':
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
            metric = session.metrics[0]['metric']
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

    def _build_pipeline_from_template(self, session, template, dataset):
        # Create workflow from a template
        pipeline_id = template(self, dataset=dataset,
                               targets=session.targets,
                               features=session.features)

        # Add it to the session
        session.add_scoring_pipeline(pipeline_id)

        logger.info("Created pipeline %s", pipeline_id)

        self._run_queue.put(ScoreJob(self, pipeline_id, dataset,
                                     session.metrics, session.problem))
        session.notify('new_pipeline', pipeline_id=pipeline_id)

    # Runs in a background thread
    def _pipeline_running_thread(self):
        running_jobs = {}
        while True:
            # Poll jobs, remove finished ones
            remove = []
            for job in running_jobs.values():
                if job.check():
                    remove.append(id(job))
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
                    running_jobs[id(job)] = job

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

    def _classification_template(self, imputer, classifier, dataset,
                                 targets, features):
        db = self.DBSession()

        pipeline = database.Pipeline(
            origin="classification_template(imputer=%s, classifier=%s)" % (
                imputer, classifier),
            dataset=dataset)

        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline,
                package=package, version=version, name=name)
            db.add(pipeline_module)
            return pipeline_module

        def make_data_module(name):
            return make_module('data', '0.0', name)

        def make_primitive_module(name):
            if name[0] == '.':
                name = 'd3m.primitives' + name
            return make_module('d3m', '2018.7.10', name)

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
            #                        Denormalize
            #                            |
            #                     DatasetToDataframe
            #                            |
            #                        ColumnParser
            #                       /     |     \
            #                     /       |       \
            #                   /         |         \
            # Extract (attribute)  Extract (target)  |
            #         |                  |        Extract (target, index)
            #     [imputer]          CastToType      |
            #         |                  |           |
            #     CastToType            /           /
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

            step0 = make_primitive_module(
                'd3m.primitives.data_transformation.denormalize.Common')
            connect(input_data, step0, from_output='dataset')

            step1 = make_primitive_module(
                'd3m.primitives.data_transformation'
                '.dataset_to_dataframe.Common')
            connect(step0, step1)

            step2 = make_primitive_module(
                'd3m.primitives.data_transformation'
                '.column_parser.DataFrameCommon')
            connect(step1, step2)

            step3 = make_primitive_module(
                'd3m.primitives.data_transformation'
                '.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(
                step3,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Attribute',
                ],
            )
            connect(step2, step3)

            step4 = make_primitive_module(imputer)
            connect(step3, step4)

            step5 = make_primitive_module(
                'd3m.primitives.data_transformation'
                '.cast_to_type.Common')
            connect(step4, step5)
            set_hyperparams(
                step5,
                type_to_cast='float',
            )

            step6 = make_primitive_module(
                'd3m.primitives.data_transformation'
                '.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(
                step6,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Target',
                ],
            )
            connect(step2, step6)

            step7 = make_primitive_module(
                'd3m.primitives.data_transformation'
                '.cast_to_type.Common')
            connect(step6, step7)

            step8 = make_primitive_module(classifier)
            connect(step5, step8)
            connect(step7, step8, to_input='outputs')

            step9 = make_primitive_module(
                'd3m.primitives.data_transformation'
                '.extract_columns_by_semantic_types.DataFrameCommon')
            set_hyperparams(
                step9,
                semantic_types=[
                    'https://metadata.datadrivendiscovery.org/types/Target',
                    ('https://metadata.datadrivendiscovery.org/types' +
                     '/PrimaryKey'),
                ],
            )
            connect(step2, step9)

            step10 = make_primitive_module(
                'd3m.primitives.data_transformation'
                '.construct_predictions.DataFrameCommon')
            connect(step8, step10)
            connect(step9, step10, to_input='reference')

            db.add(pipeline)
            db.commit()
            return pipeline.id
        finally:
            db.close()

    TEMPLATES = {
        'CLASSIFICATION': list(itertools.product(
            [_classification_template],
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.classification.random_forest.SKlearn',
                'd3m.primitives.classification.k_neighbors.SKlearn',
                'd3m.primitives.classification.bayesian_logistic_regression.Common',
                'd3m.primitives.classification.bernoulli_naive_bayes.SKlearn',
                'd3m.primitives.classification.decision_tree.SKlearn',
                'd3m.primitives.classification.gaussian_naive_bayes.SKlearn',
                'd3m.primitives.classification.gradient_boosting.SKlearn',
                'd3m.primitives.classification.linear_svc.SKlearn',
                'd3m.primitives.classification.logistic_regression.SKlearn',
                'd3m.primitives.classification.multinomial_naive_bayes.SKlearn',
                'd3m.primitives.classification.passive_aggressive.SKlearn',
                'd3m.primitives.classification.random_forest.DataFrameCommon',
                'd3m.primitives.classification.sgd.SKlearn',
            ],
        )),
        'DEBUG_CLASSIFICATION': list(itertools.product(
            [_classification_template],
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.classification.random_forest.SKlearn',
                'd3m.primitives.classification.k_neighbors.SKlearn',
            ],
        )),
        'REGRESSION': list(itertools.product(
            [_classification_template],
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.regression.random_forest.SKlearn',
                'd3m.primitives.regression.sgd.SKlearn',
                'd3m.primitives.regression.decision_tree.SKlearn',
                'd3m.primitives.regression.gaussian_process.SKlearn',
                'd3m.primitives.regression.gradient_boosting.SKlearn',
                'd3m.primitives.regression.lasso.SKlearn',
                'd3m.primitives.regression.linear_regression.Common',
                'd3m.primitives.regression.passive_aggressive.SKlearn',
            ],
        )),
        'DEBUG_REGRESSION': list(itertools.product(
            [_classification_template],
            # Imputer
            ['d3m.primitives.data_cleaning.imputer.SKlearn'],
            # Classifier
            [
                'd3m.primitives.regression.random_forest.SKlearn',
                'd3m.primitives.regression.sgd.SKlearn',
            ],
        )),
    }
