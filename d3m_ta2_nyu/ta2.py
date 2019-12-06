"""The D3mTa2 class, that creates pipelines, train, and run them.

We use multiprocessing to run training in separate processes, sending messages
back to this process via a Queue.
"""

from concurrent import futures
import datetime
import grpc
import json
import logging
import os
import pickle
from queue import Empty, Queue
from sqlalchemy import select
from sqlalchemy.orm import aliased, joinedload, lazyload
from sqlalchemy.sql import func
import shutil
import subprocess
import sys
import threading
import time
from uuid import uuid4, UUID
from d3m_ta2_nyu import __version__
from d3m_ta2_nyu.common import SCORES_RANKING_ORDER, \
    TASKS_FROM_SCHEMA, normalize_score, format_metrics
from d3m_ta2_nyu.multiprocessing import Receiver, run_process
from d3m_ta2_nyu import grpc_server
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.core_pb2 as pb_core
from d3m_ta2_nyu.utils import Observable, ProgressStatus
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.convert import to_d3m_json
from sklearn.model_selection import train_test_split
import datamart
#import datamart_nyu
#from datamart_isi.entries import Datamart
from d3m.container import Dataset
from d3m.metadata.problem import TaskKeyword


MAX_RUNNING_PROCESSES = 4
SAMPLE_SIZE = 400
RANDOM_SEED = 42

DATAMART_URL = {
    'NYU': os.environ['DATAMART_URL_NYU'] if 'DATAMART_URL_NYU' in os.environ
                                          else 'https://datamart.d3m.vida-nyu.org/',
    'ISI': os.environ['DATAMART_URL_ISI'] if 'DATAMART_URL_ISI' in os.environ
                                          else 'http://dsbox02.isi.edu:9000/'
}


TUNE_PIPELINES_COUNT = 5

if 'TA2_DEBUG_BE_FAST' in os.environ:
    TUNE_PIPELINES_COUNT = 0


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
        self.do_rank = False
        self.timeout_tuning = 0

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
            self.metrics = format_metrics(problem)

        self._targets = None
        self._features = None
        self.dataset_uri = None
        self.sample_dataset_uri = None

    @property
    def problem_id(self):
        return self.problem['id']

    @property
    def targets(self):
        if self._targets is not None:
            return set(self._targets)
        else:
            # Read targets from problem
            targets = set()
            #assert len(self.problem['inputs']) == 1
            for target in self.problem['inputs'][0]['targets']:
                targets.add((target['resource_id'], target['column_name']))
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
                self.write_searched_pipeline(new_pipeline_id)
                self.write_scored_pipeline(new_pipeline_id)
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
            try:
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

                    if len(tune) > 0:
                        # Found some pipelines to tune, do that
                        logger.warning("Found %d pipelines to tune", len(tune))
                        timeout_per_pipeline = self.timeout_tuning / len(tune)
                        for pipeline_id in tune:
                            logger.info("    %s", pipeline_id)
                            self._ta2._run_queue.put(TuneHyperparamsJob(self, pipeline_id, self.problem,
                                                                        timeout=timeout_per_pipeline))
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
            finally:
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

            obj = to_d3m_json(pipeline)

            with open(os.path.join(self._ranked_pipelines_dir, '%s.json' % pipeline_id), 'w') as fout:
                json.dump(obj, fout, indent=2)
            with open(os.path.join(self._ranked_pipelines_dir, '%s.rank' % pipeline_id), 'w') as fout:
                fout.write(str(rank))

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

    def __init__(self, ta2, pipeline_id, dataset_uri, metrics, problem, scoring_config, do_rank=False,
                 sample_dataset_uri=None):
        Job.__init__(self)
        self.ta2 = ta2
        self.pipeline_id = pipeline_id
        self.dataset_uri = dataset_uri
        self.sample_dataset_uri = sample_dataset_uri
        self.metrics = metrics
        self.problem = problem
        self.scoring_config = scoring_config
        self.do_rank = do_rank

    def start(self, db_filename, **kwargs):
        self.msg = Receiver()
        self.proc = run_process('d3m_ta2_nyu.pipeline_score.score', 'score', self.msg,
                                pipeline_id=self.pipeline_id,
                                dataset_uri=self.dataset_uri,
                                sample_dataset_uri=self.sample_dataset_uri,
                                metrics=self.metrics,
                                problem=self.problem,
                                scoring_config=self.scoring_config,
                                do_rank=self.do_rank,
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
        self.proc = run_process('d3m_ta2_nyu.pipeline_train.train', 'train', self.msg,
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
        self.proc = run_process('d3m_ta2_nyu.pipeline_test.test', 'test', self.msg,
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
    def __init__(self, session, pipeline_id, problem, store_results=True, timeout=60):
        Job.__init__(self)
        self.session = session
        self.pipeline_id = pipeline_id
        self.problem = problem
        self.store_results = store_results
        self.timeout = timeout

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

        self.msg = Receiver()

        self.proc = run_process('d3m_ta2_nyu.pipeline_tune.tune',
                                'tune', self.msg,
                                pipeline_id=self.pipeline_id,
                                metrics=self.session.metrics,
                                problem=self.problem,
                                do_rank=self.session.do_rank,
                                dataset_uri=self.session.dataset_uri,
                                sample_dataset_uri=self.session.sample_dataset_uri,
                                timeout=self.timeout,
                                targets=self.session.targets,
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
        self.run_pipelines = None

        self.create_outputfolders(self.storage_root)

        if self.pipelines_root is not None:
            self.create_outputfolders(self.pipelines_root)
            self.searched_pipelines = os.path.join(self.pipelines_root, 'pipelines_searched')
            self.scored_pipelines = os.path.join(self.pipelines_root, 'pipelines_scored')
            self.ranked_pipelines = os.path.join(self.pipelines_root, 'pipelines_ranked')
            self.run_pipelines = os.path.join(self.pipelines_root, 'pipeline_runs')
            self.create_outputfolders(self.searched_pipelines)
            self.create_outputfolders(self.scored_pipelines)
            self.create_outputfolders(self.ranked_pipelines)
            self.create_outputfolders(self.run_pipelines)

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
            self.build_pipelines(session.id, task, dataset, session.metrics, tune=TUNE_PIPELINES_COUNT, timeout=timeout)
            while queue.get(True)[0] != 'done_searching':
                pass

        logger.info("Tuning pipelines...")

        # Now do tuning, when we already have written out some solutions
        with session.with_observer_queue() as queue:
            session.tune_when_ready()
            while queue.get(True)[0] != 'done_searching':
                pass

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

    def score_pipeline(self, pipeline_id, metrics, dataset_uri, problem, scoring_config):
        job = ScoreJob(self, pipeline_id, dataset_uri, metrics, problem, scoring_config)
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

    def build_pipelines(self, session_id, task, dataset, template, metrics, targets=None, features=None, timeout=None,
                        top_pipelines=0, tune=None):
        if not metrics:
            raise ValueError("no metrics")

        self.executor.submit(self._build_pipelines, session_id, task, dataset, template, metrics, targets, features, timeout,
                             top_pipelines, tune)

    def build_fixed_pipeline(self, session_id, pipeline, dataset, targets=None, features=None):
        self.executor.submit(self._build_fixed_pipeline, session_id, pipeline, dataset, targets, features)

    # Runs in a worker thread from executor
    def _build_fixed_pipeline(self, session_id, pipeline_template, dataset, targets, features):

        session = self.sessions[session_id]
        with session.lock:
            # Force working=True so we get 'done_searching' even if no pipeline
            # gets created
            session.working = True

        db = self.DBSession()

        if dataset:
            dataset_uri = dataset
        else:
            dataset_uri = 'NA'

        pipeline_database = database.Pipeline(origin='Fixed pipeline template', dataset=dataset_uri)

        # TODO: Do it on d3mpipeline_generator.py
        def make_module(package, version, name):
            pipeline_module = database.PipelineModule(
                pipeline=pipeline_database,
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
            db.add(database.PipelineConnection(pipeline=pipeline_database,
                                               from_module=from_module,
                                               to_module=to_module,
                                               from_output_name=from_output,
                                               to_input_name=to_input))

        def set_hyperparams(module, **hyperparams):
            db.add(database.PipelineParameter(
                pipeline=pipeline_database, module=module,
                name='hyperparams', value=pickle.dumps(hyperparams),
            ))

        try:
            # TODO: Use pipeline input for this
            if dataset:
                input_data = make_data_module('dataset')
                db.add(database.PipelineParameter(
                    pipeline=pipeline_database, module=input_data,
                    name='targets', value=pickle.dumps(targets),
                ))
                db.add(database.PipelineParameter(
                    pipeline=pipeline_database, module=input_data,
                    name='features', value=pickle.dumps(features),
                ))
            prev_step = None
            prev_steps = {}
            count_template_steps = 0
            for pipeline_step in pipeline_template['steps']:
                if pipeline_step['type'] == 'PRIMITIVE':
                    step = make_primitive_module(pipeline_step['primitive']['python_path'])
                    prev_steps['steps.%d.produce' % (count_template_steps)] = step
                    count_template_steps += 1
                    if 'hyperparams' in pipeline_step:
                        hyperparams = {}
                        for hyper, desc in pipeline_step['hyperparams'].items():
                            hyperparams[hyper] = desc['data']
                        set_hyperparams(step, **hyperparams)
                else:
                    # TODO In the future we should be able to handle subpipelines
                    break
                if prev_step:
                    for argument, desc in pipeline_step['arguments'].items():
                        connect(prev_steps[desc['data']], step, to_input=argument)
                else:
                    connect(input_data, step, from_output='dataset')
                prev_step = step
            db.add(pipeline_database)
            db.commit()
            pipeline_id = pipeline_database.id
            logger.info("Created fixed pipeline %s", pipeline_id)
            session.write_searched_pipeline(pipeline_id)

            session.notify('new_fixed_pipeline', pipeline_id=pipeline_id)
            with session.lock:
                session.pipelines.add(pipeline_id)
                # Force working=True so we get 'done_searching' even if no pipeline
                # gets created
                session.working = False
        finally:
            db.close()

    # Runs in a worker thread from executor
    def _build_pipelines(self, session_id, task, dataset_uri, pipeline_template, metrics, targets, features, timeout, top_pipelines, tune):
        """Generates pipelines for the session, using the generator process.
        """
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

        # Search for data augmentation
        search_results = []

        if 'dataAugmentation' in session.problem:
            dc = Dataset.load(dataset_uri)
            keywords = []
            for aug in session.problem['dataAugmentation']:
                keywords += aug['keywords']

            # Search with NYU's Datamart
            try:
                dm = datamart_nyu.RESTDatamart(DATAMART_URL['NYU'])
                cursor = dm.search_with_data(query=datamart.DatamartQuery(
                    keywords=keywords,
                    variables=[],
                ), supplied_data=dc)
                next_page = cursor.get_next_page()
            except Exception:
                logger.exception("Error when searching for data to augment")
                next_page = None

            if next_page:
                if len(next_page) > 5:
                    next_page = next_page[:5]
                search_results = [result.serialize() for result in next_page]

        sample_dataset_uri = self._get_sample_uri(dataset_uri, session.problem)
        do_rank = True if top_pipelines > 0 else False
        timeout_search = timeout  # * 0.7  # TODO: Do it dynamic
        timeout_tuning = timeout * 0.3

        self._build_pipelines_from_generator(session, task, dataset_uri, sample_dataset_uri, search_results,
                                                 pipeline_template, metrics, timeout_search, do_rank)

        # For tuning
        session.dataset_uri = dataset_uri
        session.sample_dataset_uri = sample_dataset_uri
        session.do_rank = do_rank
        session.timeout_tuning = timeout_tuning
        session.tune_when_ready(tune)

    def _build_pipelines_from_generator(self, session, task, dataset_uri, sample_dataset_uri, search_results,
                                        pipeline_template, metrics, timeout, do_rank):
        logger.info("Starting AlphaD3M process, timeout is %s", timeout)
        msg_queue = Receiver()
        proc = run_process(
            'd3m_ta2_nyu.alphad3m.interface_alphaautoml.generate',
            'alphad3m',
            msg_queue,
            task_keywords=task,
            dataset=dataset_uri,
            search_results=search_results,
            pipeline_template=pipeline_template,
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
                if stopped:
                    return
                pipeline_id, = args
                logger.info("Got pipeline %s from generator process",
                            pipeline_id)
                score = self.run_pipeline(session, dataset_uri, sample_dataset_uri, task, pipeline_id, do_rank)

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

    def run_pipeline(self, session, dataset_uri, sample_dataset_uri, task, pipeline_id, do_rank):

        """Score a single pipeline.

        This is used by the pipeline synthesis code.
        """

        scoring_config = {'shuffle': 'true',
                          'stratified': 'true' if task == 'CLASSIFICATION' else 'false',
                          'method': pb_core.EvaluationMethod.Value('K_FOLD'),
                          'folds': '2'}
        # Add the pipeline to the session, score it
        with session.with_observer_queue() as queue:
            session.add_scoring_pipeline(pipeline_id)
            logger.info("Created pipeline %s", pipeline_id)
            self._run_queue.put(ScoreJob(self, pipeline_id, dataset_uri, session.metrics, session.problem,
                                         scoring_config, do_rank, sample_dataset_uri))
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
            scores = db.query(
                select([func.avg(database.CrossValidationScore.value),
                        database.CrossValidationScore.metric])
                    .where(
                    database.CrossValidationScore.cross_validation_id ==
                    crossval_id
                )
                    .group_by(database.CrossValidationScore.metric)
            ).all()

            first_metric = session.metrics[0]['metric']
            for value, metric in scores:
                if metric == first_metric:
                    logger.info("Evaluation result: %s -> %r", metric, value)
                    return value
            logger.info("Didn't get the requested metric from cross-validation")
            return None
        finally:
            db.close()

    def _get_sample_uri(self, dataset_uri, problem):
        logger.info('About to sample dataset %s', dataset_uri)
        task_keywords = problem['problem']['task_keywords']

        if any(tk in [TaskKeyword.OBJECT_DETECTION, TaskKeyword.SEMISUPERVISED] for tk in task_keywords):
            logger.info('Not doing sampling for task %s', '_'.join(task_keywords))
            return None

        dataset = Dataset.load(dataset_uri)
        dataset_sample_folder = 'file://%s/dataset_sample/' % os.environ.get('D3MOUTPUTDIR')
        dataset_sample_uri = None

        if os.path.exists(dataset_sample_folder[6:]):
            shutil.rmtree(dataset_sample_folder[6:])

        try:
            target_name = problem['inputs'][0]['targets'][0]['column_name']
            for res_id in dataset:
                if ('https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'
                        in dataset.metadata.query([res_id])['semantic_types']):
                    break
            else:
                res_id = next(iter(dataset))

            original_size = len(dataset[res_id])
            if hasattr(dataset[res_id], 'columns') and original_size > SAMPLE_SIZE:
                labels = dataset[res_id].get(target_name)
                ratio = SAMPLE_SIZE / original_size
                stratified_labels = None
                if TaskKeyword.CLASSIFICATION in task_keywords:
                    stratified_labels = labels
                x_train, x_test, y_train, y_test = train_test_split(dataset[res_id], labels, random_state=RANDOM_SEED,
                                                                    test_size=ratio, stratify=stratified_labels)
                dataset[res_id] = x_test
                logger.info('Sampling down data from %d to %d', original_size, len(dataset[res_id]))
                dataset.save(dataset_sample_folder + 'datasetDoc.json')
                dataset_sample_uri = dataset_sample_folder + 'datasetDoc.json'
            else:
                logger.info('Not doing sampling for small dataset (size = %d)', original_size)
        except Exception:
            logger.exception('Error sampling in datatset, using whole dataset %s', dataset_uri)

        return dataset_sample_uri

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


