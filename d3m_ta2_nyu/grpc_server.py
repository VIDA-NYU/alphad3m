"""GRPC server code, exposing D3mTa2 over the TA3-TA2 protocol.

Those adapters wrap the D3mTa2 object and handle all the GRPC and protobuf
logic, converting to/from protobuf messages. No GRPC or protobuf objects should
leave this module.
"""

import calendar
import collections
import datetime
import functools
from google.protobuf.timestamp_pb2 import Timestamp
import grpc
import logging
import math
from queue import Queue
import string
from uuid import UUID

from . import __version__

from d3m_ta2_nyu.common import TASKS_FROM_SCHEMA, \
    SCORES_TO_SCHEMA, TASKS_TO_SCHEMA, TASKS_SUBTYPE_TO_SCHEMA, normalize_score
import d3m_ta2_nyu.proto.core_pb2 as pb_core
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.problem_pb2 as pb_problem
import d3m_ta2_nyu.proto.value_pb2 as pb_value


logger = logging.getLogger(__name__)


def to_timestamp(dt):
    """Converts a UTC datetime object into a gRPC Timestamp.

    :param dt: Time to convert, or None for now.
    :type dt: datetime.datetime | None
    """
    if dt is None:
        dt = datetime.datetime.utcnow()
    return Timestamp(seconds=calendar.timegm(dt.timetuple()),
                     nanos=dt.microsecond * 1000)


def _printmsg_out(msg, name):
    logger.info("< %s", name)
    for line in str(msg).splitlines():
        logger.info("< | %s", line)
    logger.info("  ------------------")


def _printmsg_in(msg, name):
    logger.info("> %s", name)
    for line in str(msg).splitlines():
        logger.info("> | %s", line)
    logger.info("  ------------------")


def _wrap(func):
    name = func.__name__

    @functools.wraps(func)
    def wrapped(self, request, context):
        _printmsg_in(request, name)
        ret = func(self, request, context)
        if isinstance(ret, collections.Iterable):
            return _wrap_stream(ret, name)
        else:
            _printmsg_out(ret, name)
            return ret

    return wrapped


def _wrap_stream(gen, name):
    for msg in gen:
        _printmsg_out(msg, name)
        yield msg


def log_service(klass):
    base, = klass.__bases__
    for name in dir(base):
        if name[0] not in string.ascii_uppercase:
            continue
        if name not in klass.__dict__:
            continue
        setattr(klass, name, _wrap(klass.__dict__[name]))
    return klass


def error(context, code, format, *args):
    message = format % args
    context.set_code(code)
    context.set_details(message)
    if code == grpc.StatusCode.NOT_FOUND:
        return KeyError(message)
    else:
        return ValueError(message)


@log_service
class CoreService(pb_core_grpc.CoreServicer):
    grpc2metric = {k: v for v, k in pb_problem.PerformanceMetric.items()
                   if k != pb_problem.METRIC_UNDEFINED}
    metric2grpc = dict(pb_problem.PerformanceMetric.items())

    grpc2task = {k: v for v, k in pb_problem.TaskType.items()
                 if k != pb_problem.TASK_TYPE_UNDEFINED}
    grpc2tasksubtype = {k: v for v, k in pb_problem.TaskSubtype.items()
                        if k != pb_problem.TASK_TYPE_UNDEFINED}

    def __init__(self, app):
        self._app = app

    def SearchSolutions(self, request, context):
        """Create a `Session` and start generating & scoring pipelines.
        """
        if len(request.inputs) > 1:
            raise error(context, grpc.StatusCode.UNIMPLEMENTED,
                        "Search with more than 1 input is not supported")
        expected_version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]
        if request.version != expected_version:
            logger.error("TA3 is using a different protocol version: %r "
                         "(us: %r)", request.version, expected_version)
        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)
        if not dataset.startswith('file://'):
            dataset = 'file://'+dataset

        problem = self._convert_problem(context, request.problem)

        timeout = request.time_bound
        if timeout > 0.5:
            timeout = timeout * 60.0  # Minutes
        else:
            timeout = None

        search_id = self._app.new_session(problem)

        queue = Queue()
        session = self._app.sessions[search_id]
        task = TASKS_FROM_SCHEMA[session.problem['about']['taskType']]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.build_pipelines(search_id,
                                      task,
                                      dataset, session.metrics,
                                      tune=0,  # FIXME: no tuning in TA3 mode
                                      timeout=timeout)

        return pb_core.SearchSolutionsResponse(
            search_id=str(search_id),
        )

    def GetSearchSolutionsResults(self, request, context):
        """Get the created pipelines and scores.
        """
        session_id = UUID(hex=request.search_id)
        if session_id not in self._app.sessions:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown search ID %r", session_id)

        session = self._app.sessions[session_id]

        def solution(pipeline_id, get_scores=True, status=None):
            if get_scores:
                scores = self._app.get_pipeline_scores(session.id, pipeline_id)
            else:
                scores = None

            progress = session.progress

            if not scores:
                return pb_core.GetSearchSolutionsResultsResponse(
                    done_ticks=progress.current,
                    all_ticks=progress.total,
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING,
                        status=status or "New solution",
                        start=to_timestamp(session.start),
                    ),
                    solution_id=str(pipeline_id),
                    internal_score=float('nan'),
                )
            else:
                if session.metrics and session.metrics[0] in scores:
                    metric = session.metrics[0]
                    internal_score = normalize_score(metric, scores[metric],
                                                     'asc')
                else:
                    internal_score = float('nan')
                scores = [
                    pb_core.Score(
                        metric=pb_problem.ProblemPerformanceMetric(
                            metric=self.metric2grpc[m],
                            k=0,
                            pos_label=''),
                        value=pb_value.Value(
                            raw=pb_value.ValueRaw(double=s)
                        ),
                    )
                    for m, s in scores.items()
                    if m in self.metric2grpc
                ]
                scores = [pb_core.SolutionSearchScore(scores=scores)]
                return pb_core.GetSearchSolutionsResultsResponse(
                    done_ticks=progress.current,
                    all_ticks=progress.total,
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING,
                        status=status or "Solution scored",
                        start=to_timestamp(session.start),
                    ),
                    solution_id=str(pipeline_id),
                    internal_score=internal_score,
                    scores=scores,
                )

        queue = Queue()
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            # Send the solutions that already exist
            for pipeline_id in session.pipelines:
                yield solution(pipeline_id)

            if not session.working:
                return

            # Send updates by listening to notifications on session
            while True:
                if not context.is_active():
                    logger.info(
                        "Client closed GetSearchSolutionsResults stream")
                    break
                event, kwargs = queue.get()
                if event == 'finish_session' or event == 'done_searching':
                    yield pb_core.GetSearchSolutionsResultsResponse(
                        done_ticks=len(session.pipelines),
                        all_ticks=len(session.pipelines),
                        progress=pb_core.Progress(
                            state=pb_core.COMPLETED,
                            status="End of search",
                            start=to_timestamp(session.start),
                            end=to_timestamp(None),
                        ),
                        internal_score=float('nan'),
                    )
                    break
                elif event == 'new_pipeline':
                    yield solution(kwargs['pipeline_id'], get_scores=False)
                elif event == 'scoring_success':
                    pipeline_id = kwargs['pipeline_id']
                    yield solution(pipeline_id)
                elif event == 'scoring_error':
                    pipeline_id = kwargs['pipeline_id']
                    yield solution(pipeline_id, get_scores=False,
                                   status="Solution scoring failed")

    def EndSearchSolutions(self, request, context):
        """Stop the search and delete the `Session`.
        """
        session_id = UUID(hex=request.search_id)
        if session_id in self._app.sessions:
            self._app.finish_session(session_id)
            logger.info("Search terminated: %s", session_id)
        return pb_core.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
        """Stop the search without deleting the `Session`.
        """
        session_id = UUID(hex=request.search_id)
        if session_id in self._app.sessions:
            self._app.stop_session(session_id)
            logger.info("Search stopped: %s", session_id)
        return pb_core.StopSearchSolutionsResponse()

    def ScoreSolution(self, request, context):
        """Request scores for a pipeline.

        If the scores exist, return them immediately.
        """
        pipeline_id = UUID(hex=request.solution_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Unknown solution ID %s", request.solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        logger.info("Got ExecutePipeline request, session=%s, dataset=%s",
                    session_id, dataset)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            # TODO: This is not TEST, this is SCORE
            # TODO: We can actually check if a score already exists
            self._app.test_pipeline(session_id, pipeline_id, dataset)

        return pb_core.ScoreSolutionResponse(
            # TODO: Figure out an ID for this
            request_id=request.solution_id
        )

    def GetScoreSolutionResults(self, request, context):
        """Wait for the requested scores to be available.
        """
        req_pipeline_id = UUID(hex=request.request_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if req_pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown ID %r", request.request_id)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            # TODO: Find existing result, and possibly return

            while True:
                if not context.is_active():
                    logger.info("Client closed GetScoreSolutionResults stream")
                    break
                event, kwargs = queue.get()
                if event == 'finish_session':
                    yield pb_core.GetScoreSolutionResultsResponse(
                        progress=pb_core.Progress(
                            state=pb_core.COMPLETED,
                        )
                    )
                    break
                elif event == 'test_done':
                    pipeline_id = kwargs['pipeline_id']
                    if pipeline_id != req_pipeline_id:
                        continue
                    if kwargs['success']:
                        scores = self._app.get_pipeline_scores(session.id,
                                                               pipeline_id)

                        scores = [
                            pb_core.Score(
                                metric=pb_problem.ProblemPerformanceMetric(
                                    metric=self.metric2grpc[m],
                                    k=0,
                                    pos_label=''
                                ),
                                value=pb_value.Value(
                                    raw=pb_value.ValueRaw(double=s)
                                ),
                            )
                            for m, s in scores.items()
                            if m in self.metric2grpc
                        ]
                        yield pb_core.GetScoreSolutionResultsResponse(
                            progress=pb_core.Progress(
                                state=pb_core.COMPLETED,
                            ),
                            scores=scores
                        )
                    else:
                        yield pb_core.GetScoreSolutionResultsResponse(
                            progress=pb_core.Progress(
                                state=pb_core.ERRORED,
                            ),
                        )
                    break

    def FitSolution(self, request, context):
        """Train a pipeline on a dataset.

        This will make it available for testing and exporting.
        """
        pipeline_id = UUID(hex=request.solution_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown solution ID %s", request.solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.fit_solution(session_id, pipeline_id)

        return pb_core.FitSolutionResponse(
            # TODO: Figure out an ID for this
            request_id=request.solution_id
        )

    def GetFitSolutionResults(self, request, context):
        """Wait for training to be done.
        """
        req_pipeline_id = UUID(hex=request.request_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if req_pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown ID %r", request.request_id)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            # TODO: Find existing result, and possibly return

            while True:
                if not context.is_active():
                    logger.info("Client closed GetFitSolutionsResults stream")
                    break
                event, kwargs = queue.get()
                if event == 'training_start':
                    pipeline_id = kwargs['pipeline_id']
                    if pipeline_id != req_pipeline_id:
                        continue
                    yield pb_core.GetFitSolutionResultsResponse(
                        progress=pb_core.Progress(
                            state=pb_core.RUNNING
                        ),
                        fitted_solution_id=str(pipeline_id),
                    )
                elif event == 'training_success':
                    pipeline_id = kwargs['pipeline_id']
                    if pipeline_id != req_pipeline_id:
                        continue
                    yield pb_core.GetFitSolutionResultsResponse(
                        progress=pb_core.Progress(
                            state=pb_core.COMPLETED
                        ),
                        fitted_solution_id=str(pipeline_id),
                    )
                    break
                elif event == 'training_error':
                    pipeline_id = kwargs['pipeline_id']
                    if pipeline_id != req_pipeline_id:
                        continue
                    yield pb_core.GetFitSolutionResultsResponse(
                        progress=pb_core.Progress(
                            state=pb_core.ERRORED
                        ),
                        fitted_solution_id=str(pipeline_id),
                    )
                    break
                elif event == 'done_searching':
                    break

    def ProduceSolution(self, request, context):
        """Run testing from a trained pipeline.
        """
        pipeline_id = UUID(hex=request.fitted_solution_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown solution ID %r", request.fitted_solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Dataset is not in D3M format: %s", dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.test_pipeline(session_id, pipeline_id, dataset)

        return pb_core.ProduceSolutionResponse(
            # TODO: Figure out an ID for this
            request_id=request.fitted_solution_id
        )

    def GetProduceSolutionResults(self, request, context):
        """Wait for the requested test run to be done.
        """
        req_pipeline_id = UUID(hex=request.request_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if req_pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown ID %r", request.request_id)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            # TODO: Find existing result, and possibly return

            while True:
                if not context.is_active():
                    logger.info("Client closed GetProduceSolutionResults "
                                "stream")
                    break
                event, kwargs = queue.get()
                if event == 'finish_session':
                    yield pb_core.GetScoreSolutionResultsResponse(
                        progress=pb_core.Progress(
                            state=pb_core.COMPLETED,
                        ),
                    )
                    break
                elif event == 'test_done':
                    pipeline_id = kwargs['pipeline_id']
                    if pipeline_id != req_pipeline_id:
                        continue
                    if kwargs['success']:
                        scores = self._app.get_pipeline_scores(session.id,
                                                               pipeline_id)
                        scores = [
                            pb_core.Score(
                                metric=pb_problem.ProblemPerformanceMetric(
                                    metric=self.metric2grpc[m],
                                    k=0,
                                    pos_label='',
                                ),
                                value=pb_value.Value(
                                    raw=pb_value.ValueRaw(double=s)
                                ),
                            )
                            for m, s in scores.items()
                            if m in self.metric2grpc
                        ]
                        yield pb_core.GetScoreSolutionResultsResponse(
                            progress=pb_core.Progress(
                                state=pb_core.COMPLETED,
                            ),
                            scores=scores,
                        )
                    else:
                        yield pb_core.GetScoreSolutionResultsResponse(
                            progress=pb_core.Progress(
                                state=pb_core.ERRORED,
                            ),
                        )
                    break

    def SolutionExport(self, request, context):
        """Export a trained pipeline as an executable.
        """
        pipeline_id = UUID(hex=request.fitted_solution_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            with session.lock:
                if pipeline_id in session.pipelines:
                    session_id = session.id
                    break
        if session_id is None:
            raise error(context, grpc.StatusCode.NOT_FOUND,
                        "Unknown solution ID %r", request.fitted_solution_id)
        session = self._app.sessions[session_id]
        with session.lock:
            pipeline = self._app.get_workflow(session_id, pipeline_id)
            if not pipeline.trained:
                raise error(context, grpc.StatusCode.NOT_FOUND,
                            "Solution not fitted: %r",
                            request.fitted_solution_id)
            self._app.write_executable(pipeline)
        return pb_core.SolutionExportResponse()

    def Hello(self, request, context):
        version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]
        user_agent = "nyu_ta2 %s" % __version__

        return pb_core.HelloResponse(
            user_agent=user_agent,
            version=version
        )

    def ListPrimitives(self, request, context):
        raise NotImplementedError

    def DescribeSolution(self, request, context):
        raise NotImplementedError

    def _convert_problem(self, context, problem):
        """Convert the problem from the gRPC message to the JSON schema.
        """
        task = problem.problem.task_type
        if task not in self.grpc2task:
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Got unknown task %r", task)
        task = self.grpc2task[task]

        metrics = problem.problem.performance_metrics
        if any(m.metric not in self.grpc2metric for m in metrics):
            logger.warning("Got metrics that we don't know about: %s",
                           ", ".join(m.metric for m in metrics
                                     if m.metric not in self.grpc2metric))

        metrics = [{'metric': SCORES_TO_SCHEMA[self.grpc2metric[m.metric]]}
                   for m in metrics
                   if m.metric in self.grpc2metric]
        if not metrics:
            raise error(context, grpc.StatusCode.INVALID_ARGUMENT,
                        "Didn't get any metrics we know")

        return {
            'about': {
                'problemID': problem.problem.id,
                'problemVersion': problem.problem.version,
                'problemDescription': problem.problem.description,
                "taskType": TASKS_TO_SCHEMA.get(task, ''),
                "taskSubType": TASKS_SUBTYPE_TO_SCHEMA.get(
                    self.grpc2tasksubtype.get(problem.problem.task_type),
                    ''),
                "problemSchemaVersion": "3.0",
                "problemVersion": "1.0",
                "problemName": problem.problem.name,
            },
            'inputs': {
                'performanceMetrics': metrics,
                'data': [
                    {
                        'datasetID': i.dataset_id,
                        'targets': [
                            {
                                'targetIndex': t.target_index,
                                'resID': t.resource_id,
                                'colIndex': t.column_index,
                                'colName': t.column_name,

                            }
                            for t in i.targets
                        ],
                    }
                    for i in problem.inputs
                ],
            },
        }
