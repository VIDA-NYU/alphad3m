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
from queue import Queue
import string
from uuid import UUID

from . import __version__

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
        logger.info("< | %s", line)
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


def log_service(klass):
    base, = klass.__bases__
    for name in dir(base):
        if name[0] not in string.ascii_uppercase:
            continue
        setattr(klass, name, _wrap(klass.__dict__[name]))
    return klass


@log_service
class CoreService(pb_core_grpc.CoreServicer):
    grpc2metric = dict((k, v) for v, k in pb_problem.PerformanceMetric.items()
                       if k != pb_problem.METRIC_UNDEFINED)
    metric2grpc = dict(pb_problem.PerformanceMetric.items())
    grpc2task = dict((k, v) for v, k in pb_problem.TaskType.items()
                     if k != pb_problem.TASK_TYPE_UNDEFINED)

    def __init__(self, app):
        self._app = app

    def SearchSolutions(self, request, context):
        search_id = self._app.new_session()
        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Dataset is not in D3M format")
            raise ValueError("Dataset is not in D3M format: %s" % dataset)

        task = request.problem.problem.task_type
        if task not in self.grpc2task:
            logger.error("Got unknown task %r", task)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Got unknown task %r" % task)
            raise ValueError("Got unknown task %r" % task)
        task = self.grpc2task[task]
        metrics = request.problem.problem.performance_metrics
        if any(m.metric not in self.grpc2metric for m in metrics):
            logger.warning("Got metrics that we don't know about: %s",
                           ", ".join(m.metric for m in metrics
                                     if m.metric not in self.grpc2metric))
        metrics = [self.grpc2metric[m.metric] for m in metrics
                   if m.metric in self.grpc2metric]
        if not metrics:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Didn't get any metrics we know")
            raise ValueError("Didn't get any metrics we know")

        if not dataset.startswith('file://'):
            dataset = 'file://'+dataset

        queue = Queue()
        session = self._app.sessions[search_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.build_pipelines(search_id, task, dataset, metrics)

        return pb_core.SearchSolutionsResponse(
            search_id=search_id.hex
        )

    def GetSearchSolutionsResults(self, request, context):
        session_id = UUID(hex=request.search_id)
        if session_id not in self._app.sessions:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Unknown search")
            raise KeyError("Unknown search ID %r" % session_id)

        session = self._app.sessions[session_id]
        queue = Queue()

        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            while True:
                if not context.is_active():
                    logger.info(
                        "Client closed GetSearchSolutionsResults stream")
                    break
                event, kwargs = queue.get()
                if event == 'finish_session':
                    yield pb_core.GetSearchSolutionsResultsResponse(
                        progress=pb_core.Progress(
                            state=pb_core.COMPLETED,
                            status='End of search solution',
                            start=to_timestamp(session.start),
                            end=to_timestamp(None),
                        )
                    )
                    break
                elif event == 'new_pipeline':
                    pipeline_id = kwargs['pipeline_id']
                    yield pb_core.GetSearchSolutionsResultsResponse(
                        done_ticks=3,
                        all_ticks=3,
                        progress=pb_core.Progress(
                            state=pb_core.RUNNING,
                            status='New solution',
                            start=session.start
                        ),  # TODO not sure if it is Pending or Running
                        solution_id=str(pipeline_id),
                    )
                elif event == 'training_start':
                    pipeline_id = kwargs['pipeline_id']
                    yield pb_core.GetSearchSolutionsResultsResponse(
                        done_ticks=3,
                        all_ticks=3,
                        progress=pb_core.Progress(
                            state=pb_core.RUNNING,
                            status='Training solution',
                            start=session.start
                        ),
                        solution_id=str(pipeline_id),
                    )
                elif event == 'training_success':
                    pipeline_id = kwargs['pipeline_id']
                    predictions = kwargs.get('predict_result', None)
                    scores = self._app.get_pipeline_scores(session.id,
                                                           pipeline_id)
                    scores = [pb_core.SolutionSearchScore(
                        scores=[pb_core.Score(
                            metric=pb_problem.ProblemPerformanceMetric(
                                metric=self.metric2grpc[m],
                                k=0,
                                pos_label=''),
                            value=pb_value.Value(double=s),
                        )
                            for m, s in scores.items()
                            if m in self.metric2grpc
                        ],
                    )
                    ]
                    yield pb_core.GetSearchSolutionsResultsResponse(
                        done_ticks=3,
                        all_ticks=3,
                        progress=pb_core.Progress(
                            state=pb_core.RUNNING,
                            status='Solution trained',
                            start=session.start
                        ),
                        solution_id=str(pipeline_id),
                        scores=scores

                    )  # TODO not sure if it is Running or Completed
                elif event == 'training_error':
                    pipeline_id = kwargs['pipeline_id']
                    yield pb_core.GetSearchSolutionsResultsResponse(
                        done_ticks=3,
                        all_ticks=3,
                        progress=pb_core.Progress(
                            state=pb_core.RUNNING,
                            status='Solution training failed',
                            start=session.start
                        ),
                        solution_id=str(pipeline_id),
                    )
                elif event == 'done_training':
                    break

    def EndSearchSolutions(self, request, context):
        session_id = UUID(hex=request.search_id)
        if session_id in self._app.sessions:
            self._app.finish_session(session_id)
            logger.info("Search terminated: %s", session_id)
        return pb_core.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
        session_id = UUID(hex=request.search_id)
        if session_id in self._app.sessions:
            self._app.stop_session(session_id)
            logger.info("Search stopped: %s", session_id)
        return pb_core.StopSearchSolutionsResponse()

    def ScoreSolution(self, request, context):
        pipeline_id = UUID(hex=request.solution_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            logger.error("Solution ID not found: %s", request.solution_id)
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Solution ID not found")
            raise KeyError("Unknown solution ID %s", request.solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Dataset is not in D3M format")
            raise ValueError("Dataset is not in D3M format: %s" % dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        logger.info("Got ExecutePipeline request, session=%s, dataset=%s",
                    session_id, dataset)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.test_pipeline(session_id, pipeline_id, dataset)

        return pb_core.ScoreSolutionResponse(
            request_id=request.solution_id
        )

    def GetScoreSolutionResults(self, request, context):
        pipeline_id = UUID(hex=request.request_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Unknown ID")
            raise KeyError("Unknown ID %r" % request.request_id)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            yield from self._pipelinescoreresult_stream(context, queue,session,
                                                        [pipeline_id])

    def FitSolution(self, request, context):
        pipeline_id = UUID(hex=request.solution_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Solution ID not found")
            raise KeyError("Unknown solution ID %s" % request.solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Dataset is not in D3M format")
            raise ValueError("Dataset is not in D3M format: %s" % dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.fit_solution(session_id, pipeline_id)

        return pb_core.FitSolutionResponse(
            request_id=request.solution_id
        )

    def GetFitSolutionResults(self, request, context):
        req_pipeline_id = UUID(hex=request.request_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if req_pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Unknown ID")
            raise KeyError("Unknown ID %r" % request.request_id)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
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
                elif event == 'done_training':
                    break

    def ProduceSolution(self, request, context):
        pipeline_id = UUID(hex=request.fitted_solution_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Unknown ID")
            raise ValueError("Unknown solution ID %r" %
                             request.fitted_solution_id)

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Dataset is not in D3M format")
            raise ValueError("Dataset is not in D3M format: %s" % dataset)

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.test_pipeline(session_id, pipeline_id, dataset)

        return pb_core.ProduceSolutionResponse(
            request_id=request.fitted_solution_id
        )

    def GetProduceSolutionResults(self, request, context):
        req_pipeline_id = UUID(hex=request.request_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if req_pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Unknown ID")
            raise KeyError("Unknown ID %r" % request.request_id)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            while True:
                if not context.is_active():
                    logger.info("Client closed ExecutePipeline stream")
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
                        scores = self._app.get_pipeline_scores(session.id, pipeline_id)

                        scores=[pb_core.Score(
                                metric=pb_problem.ProblemPerformanceMetric(metric=self.metric2grpc[m],
                                                                           k=0,
                                                                           pos_label=''),
                                value=pb_value.Value(double=s),
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
                            )
                        )
                    break

    def SolutionExport(self, request, context):
        pipeline_id = UUID(hex=request.fitted_solution_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            with session.lock:
                if pipeline_id in session.pipelines:
                    session_id = session.id
                    break
        if session_id is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details("Solution ID not found")
            raise KeyError("Unknown solution ID %r" %
                           request.fitted_solution_id)
        session = self._app.sessions[session_id]
        with session.lock:
            pipeline = self._app.get_workflow(session_id, pipeline_id)
            if not pipeline.trained:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details("Solution not fitted")
                raise ValueError("Solution not fitted: %r" %
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
