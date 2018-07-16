"""GRPC server code, exposing D3mTa2 over the TA3-TA2 protocol.

Those adapters wrap the D3mTa2 object and handle all the GRPC and protobuf
logic, converting to/from protobuf messages. No GRPC or protobuf objects should
leave this module.
"""

import logging
import os
from queue import Queue
from uuid import UUID
import grpc
import time
from google.protobuf.timestamp_pb2 import Timestamp


from . import __version__

import d3m_ta2_nyu.proto.core_pb2 as pb_core
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.problem_pb2 as pb_problem
import d3m_ta2_nyu.proto.value_pb2 as pb_value


logger = logging.getLogger(__name__)


class CoreService(pb_core_grpc.CoreServicer):
    grpc2metric = dict((k, v) for v, k in pb_problem.PerformanceMetric.items()
                       if k != pb_problem.METRIC_UNDEFINED)
    metric2grpc = dict(pb_problem.PerformanceMetric.items())
    grpc2task = dict((k, v) for v, k in pb_problem.TaskType.items()
                     if k != pb_problem.TASK_TYPE_UNDEFINED)

    def __init__(self, app):
        self._app = app


    def _pipelinecreateresult_stream(self, context, queue,
                                     session, pipeline_ids=None):
        if pipeline_ids is None:
            pipeline_filter = lambda p_id: True
        else:
            pipeline_filter = lambda p_id, s=set(pipeline_ids): p_id in s

        while True:
            if not context.is_active():
                logger.info("Client closed GetSearchSolutionsResults stream")
                break
            event, kwargs = queue.get()
            if event == 'finish_session':
                now = time.time()
                seconds = int(now)
                nanos = int((now - seconds) * 10 ** 9)
                end = Timestamp(seconds=seconds, nanos=nanos)
                yield pb_core.GetSearchSolutionsResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.COMPLETED,
                        status='End of search solution',
                        start=session.start,
                        end=end
                    )
                )
                break
            elif event == 'new_pipeline':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                yield pb_core.GetSearchSolutionsResultsResponse(
                    done_ticks=3,
                    all_ticks=3,
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING,
                        status='New solution',
                        start=session.start
                    ), #TODO not sure if it is Pending or Running
                    solution_id=str(pipeline_id),
                )
            elif event == 'training_start':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
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
                if not pipeline_filter(pipeline_id):
                    continue
                scores = self._app.get_pipeline_scores(session.id, pipeline_id)
                scores = [pb_core.SolutionSearchScore(
                                scores=[pb_core.Score(
                                    metric=pb_problem.ProblemPerformanceMetric(metric=self.metric2grpc[m],
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

                )#TODO not sure if it is Running or Completed
            elif event == 'training_error':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
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


    def _fitsolutionresult_stream(self, context, queue,
                                     session, pipeline_ids=None):
        logger.info('fitting result for pipeline id: '+str(pipeline_ids[0]))
        if pipeline_ids is None:
            pipeline_filter = lambda p_id: True
        else:
            pipeline_filter = lambda p_id, s=set(pipeline_ids): p_id in s

        while True:
            if not context.is_active():
                logger.info("Client closed GetFitSolutionsResults stream")
                break
            event, kwargs = queue.get()
            if event == 'training_start':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                yield pb_core.GetFitSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.RUNNING
                    ),
                    fitted_solution_id=str(pipeline_id),
                )
            elif event == 'training_success':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
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
                if not pipeline_filter(pipeline_id):
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


    def _pipelinescoreresult_stream(self, context, queue, session, pipeline_ids=None):
        if pipeline_ids is None:
            pipeline_filter = lambda p_id: True
        else:
            pipeline_filter = lambda p_id, s=set(pipeline_ids): p_id in s

        while True:
            if not context.is_active():
                logger.info("Client closed ExecutePipeline stream")
                break
            event, kwargs = queue.get()
            logger.info("Event: %s Args: %s ",str(event),str(kwargs))
            if event == 'finish_session':
                yield pb_core.GetScoreSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.COMPLETED,
                    )
                )
                break
            elif event == 'test_done':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    logger.info("Continuing id: "+pipeline_id.hex)
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


    def _producesolutionresult_stream(self, context, queue, session, pipeline_ids=None):
        if pipeline_ids is None:
            pipeline_filter = lambda p_id: True
        else:
            pipeline_filter = lambda p_id, s=set(pipeline_ids): p_id in s

        while True:
            if not context.is_active():
                logger.info("Client closed ExecutePipeline stream")
                break
            event, kwargs = queue.get()
            logger.info("Event: %s Args: %s ",str(event),str(kwargs))
            if event == 'finish_session':
                yield pb_core.GetScoreSolutionResultsResponse(
                    progress=pb_core.Progress(
                        state=pb_core.COMPLETED,
                    )
                )
                break
            elif event == 'test_done':
                pipeline_id = kwargs['pipeline_id']
                predictions = kwargs.get('predict_result', None)
                if not pipeline_filter(pipeline_id):
                    logger.info("Continuing id: "+pipeline_id.hex)
                    continue

                if predictions:
                    predict_result_uri = 'file://{}'.format(predictions)
                else:
                    predict_result_uri = ''

                if kwargs['success']:
                    yield pb_core.GetProduceSolutionResultsResponse(
                        progress=pb_core.Progress(
                            state=pb_core.COMPLETED,
                        ),
                    )
                else:
                    yield pb_core.GetProduceSolutionResultsResponse(
                        progress=pb_core.Progress(
                            state=pb_core.ERRORED,
                        )
                    )
                break


    def SearchSolutions(self, request, context):
        search_id = self._app.new_session()
        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            logger.error("Dataset is not in D3M format: %s", dataset)
            # TODO check the correct error message
            '''
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.INVALID_ARGUMENT,
                        details="Dataset is not in D3M format",
                    ),
                ),
            )

            '''

            return
        task = request.problem.problem.task_type
        if task not in self.grpc2task:
            logger.error("Got unknown task %r", task)
            return
        task = self.grpc2task[task]
        if task not in ('CLASSIFICATION', 'REGRESSION'):  # TODO
            logger.error("Unsupported task %s requested", task)
            # TODO check the correct error message
            return
        metrics = request.problem.problem.performance_metrics
        if any(m.metric not in self.grpc2metric for m in metrics):
            logger.warning("Got metrics that we don't know about: %s",
                           ", ".join(m.metric for m in metrics
                                     if m.metric not in self.grpc2metric))
        metrics = [self.grpc2metric[m.metric] for m in metrics
                   if m.metric in self.grpc2metric]
        if not metrics:
            logger.error("Didn't get any metrics we know")
            return

        if not dataset.startswith('file://'):
            dataset = 'file://'+dataset

        logger.info("Got CreatePipelines request, session=%s, task=%s, "
                    "dataset=%s, metrics=%s, ",
                    search_id, task,
                    dataset, ", ".join(metrics))

        queue = Queue()
        session = self._app.sessions[search_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.build_pipelines(search_id, task, dataset, metrics)

        return pb_core.SearchSolutionsResponse(
            search_id=search_id.hex
        )

    def GetSearchSolutionsResults(self, request, context):
        # missing associated documentation comment in .proto file
        session_id = UUID(hex=request.search_id)
        if session_id not in self._app.sessions:
            return

        session = self._app.sessions[session_id]
        logger.info("Got GetSearchSolutionsResults request, search id=%s",
                    session_id)
        queue = Queue()

        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            yield from self._pipelinecreateresult_stream(context, queue,
                                                         session,
                                                         session.pipelines)

    def EndSearchSolutions(self, request, context):
        # missing associated documentation comment in .proto file
        session_id = UUID(hex=request.search_id)
        if session_id in self._app.sessions:
            self._app.finish_session(session_id)
            logger.info("Search terminated: %s", session_id)
        return pb_core.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
        # missing associated documentation comment in .proto file
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
            logger.error("Solution id not found: %s", request.solution_id)
            return


        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            logger.error("Dataset is not in D3M format: %s", dataset)
            return

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
        # missing associated documentation comment in .proto file
        logger.info("Got GetScoreSolutionResults request, request=%s",
                    request.request_id)

        pipeline_id = UUID(hex=request.request_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            logger.error("Request id not found: %s", request.request_id)
            return

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
            logger.error("Solution id not found: %s", request.solution_id)
            return

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            logger.error("Dataset is not in D3M format: %s", dataset)
            return

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        logger.info("Got FitSolution request, session=%s, dataset=%s",
                    session_id, dataset)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.fit_solution(session_id, pipeline_id)

        return pb_core.FitSolutionResponse(
            request_id=request.solution_id
        )

    def GetFitSolutionResults(self, request, context):
        # missing associated documentation comment in .proto file
        logger.info("Got GetFitSolutionResults request, request=%s",
                    request.request_id)

        pipeline_id = UUID(hex=request.request_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            logger.error("Request id not found: %s", request.request_id)
            return

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            yield from self._fitsolutionresult_stream(context, queue, session,
                                                        [pipeline_id])

    def ProduceSolution(self, request, context):
        pipeline_id = UUID(hex=request.fitted_solution_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            logger.error("Solution id not found: %s", request.fitted_solution_id)
            return

        dataset = request.inputs[0].dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            logger.error("Dataset is not in D3M format: %s", dataset)
            return

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        logger.info("Got ProduceSolution request, session=%s, dataset=%s",
                    session_id, dataset)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.test_pipeline(session_id, pipeline_id, dataset)

        return pb_core.ProduceSolutionResponse(
            request_id=request.fitted_solution_id
        )

    def GetProduceSolutionResults(self, request, context):
        # missing associated documentation comment in .proto file
        logger.info("Got GetProduceSolutionResults request, request=%s",
                    request.request_id)

        pipeline_id = UUID(hex=request.request_id)
        session_id = None
        for session_key in self._app.sessions:
            session = self._app.sessions[session_key]
            if pipeline_id in session.pipelines:
                session_id = session.id
                break
        if session_id is None:
            logger.error("Request id not found: %s", request.request_id)
            return

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            yield from self._producesolutionresult_stream(context, queue, session,
                                                        [pipeline_id])


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
            logger.error("Solution id not found: %s", request.fitted_solution_id)
            return
        session = self._app.sessions[session_id]
        with session.lock:
            pipeline = self._app.get_workflow(session_id, pipeline_id)
            if not pipeline.trained:
                logger.error("Solution not fitted: %s", request.fitted_solution_id)
                return
            self._app.write_executable(pipeline)
        return pb_core.SolutionExportResponse()



    def Hello(self, request, context):
        version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]
        user_agent = "ta2_stub %s" % __version__

        logger.info("Responding Hello! with user_agent=[%s] "
                    "and protocol version=[%s])",
                    user_agent, version)

        return pb_core.HelloResponse(
            user_agent=user_agent,
            version=version
        )

    def UpdateProblem(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListPrimitives(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


    def DescribeSolution(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')





