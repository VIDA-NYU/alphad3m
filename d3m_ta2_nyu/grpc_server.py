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

logger = logging.getLogger(__name__)


class CoreService(pb_core_grpc.CoreServicer):
    grpc2metric = dict((k, v) for v, k in pb_problem.PerformanceMetric.items()
                       if k != pb_problem.METRIC_UNDEFINED)
    metric2grpc = dict(pb_problem.PerformanceMetric.items())
    grpc2task = dict((k, v) for v, k in pb_problem.TaskType.items()
                     if k != pb_problem.TASK_TYPE_UNDEFINED)

    def __init__(self, app):
        self._app = app

    def EndSession(self, request, context):
        session_id = UUID(hex=request.session_id)
        if session_id in self._app.sessions:
            status = pb_core.OK
            self._app.finish_session(request.session_id)
            logger.info("Session terminated: %s", session_id)
        else:
            status = pb_core.SESSION_UNKNOWN
        return pb_core.Response(
            status=pb_core.Status(code=status),
        )

    def CreatePipelines(self, request, context):
        session_id = UUID(hex=request.context.session_id)
        if session_id not in self._app.sessions:
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
                )
            )
            return
        dataset = request.dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            logger.error("Dataset is not in D3M format: %s", dataset)
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.INVALID_ARGUMENT,
                        details="Dataset is not in D3M format",
                    ),
                ),
            )
            return
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
        target_features = [(f.resource_id, f.feature_name)
                           for f in request.target_features]
        predict_features = [(f.resource_id, f.feature_name)
                            for f in request.predict_features]
        max_pipelines = request.max_pipelines

        if dataset.startswith('/'):
            logger.warning("Dataset is a path, turning it into a file:// URL")
            dataset = 'file://' + dataset

        logger.info("Got CreatePipelines request, session=%s, task=%s, "
                    "dataset=%s, metrics=%s, "
                    "target_features=%r, predict_features=%r",
                    session_id, task,
                    dataset, ", ".join(metrics),
                    target_features, predict_features)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.build_pipelines(session_id, task, dataset, metrics,
                                      target_features, predict_features)
            # FIXME: predict_features now ignored

            yield from self._pipelinecreateresult_stream(context, queue,
                                                         session)

    def GetCreatePipelineResults(self, request, context):
        session_id = UUID(hex=request.context.session_id)
        if session_id not in self._app.sessions:
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
                ),
            )
            return
        pipeline_ids = [UUID(hex=i) for i in request.pipeline_ids]

        logger.info("Got GetCreatePipelineResults request, session=%s",
                    session_id)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            yield from self._pipelinecreateresult_stream(context, queue,
                                                         session,
                                                         pipeline_ids)

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
                                    metric=self.metric2grpc[m],
                                    value=s,
                                )
                                for m, s in scores.items()
                                if m in self.metric2grpc
                                ],
                         )
                    ]
                if predictions:
                    predict_result_uri = 'file://{}'.format(predictions)
                else:
                    predict_result_uri = ''

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

    def ExecutePipeline(self, request, context):
        session_id = UUID(hex=request.context.session_id)
        if session_id not in self._app.sessions:
            yield pb_core.PipelineExecuteResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
                ),
            )
            return
        pipeline_id = UUID(hex=request.pipeline_id)

        dataset = request.dataset_uri
        if not dataset.endswith('datasetDoc.json'):
            logger.error("Dataset is not in D3M format: %s", dataset)
            yield pb_core.PipelineExecuteResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.INVALID_ARGUMENT,
                        details="Dataset is not in D3M format",
                    ),
                ),
            )
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

            yield from self._pipelineexecuteresult_stream(context, queue)

    def GetExecutePipelineResults(self, request, context):
        session_id = UUID(hex=request.context.session_id)
        if session_id not in self._app.sessions:
            yield pb_core.PipelineExecuteResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
                ),
            )
            return
        pipeline_ids = [UUID(hex=i) for i in request.pipeline_ids]

        logger.info("Got GetExecutePipelineResults request, session=%s",
                    session_id)

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            yield from self._pipelineexecuteresult_stream(context, queue,
                                                          pipeline_ids)

    def _pipelineexecuteresult_stream(self, context, queue, pipeline_ids=None):
        if pipeline_ids is None:
            pipeline_filter = lambda p_id: True
        else:
            pipeline_filter = lambda p_id, s=set(pipeline_ids): p_id in s

        while True:
            if not context.is_active():
                logger.info("Client closed ExecutePipeline stream")
                break
            event, kwargs = queue.get()
            if event == 'finish_session':
                yield pb_core.PipelineExecuteResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(code=pb_core.SESSION_ENDED),
                    )
                )
                break
            elif event == 'test_done':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                results_path = kwargs['results_path']
                if kwargs['success']:
                    yield pb_core.PipelineExecuteResult(
                        response_info=pb_core.Response(
                            status=pb_core.Status(code=pb_core.OK),
                        ),
                        progress_info=pb_core.COMPLETED,
                        pipeline_id=str(pipeline_id),
                        result_uri='file://{}'.format(results_path),
                    )
                else:
                    yield pb_core.PipelineExecuteResult(
                        response_info=pb_core.Response(
                            status=pb_core.Status(code=pb_core.ABORTED),
                        ),
                        progress_info=pb_core.ERRORED,
                        pipeline_id=str(pipeline_id),
                    )
                break

    def ListPipelines(self, request, context):
        session_id = UUID(hex=request.context.session_id)
        if session_id not in self._app.sessions:
            return pb_core.PipelineListResult(
                status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
            )
        session = self._app.sessions[session_id]
        with session.lock:
            pipelines = list(session.pipelines)
        return pb_core.PipelineListResult(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK),
            ),
            pipeline_ids=[str(i) for i in pipelines],
        )

    def ExportPipeline(self, request, context):
        session_id = UUID(hex=request.context.session_id)
        if session_id not in self._app.sessions:
            return pb_core.Response(
                status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
            )
        session = self._app.sessions[session_id]
        pipeline_id = UUID(hex=request.pipeline_id)
        with session.lock:
            if pipeline_id not in session.pipelines:
                return pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.INVALID_ARGUMENT,
                        details="No such pipeline"),
                )
            pipeline = self._app.get_workflow(session_id, pipeline_id)
            if not pipeline.trained:
                return pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.UNAVAILABLE,
                        details="This pipeline is not trained yet"),
                )
            uri = request.pipeline_exec_uri
            if uri.startswith('file:///'):
                uri = uri[7:]
            if not uri:
                uri = None
            elif os.path.splitext(os.path.basename(uri))[0] != pipeline_id:
                logger.warning("Got ExportPipeline request with "
                               "pipeline_exec_uri which doesn't match the "
                               "pipeline ID! This means the executable will "
                               "not match the log file.")
                logger.warning("pipeline_id=%r pipeline_exec_uri=%r",
                               pipeline_id, request.pipeline_exec_uri)
            self._app.write_executable(pipeline, filename=uri)
        return pb_core.Response(
            status=pb_core.Status(code=pb_core.OK)
        )

    def SetProblemDoc(self, request, context):
        raise NotImplementedError  # TODO: SetProblemDoc


    def SearchSolutions(self, request, context):
        # missing associated documentation comment in .proto file
        search_id = self._app.new_session()

        # Remi said we can just consider the first one, otherwise we can just crash
        problem_input = request.problem.inputs[0]

        dataset = problem_input.dataset_id
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
        dataset = dataset[:-15]
        task = request.problem.problem.task_type
        if task not in self.grpc2task:
            logger.error("Got unknown task %r", task)
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
        task = self.grpc2task[task]
        if task not in ('CLASSIFICATION', 'REGRESSION'):  # TODO
            logger.error("Unsupported task %s requested", task)
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
        metrics = request.problem.problem.performance_metrics.metrics
        if any(m not in self.grpc2metric for m in metrics):
            logger.warning("Got metrics that we don't know about: %s",
                           ", ".join(m for m in metrics
                                     if m not in self.grpc2metric))
        metrics = [self.grpc2metric[m] for m in metrics
                   if m in self.grpc2metric]
        if not metrics:
            logger.error("Didn't get any metrics we know")
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
        target_features = request.target_features
        max_pipelines = request.max_pipelines

        if dataset.startswith('file:///'):
            dataset = dataset[7:]

        logger.info("Got CreatePipelines request, session=%s, task=%s, "
                    "dataset=%s, metrics=%s, ",
                    search_id, task,
                    dataset, ", ".join(metrics))

        queue = Queue()
        session = self._app.sessions[search_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.build_pipelines(search_id, task, dataset, metrics)

        return pb_core.SearchSolutionsResponse(
            search_id=search_id
        )

    def GetSearchSolutionsResults(self, request, context):
        # missing associated documentation comment in .proto file
        session_id = UUID(hex=request.search_id)
        if session_id not in self._app.sessions:
            # TODO check best response
            '''
            yield pb_core.PipelineCreateResult(
                response_info=pb_core.Response(
                    status=pb_core.Status(code=pb_core.SESSION_UNKNOWN),
                ),
            )
            '''
            return

        session = self._app.sessions[session_id]

        pipeline_ids = [UUID(hex=i) for i in session.pipelines]

        logger.info("Got GetSearchSolutionsResults request, search id=%s",
                    session_id)

        queue = Queue()

        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            yield from self._pipelinecreateresult_stream(context, queue,
                                                         session,
                                                         pipeline_ids)

    def EndSearchSolutions(self, request, context):
        # missing associated documentation comment in .proto file
        session_id = UUID(hex=request.search_id)
        if session_id in self._app.sessions:
            self._app.finish_session(session_id)
            logger.info("Search terminated: %s", session_id)
        return pb_core.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
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

    def ScoreSolution(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetScoreSolutionResults(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def FitSolution(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetFitSolutionResults(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ProduceSolution(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetProduceSolutionResults(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SolutionExport(self, request, context):
        # missing associated documentation comment in .proto file
        pass
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

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

    def Hello(self, request, context):
        version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]
        user_agent = "nyu_ta2 %s" % __version__

        logger.info("Responding Hello! with user_agent=[%s] "
                    "and protocol version=[%s])",
                    user_agent, version)

        return pb_core.HelloResponse(
            user_agent=user_agent,
            version=version
        )


class DataflowService(pb_dataflow_grpc.DataflowExtServicer):
    def __init__(self, app):
        self._app = app

