"""GRPC server code, exposing D3mTa2 over the TA3-TA2 protocol.

Those adapters wrap the D3mTa2 object and handle all the GRPC and protobuf
logic, converting to/from protobuf messages. No GRPC or protobuf objects should
leave this module.
"""

import logging
from queue import Queue
from uuid import UUID

from . import __version__

import d3m_ta2_nyu.proto.core_pb2 as pb_core
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
import d3m_ta2_nyu.proto.dataflow_ext_pb2 as pb_dataflow
import d3m_ta2_nyu.proto.dataflow_ext_pb2_grpc as pb_dataflow_grpc


logger = logging.getLogger(__name__)


class CoreService(pb_core_grpc.CoreServicer):
    grpc2metric = dict((k, v) for v, k in pb_core.PerformanceMetric.items()
                       if k != pb_core.METRIC_UNDEFINED)
    metric2grpc = dict(pb_core.PerformanceMetric.items())
    grpc2task = dict((k, v) for v, k in pb_core.TaskType.items()
                     if k != pb_core.TASK_TYPE_UNDEFINED)

    def __init__(self, app):
        self._app = app

    def StartSession(self, request, context):
        version = pb_core.DESCRIPTOR.GetOptions().Extensions[
            pb_core.protocol_version]
        session_id = self._app.new_session()
        logger.info("Session started: %s (protocol version %s)",
                    session_id, version)
        return pb_core.SessionResponse(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK)
            ),
            user_agent='nyu_ta2 %s' % __version__,
            version=version,
            context=pb_core.SessionContext(session_id=str(session_id)),
        )

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
        dataset = request.dataset
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
        predict_features = request.predict_features
        max_pipelines = request.max_pipelines

        if dataset.startswith('file:///'):
            dataset = dataset[7:]

        logger.info("Got CreatePipelines request, session=%s, task=%s, "
                    "dataset=%s, metrics=%s, ",
                    session_id, task,
                    dataset, ", ".join(metrics))

        queue = Queue()
        session = self._app.sessions[session_id]
        with session.with_observer(lambda e, **kw: queue.put((e, kw))):
            self._app.build_pipelines(session_id, task, dataset, metrics)

            for msg in self._pipelinecreateresult_stream(context, queue,
                                                         session):
                yield msg

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
            for msg in self._pipelinecreateresult_stream(context, queue,
                                                         session,
                                                         pipeline_ids):
                yield msg

    def _pipelinecreateresult_stream(self, context, queue,
                                     session, pipeline_ids=None):
        if pipeline_ids is None:
            pipeline_filter = lambda p_id: True
        else:
            pipeline_filter = lambda p_id, s=set(pipeline_ids): p_id in s

        while True:
            if not context.is_active():
                logger.info("Client closed CreatePipelines stream")
                break
            event, kwargs = queue.get()
            if event == 'finish_session':
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(code=pb_core.SESSION_ENDED),
                    )
                )
                break
            elif event == 'new_pipeline':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(code=pb_core.OK),
                    ),
                    progress_info=pb_core.SUBMITTED,
                    pipeline_id=str(pipeline_id),
                    pipeline_info=pb_core.Pipeline(),
                )
            elif event == 'training_start':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(code=pb_core.OK),
                    ),
                    progress_info=pb_core.RUNNING,
                    pipeline_id=str(pipeline_id),
                )
            elif event == 'training_success':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                pipeline = session.pipelines[pipeline_id]
                scores = [
                    pb_core.Score(
                        metric=self.metric2grpc[m],
                        value=s,
                    )
                    for m, s in pipeline.scores.items()
                    if m in self.metric2grpc
                ]
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(code=pb_core.OK),
                    ),
                    progress_info=pb_core.COMPLETED,
                    pipeline_id=str(pipeline_id),
                    pipeline_info=pb_core.Pipeline(
                        scores=scores,
                    ),
                )
            elif event == 'training_error':
                pipeline_id = kwargs['pipeline_id']
                if not pipeline_filter(pipeline_id):
                    continue
                yield pb_core.PipelineCreateResult(
                    response_info=pb_core.Response(
                        status=pb_core.Status(
                            code=pb_core.ABORTED,
                            details="Pipeline execution failed",
                        ),
                    ),
                    progress_info=pb_core.ERRORED,
                    pipeline_id=str(pipeline_id),
                )
            elif event == 'done_training':
                break
            else:
                logger.error("Unexpected notification event %s",
                             event)

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

        logger.info("Got ExecutePipeline request, session=%s", session_id)

        # TODO: ExecutePipeline
        yield pb_core.PipelineExecuteResult(
            response_info=pb_core.Response(
                status=pb_core.Status(code=pb_core.OK),
            ),
            progress_info=pb_core.COMPLETED,
            pipeline_id=str(pipeline_id),
        )

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

    def GetExecutePipelineResults(self, request, context):
        raise NotImplementedError  # TODO: GetExecutePipelineResults

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
            pipeline = session.pipelines[pipeline_id]
            if not pipeline.trained:
                return pb_core.Response(
                    status=pb_core.Status(
                        code=pb_core.UNAVAILABLE,
                        details="This pipeline is not trained yet"),
                )
            uri = request.pipeline_exec_uri
            if uri.startswith('file:///'):
                uri = uri[7:]
            self._app.write_executable(pipeline, filename=uri)
        return pb_core.Response(
            status=pb_core.Status(code=pb_core.OK)
        )

    def SetProblemDoc(self, request, context):
        raise NotImplementedError  # TODO: SetProblemDoc


class DataflowService(pb_dataflow_grpc.DataflowExtServicer):
    def __init__(self, app):
        self._app = app
