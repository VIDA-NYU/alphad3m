from concurrent import futures
import grpc
import logging
import time

import d3m_ta2_vistrails.proto.pipeline_service_pb2 as ps_pb2
import d3m_ta2_vistrails.proto.pipeline_service_pb2_grpc as ps_pb2_grpc


logger = logging.getLogger(__name__)


class ComputeService(ps_pb2_grpc.PipelineComputeServicer):
    def StartSession(self, request, context):
        logger.info("Session started: 1")
        return ps_pb2.Response(
            context=ps_pb2.SessionContext(session_id='1'),
            status=ps_pb2.Status(code=ps_pb2.OK),
        )

    def EndSession(self, request, context):
        assert request.session_id == '1'
        logger.info("Session terminated: 1")
        return ps_pb2.Response(
            context=ps_pb2.SessionContext(session_id=request.session_id),
            status=ps_pb2.Status(code=ps_pb2.OK),
        )

    def CreatePipelines(self, request, context):
        sessioncontext = request.context
        assert sessioncontext.session_id == '1'
        dataset_uris = request.train_dataset_uris
        task = request.task
        assert task == ps_pb2.TaskType.CLASSIFICATION
        task_description = request.task_description
        output = request.output
        metrics = request.metric
        target_features = request.target_features
        max_pipelines = request.max_pipelines

        logger.info("Got CreatePipelines request, session=%s",
                    sessioncontext.session_id)

        while True:
            yield ps_pb2.PipelineCreateResult(
                response_info=ps_pb2.Response(
                    context=sessioncontext,
                    status=ps_pb2.Status(code=ps_pb2.OK),
                ),
                progress_info=ps_pb2.Progress.SUBMITTED,
                pipeline_id="pipeline_1",
                pipeline_info=ps_pb2.Pipeline(
                    predict_result_uris=[],
                    output=output,
                    score=ps_pb2.Score(
                        metric=ps_pb2.Metric.ACCURACY,
                        value=1.0,
                    ),
                ),
            )

    def ExecutePipeline(self, request, context):
        raise NotImplementedError


def main():
    logging.basicConfig(level=logging.INFO)

    with futures.ThreadPoolExecutor(max_workers=10) as executor:
        server = grpc.server(executor)
        ps_pb2_grpc.add_PipelineComputeServicer_to_server(
            ComputeService(), server)
        server.add_insecure_port('[::]:50051')
        server.start()
        while True:
            time.sleep(60)
