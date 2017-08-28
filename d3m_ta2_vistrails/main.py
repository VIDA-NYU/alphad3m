from concurrent import futures
import grpc
import logging
import time

from d3m_ta2_vistrails import __version__
import d3m_ta2_vistrails.proto.pipeline_service_pb2 as ps_pb2
import d3m_ta2_vistrails.proto.pipeline_service_pb2_grpc as ps_pb2_grpc


logger = logging.getLogger(__name__)


class D3mTa2(object):
    executor = None

    def __init__(self, insecure_ports=None):
        if insecure_ports is None:
            self._insecure_ports = ['[::]:50051']
            self.core_rpc = CoreService()

    def run(self):
        self.executor = futures.ThreadPoolExecutor(max_workers=10)
        try:
            server = grpc.server(self.executor)
            ps_pb2_grpc.add_PipelineComputeServicer_to_server(
                self.core_rpc, server)
            for port in self._insecure_ports:
                server.add_insecure_port(port)
            server.start()
            while True:
                time.sleep(60)
        finally:
            self.executor.shutdown(wait=True)


class CoreService(ps_pb2_grpc.PipelineComputeServicer):
    def StartSession(self, request, context):
        version = ps_pb2.DESCRIPTOR.GetOptions().Extensions[
            ps_pb2.protocol_version]
        logger.info("Session started: 1 (protocol version %s)", version)
        return ps_pb2.SessionResponse(
            response_info=ps_pb2.Response(
                status=ps_pb2.Status(code=ps_pb2.OK)
            ),
            user_agent='vistrails_ta2 %s' % __version__,
            version=version,
            context=ps_pb2.SessionContext(session_id='1'),
        )

    def EndSession(self, request, context):
        assert request.session_id == '1'
        logger.info("Session terminated: 1")
        return ps_pb2.Response(
            status=ps_pb2.Status(code=ps_pb2.OK),
        )

    def CreatePipelines(self, request, context):
        sessioncontext = request.context
        assert sessioncontext.session_id == '1'
        train_features = request.train_features
        task = request.task
        assert task == ps_pb2.CLASSIFICATION
        task_subtype = request.task_subtype
        task_description = request.task_description
        output = request.output
        metrics = request.metrics
        target_features = request.target_features
        max_pipelines = request.max_pipelines

        logger.info("Got CreatePipelines request, session=%s",
                    sessioncontext.session_id)

        results = [
            (ps_pb2.SUBMITTED, 'pipeline_1', False),
            (ps_pb2.RUNNING, 'pipeline_1', False),
            (ps_pb2.COMPLETED, 'pipeline_1', True),
        ]

        for progress, pipeline_id, send_pipeline in results:
            if not context.is_active():
                logger.info("Client closed CreatePipelines stream")
            msg = ps_pb2.PipelineCreateResult(
                response_info=ps_pb2.Response(
                    status=ps_pb2.Status(code=ps_pb2.OK),
                ),
                progress_info=progress,
                pipeline_id=pipeline_id,
            )
            if send_pipeline:
                msg.pipeline_info.CopyFrom(ps_pb2.Pipeline(
                    predict_result_uris=['file:///out/predict1.csv'],
                    output=output,
                    scores=[
                        ps_pb2.Score(
                            metric=ps_pb2.ACCURACY,
                            value=0.8,
                        ),
                        ps_pb2.Score(
                            metric=ps_pb2.ROC_AUC,
                            value=0.5,
                        ),
                    ],
                ))
            yield msg

    def ExecutePipeline(self, request, context):
        raise NotImplementedError


def main():
    logging.basicConfig(level=logging.INFO)

    D3mTa2().run()
