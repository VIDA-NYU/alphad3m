import grpc
import d3m_ta2_vistrails.proto.pipeline_service_pb2 as ps_pb2
import d3m_ta2_vistrails.proto.pipeline_service_pb2_grpc as ps_pb2_grpc


__version__ = '0.1'


if __name__ == '__main__':
    channel = grpc.insecure_channel('localhost:50051')
    stub = ps_pb2_grpc.PipelineComputeStub(channel)

    version = ps_pb2.DESCRIPTOR.GetOptions().Extensions[
        ps_pb2.protocol_version]

    reply = stub.StartSession(ps_pb2.SessionRequest(
        user_agent='text_client %s' % __version__,
        version=version,
    ))
    context = reply.context
    print "Started session %r, status %s" % (context.session_id,
                                             reply.response_info.status.code)

    reply = stub.CreatePipelines(ps_pb2.PipelineCreateRequest(
        context=context,
        train_features=[
            ps_pb2.Feature(feature_id='feature1',
                           data_uri='file:///data/feature1.csv'),
            ps_pb2.Feature(feature_id='feature2',
                           data_uri='file:///data/feature2.csv'),
        ],
        task=ps_pb2.CLASSIFICATION,
        task_subtype=ps_pb2.NONE,
        task_description="Debugging task",
        output=ps_pb2.PROBABILITY,
        metrics=[
            ps_pb2.ACCURACY,
            ps_pb2.ROC_AUC,
        ],
        target_features=[
            ps_pb2.Feature(feature_id='targetfeature',
                           data_uri='file:///data/targetfeature.csv'),
        ],
        max_pipelines=10,
    ))
    print("Requested pipelines")

    reply = stub.EndSession(context)
    print "Ended session %r, status %s" % (context.session_id,
                                           reply.status.code)
