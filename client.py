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

    reply_stream = stub.CreatePipelines(ps_pb2.PipelineCreateRequest(
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
    for result in reply_stream:
        if result.response_info.status.code == ps_pb2.CANCELLED:
            print "Pipelines creation cancelled"
            break
        elif result.response_info.status.code != ps_pb2.OK:
            print "Error during pipelines creation"
            if result.response_info.status.details:
                print "details: %r" % result.response_info.status.details
            break
        progress = result.progress_info
        pipeline_id = result.pipeline_id
        pipeline = result.pipeline_info
        if not result.HasField('pipeline_info'):
            pipeline = None

        print "%s %s %s" % (progress, pipeline_id, pipeline)

    reply = stub.EndSession(context)
    print "Ended session %r, status %s" % (context.session_id,
                                           reply.status.code)
