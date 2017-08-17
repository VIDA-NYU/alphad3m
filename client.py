import grpc
import d3m_ta2_vistrails.proto.pipeline_service_pb2 as ps_pb2
import d3m_ta2_vistrails.proto.pipeline_service_pb2_grpc as ps_pb2_grpc


if __name__ == '__main__':
    channel = grpc.insecure_channel('localhost:50051')
    stub = ps_pb2_grpc.PipelineComputeStub(channel)

    reply = stub.StartSession(ps_pb2.SessionRequest())
    context = reply.context
    print "Started session %r, status %s" % (context.session_id,
                                             reply.status.code)

    reply = stub.EndSession(context)
    print "Ended session %r, status %s" % (context.session_id,
                                           reply.status.code)
