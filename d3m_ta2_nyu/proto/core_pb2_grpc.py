# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

from . import core_pb2 as core__pb2


class CoreStub(object):
  # missing associated documentation comment in .proto file
  pass

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.CreatePipelines = channel.unary_stream(
        '/Core/CreatePipelines',
        request_serializer=core__pb2.PipelineCreateRequest.SerializeToString,
        response_deserializer=core__pb2.PipelineCreateResult.FromString,
        )
    self.ExecutePipeline = channel.unary_stream(
        '/Core/ExecutePipeline',
        request_serializer=core__pb2.PipelineExecuteRequest.SerializeToString,
        response_deserializer=core__pb2.PipelineExecuteResult.FromString,
        )
    self.ListPipelines = channel.unary_unary(
        '/Core/ListPipelines',
        request_serializer=core__pb2.PipelineListRequest.SerializeToString,
        response_deserializer=core__pb2.PipelineListResult.FromString,
        )
    self.DeletePipelines = channel.unary_unary(
        '/Core/DeletePipelines',
        request_serializer=core__pb2.PipelineDeleteRequest.SerializeToString,
        response_deserializer=core__pb2.PipelineListResult.FromString,
        )
    self.GetCreatePipelineResults = channel.unary_stream(
        '/Core/GetCreatePipelineResults',
        request_serializer=core__pb2.PipelineCreateResultsRequest.SerializeToString,
        response_deserializer=core__pb2.PipelineCreateResult.FromString,
        )
    self.GetExecutePipelineResults = channel.unary_stream(
        '/Core/GetExecutePipelineResults',
        request_serializer=core__pb2.PipelineExecuteResultsRequest.SerializeToString,
        response_deserializer=core__pb2.PipelineExecuteResult.FromString,
        )
    self.ExportPipeline = channel.unary_unary(
        '/Core/ExportPipeline',
        request_serializer=core__pb2.PipelineExportRequest.SerializeToString,
        response_deserializer=core__pb2.Response.FromString,
        )
    self.UpdateProblemSchema = channel.unary_unary(
        '/Core/UpdateProblemSchema',
        request_serializer=core__pb2.UpdateProblemSchemaRequest.SerializeToString,
        response_deserializer=core__pb2.Response.FromString,
        )
    self.StartSession = channel.unary_unary(
        '/Core/StartSession',
        request_serializer=core__pb2.SessionRequest.SerializeToString,
        response_deserializer=core__pb2.SessionResponse.FromString,
        )
    self.EndSession = channel.unary_unary(
        '/Core/EndSession',
        request_serializer=core__pb2.SessionContext.SerializeToString,
        response_deserializer=core__pb2.Response.FromString,
        )


class CoreServicer(object):
  # missing associated documentation comment in .proto file
  pass

  def CreatePipelines(self, request, context):
    """Train step - multiple result messages returned via GRPC streaming.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ExecutePipeline(self, request, context):
    """Predict step - multiple results messages returned via GRPC streaming.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ListPipelines(self, request, context):
    """Manage pipelines already present in the session.
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def DeletePipelines(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetCreatePipelineResults(self, request, context):
    """Obtain results; lists existing pipelines then streams new results as they become available
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def GetExecutePipelineResults(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def ExportPipeline(self, request, context):
    """Export executable of a pipeline, including any optional preprocessing used in session
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def UpdateProblemSchema(self, request, context):
    """Update problem schema
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def StartSession(self, request, context):
    """Session management
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')

  def EndSession(self, request, context):
    # missing associated documentation comment in .proto file
    pass
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_CoreServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'CreatePipelines': grpc.unary_stream_rpc_method_handler(
          servicer.CreatePipelines,
          request_deserializer=core__pb2.PipelineCreateRequest.FromString,
          response_serializer=core__pb2.PipelineCreateResult.SerializeToString,
      ),
      'ExecutePipeline': grpc.unary_stream_rpc_method_handler(
          servicer.ExecutePipeline,
          request_deserializer=core__pb2.PipelineExecuteRequest.FromString,
          response_serializer=core__pb2.PipelineExecuteResult.SerializeToString,
      ),
      'ListPipelines': grpc.unary_unary_rpc_method_handler(
          servicer.ListPipelines,
          request_deserializer=core__pb2.PipelineListRequest.FromString,
          response_serializer=core__pb2.PipelineListResult.SerializeToString,
      ),
      'DeletePipelines': grpc.unary_unary_rpc_method_handler(
          servicer.DeletePipelines,
          request_deserializer=core__pb2.PipelineDeleteRequest.FromString,
          response_serializer=core__pb2.PipelineListResult.SerializeToString,
      ),
      'GetCreatePipelineResults': grpc.unary_stream_rpc_method_handler(
          servicer.GetCreatePipelineResults,
          request_deserializer=core__pb2.PipelineCreateResultsRequest.FromString,
          response_serializer=core__pb2.PipelineCreateResult.SerializeToString,
      ),
      'GetExecutePipelineResults': grpc.unary_stream_rpc_method_handler(
          servicer.GetExecutePipelineResults,
          request_deserializer=core__pb2.PipelineExecuteResultsRequest.FromString,
          response_serializer=core__pb2.PipelineExecuteResult.SerializeToString,
      ),
      'ExportPipeline': grpc.unary_unary_rpc_method_handler(
          servicer.ExportPipeline,
          request_deserializer=core__pb2.PipelineExportRequest.FromString,
          response_serializer=core__pb2.Response.SerializeToString,
      ),
      'UpdateProblemSchema': grpc.unary_unary_rpc_method_handler(
          servicer.UpdateProblemSchema,
          request_deserializer=core__pb2.UpdateProblemSchemaRequest.FromString,
          response_serializer=core__pb2.Response.SerializeToString,
      ),
      'StartSession': grpc.unary_unary_rpc_method_handler(
          servicer.StartSession,
          request_deserializer=core__pb2.SessionRequest.FromString,
          response_serializer=core__pb2.SessionResponse.SerializeToString,
      ),
      'EndSession': grpc.unary_unary_rpc_method_handler(
          servicer.EndSession,
          request_deserializer=core__pb2.SessionContext.FromString,
          response_serializer=core__pb2.Response.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'Core', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))