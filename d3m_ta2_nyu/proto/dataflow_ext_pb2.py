# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: dataflow_ext.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import core_pb2 as core__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='dataflow_ext.proto',
  package='',
  syntax='proto3',
  serialized_pb=_b('\n\x12\x64\x61taflow_ext.proto\x1a\ncore.proto\"J\n\x11PipelineReference\x12 \n\x07\x63ontext\x18\x01 \x01(\x0b\x32\x0f.SessionContext\x12\x13\n\x0bpipeline_id\x18\x02 \x01(\t\"\x85\x04\n\x13\x44\x61taflowDescription\x12 \n\rresponse_info\x18\x04 \x01(\x0b\x32\t.Response\x12\x13\n\x0bpipeline_id\x18\x01 \x01(\t\x12,\n\x07modules\x18\x02 \x03(\x0b\x32\x1b.DataflowDescription.Module\x12\x34\n\x0b\x63onnections\x18\x03 \x03(\x0b\x32\x1f.DataflowDescription.Connection\x1a\x32\n\x05Input\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\r\n\x05value\x18\x03 \x01(\t\x1a$\n\x06Output\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x1a\x8b\x01\n\x06Module\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04type\x18\x02 \x01(\t\x12\r\n\x05label\x18\x03 \x01(\t\x12*\n\x06inputs\x18\x04 \x03(\x0b\x32\x1a.DataflowDescription.Input\x12,\n\x07outputs\x18\x05 \x03(\x0b\x32\x1b.DataflowDescription.Output\x1ak\n\nConnection\x12\x16\n\x0e\x66rom_module_id\x18\x01 \x01(\t\x12\x18\n\x10\x66rom_output_name\x18\x02 \x01(\t\x12\x14\n\x0cto_module_id\x18\x03 \x01(\t\x12\x15\n\rto_input_name\x18\x04 \x01(\t\"2\n\x0cModuleOutput\x12\x13\n\x0boutput_name\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\"\xec\x01\n\x0cModuleResult\x12 \n\rresponse_info\x18\x06 \x01(\x0b\x32\t.Response\x12\x11\n\tmodule_id\x18\x01 \x01(\t\x12$\n\x06status\x18\x02 \x01(\x0e\x32\x14.ModuleResult.Status\x12\x10\n\x08progress\x18\x03 \x01(\x02\x12\x1e\n\x07outputs\x18\x04 \x03(\x0b\x32\r.ModuleOutput\x12\x16\n\x0e\x65xecution_time\x18\x05 \x01(\x02\"7\n\x06Status\x12\x0b\n\x07PENDING\x10\x00\x12\x0b\n\x07RUNNING\x10\x01\x12\x08\n\x04\x44ONE\x10\x02\x12\t\n\x05\x45RROR\x10\x03\x32\x8a\x01\n\x0b\x44\x61taflowExt\x12>\n\x10\x44\x65scribeDataflow\x12\x12.PipelineReference\x1a\x14.DataflowDescription\"\x00\x12;\n\x12GetDataflowResults\x12\x12.PipelineReference\x1a\r.ModuleResult\"\x00\x30\x01\x42\nZ\x08pipelineb\x06proto3')
  ,
  dependencies=[core__pb2.DESCRIPTOR,])



_MODULERESULT_STATUS = _descriptor.EnumDescriptor(
  name='Status',
  full_name='ModuleResult.Status',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='PENDING', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RUNNING', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DONE', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ERROR', index=3, number=3,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=864,
  serialized_end=919,
)
_sym_db.RegisterEnumDescriptor(_MODULERESULT_STATUS)


_PIPELINEREFERENCE = _descriptor.Descriptor(
  name='PipelineReference',
  full_name='PipelineReference',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='context', full_name='PipelineReference.context', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pipeline_id', full_name='PipelineReference.pipeline_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=108,
)


_DATAFLOWDESCRIPTION_INPUT = _descriptor.Descriptor(
  name='Input',
  full_name='DataflowDescription.Input',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='DataflowDescription.Input.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='DataflowDescription.Input.type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='DataflowDescription.Input.value', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=289,
  serialized_end=339,
)

_DATAFLOWDESCRIPTION_OUTPUT = _descriptor.Descriptor(
  name='Output',
  full_name='DataflowDescription.Output',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='DataflowDescription.Output.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='DataflowDescription.Output.type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=341,
  serialized_end=377,
)

_DATAFLOWDESCRIPTION_MODULE = _descriptor.Descriptor(
  name='Module',
  full_name='DataflowDescription.Module',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='DataflowDescription.Module.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='type', full_name='DataflowDescription.Module.type', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='label', full_name='DataflowDescription.Module.label', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='DataflowDescription.Module.inputs', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='outputs', full_name='DataflowDescription.Module.outputs', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=380,
  serialized_end=519,
)

_DATAFLOWDESCRIPTION_CONNECTION = _descriptor.Descriptor(
  name='Connection',
  full_name='DataflowDescription.Connection',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='from_module_id', full_name='DataflowDescription.Connection.from_module_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='from_output_name', full_name='DataflowDescription.Connection.from_output_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='to_module_id', full_name='DataflowDescription.Connection.to_module_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='to_input_name', full_name='DataflowDescription.Connection.to_input_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=521,
  serialized_end=628,
)

_DATAFLOWDESCRIPTION = _descriptor.Descriptor(
  name='DataflowDescription',
  full_name='DataflowDescription',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='response_info', full_name='DataflowDescription.response_info', index=0,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='pipeline_id', full_name='DataflowDescription.pipeline_id', index=1,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='modules', full_name='DataflowDescription.modules', index=2,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='connections', full_name='DataflowDescription.connections', index=3,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_DATAFLOWDESCRIPTION_INPUT, _DATAFLOWDESCRIPTION_OUTPUT, _DATAFLOWDESCRIPTION_MODULE, _DATAFLOWDESCRIPTION_CONNECTION, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=111,
  serialized_end=628,
)


_MODULEOUTPUT = _descriptor.Descriptor(
  name='ModuleOutput',
  full_name='ModuleOutput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='output_name', full_name='ModuleOutput.output_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='ModuleOutput.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=630,
  serialized_end=680,
)


_MODULERESULT = _descriptor.Descriptor(
  name='ModuleResult',
  full_name='ModuleResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='response_info', full_name='ModuleResult.response_info', index=0,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='module_id', full_name='ModuleResult.module_id', index=1,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='status', full_name='ModuleResult.status', index=2,
      number=2, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='progress', full_name='ModuleResult.progress', index=3,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='outputs', full_name='ModuleResult.outputs', index=4,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='execution_time', full_name='ModuleResult.execution_time', index=5,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _MODULERESULT_STATUS,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=683,
  serialized_end=919,
)

_PIPELINEREFERENCE.fields_by_name['context'].message_type = core__pb2._SESSIONCONTEXT
_DATAFLOWDESCRIPTION_INPUT.containing_type = _DATAFLOWDESCRIPTION
_DATAFLOWDESCRIPTION_OUTPUT.containing_type = _DATAFLOWDESCRIPTION
_DATAFLOWDESCRIPTION_MODULE.fields_by_name['inputs'].message_type = _DATAFLOWDESCRIPTION_INPUT
_DATAFLOWDESCRIPTION_MODULE.fields_by_name['outputs'].message_type = _DATAFLOWDESCRIPTION_OUTPUT
_DATAFLOWDESCRIPTION_MODULE.containing_type = _DATAFLOWDESCRIPTION
_DATAFLOWDESCRIPTION_CONNECTION.containing_type = _DATAFLOWDESCRIPTION
_DATAFLOWDESCRIPTION.fields_by_name['response_info'].message_type = core__pb2._RESPONSE
_DATAFLOWDESCRIPTION.fields_by_name['modules'].message_type = _DATAFLOWDESCRIPTION_MODULE
_DATAFLOWDESCRIPTION.fields_by_name['connections'].message_type = _DATAFLOWDESCRIPTION_CONNECTION
_MODULERESULT.fields_by_name['response_info'].message_type = core__pb2._RESPONSE
_MODULERESULT.fields_by_name['status'].enum_type = _MODULERESULT_STATUS
_MODULERESULT.fields_by_name['outputs'].message_type = _MODULEOUTPUT
_MODULERESULT_STATUS.containing_type = _MODULERESULT
DESCRIPTOR.message_types_by_name['PipelineReference'] = _PIPELINEREFERENCE
DESCRIPTOR.message_types_by_name['DataflowDescription'] = _DATAFLOWDESCRIPTION
DESCRIPTOR.message_types_by_name['ModuleOutput'] = _MODULEOUTPUT
DESCRIPTOR.message_types_by_name['ModuleResult'] = _MODULERESULT
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PipelineReference = _reflection.GeneratedProtocolMessageType('PipelineReference', (_message.Message,), dict(
  DESCRIPTOR = _PIPELINEREFERENCE,
  __module__ = 'dataflow_ext_pb2'
  # @@protoc_insertion_point(class_scope:PipelineReference)
  ))
_sym_db.RegisterMessage(PipelineReference)

DataflowDescription = _reflection.GeneratedProtocolMessageType('DataflowDescription', (_message.Message,), dict(

  Input = _reflection.GeneratedProtocolMessageType('Input', (_message.Message,), dict(
    DESCRIPTOR = _DATAFLOWDESCRIPTION_INPUT,
    __module__ = 'dataflow_ext_pb2'
    # @@protoc_insertion_point(class_scope:DataflowDescription.Input)
    ))
  ,

  Output = _reflection.GeneratedProtocolMessageType('Output', (_message.Message,), dict(
    DESCRIPTOR = _DATAFLOWDESCRIPTION_OUTPUT,
    __module__ = 'dataflow_ext_pb2'
    # @@protoc_insertion_point(class_scope:DataflowDescription.Output)
    ))
  ,

  Module = _reflection.GeneratedProtocolMessageType('Module', (_message.Message,), dict(
    DESCRIPTOR = _DATAFLOWDESCRIPTION_MODULE,
    __module__ = 'dataflow_ext_pb2'
    # @@protoc_insertion_point(class_scope:DataflowDescription.Module)
    ))
  ,

  Connection = _reflection.GeneratedProtocolMessageType('Connection', (_message.Message,), dict(
    DESCRIPTOR = _DATAFLOWDESCRIPTION_CONNECTION,
    __module__ = 'dataflow_ext_pb2'
    # @@protoc_insertion_point(class_scope:DataflowDescription.Connection)
    ))
  ,
  DESCRIPTOR = _DATAFLOWDESCRIPTION,
  __module__ = 'dataflow_ext_pb2'
  # @@protoc_insertion_point(class_scope:DataflowDescription)
  ))
_sym_db.RegisterMessage(DataflowDescription)
_sym_db.RegisterMessage(DataflowDescription.Input)
_sym_db.RegisterMessage(DataflowDescription.Output)
_sym_db.RegisterMessage(DataflowDescription.Module)
_sym_db.RegisterMessage(DataflowDescription.Connection)

ModuleOutput = _reflection.GeneratedProtocolMessageType('ModuleOutput', (_message.Message,), dict(
  DESCRIPTOR = _MODULEOUTPUT,
  __module__ = 'dataflow_ext_pb2'
  # @@protoc_insertion_point(class_scope:ModuleOutput)
  ))
_sym_db.RegisterMessage(ModuleOutput)

ModuleResult = _reflection.GeneratedProtocolMessageType('ModuleResult', (_message.Message,), dict(
  DESCRIPTOR = _MODULERESULT,
  __module__ = 'dataflow_ext_pb2'
  # @@protoc_insertion_point(class_scope:ModuleResult)
  ))
_sym_db.RegisterMessage(ModuleResult)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('Z\010pipeline'))

_DATAFLOWEXT = _descriptor.ServiceDescriptor(
  name='DataflowExt',
  full_name='DataflowExt',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=922,
  serialized_end=1060,
  methods=[
  _descriptor.MethodDescriptor(
    name='DescribeDataflow',
    full_name='DataflowExt.DescribeDataflow',
    index=0,
    containing_service=None,
    input_type=_PIPELINEREFERENCE,
    output_type=_DATAFLOWDESCRIPTION,
    options=None,
  ),
  _descriptor.MethodDescriptor(
    name='GetDataflowResults',
    full_name='DataflowExt.GetDataflowResults',
    index=1,
    containing_service=None,
    input_type=_PIPELINEREFERENCE,
    output_type=_MODULERESULT,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_DATAFLOWEXT)

DESCRIPTOR.services_by_name['DataflowExt'] = _DATAFLOWEXT

try:
  # THESE ELEMENTS WILL BE DEPRECATED.
  # Please use the generated *_pb2_grpc.py files instead.
  import grpc
  from grpc.beta import implementations as beta_implementations
  from grpc.beta import interfaces as beta_interfaces
  from grpc.framework.common import cardinality
  from grpc.framework.interfaces.face import utilities as face_utilities


  class DataflowExtStub(object):
    """Optional dataflow service, through which a TA2 system can expose the
    internals of computations outlining a machine learning pipeline.
    """

    def __init__(self, channel):
      """Constructor.

      Args:
        channel: A grpc.Channel.
      """
      self.DescribeDataflow = channel.unary_unary(
          '/DataflowExt/DescribeDataflow',
          request_serializer=PipelineReference.SerializeToString,
          response_deserializer=DataflowDescription.FromString,
          )
      self.GetDataflowResults = channel.unary_stream(
          '/DataflowExt/GetDataflowResults',
          request_serializer=PipelineReference.SerializeToString,
          response_deserializer=ModuleResult.FromString,
          )


  class DataflowExtServicer(object):
    """Optional dataflow service, through which a TA2 system can expose the
    internals of computations outlining a machine learning pipeline.
    """

    def DescribeDataflow(self, request, context):
      """Get a description of the training process as a dataflow.
      """
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')

    def GetDataflowResults(self, request, context):
      """Get the results of each module making up the dataflow, for the most
      recent training run.
      """
      context.set_code(grpc.StatusCode.UNIMPLEMENTED)
      context.set_details('Method not implemented!')
      raise NotImplementedError('Method not implemented!')


  def add_DataflowExtServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'DescribeDataflow': grpc.unary_unary_rpc_method_handler(
            servicer.DescribeDataflow,
            request_deserializer=PipelineReference.FromString,
            response_serializer=DataflowDescription.SerializeToString,
        ),
        'GetDataflowResults': grpc.unary_stream_rpc_method_handler(
            servicer.GetDataflowResults,
            request_deserializer=PipelineReference.FromString,
            response_serializer=ModuleResult.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        'DataflowExt', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


  class BetaDataflowExtServicer(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    """Optional dataflow service, through which a TA2 system can expose the
    internals of computations outlining a machine learning pipeline.
    """
    def DescribeDataflow(self, request, context):
      """Get a description of the training process as a dataflow.
      """
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)
    def GetDataflowResults(self, request, context):
      """Get the results of each module making up the dataflow, for the most
      recent training run.
      """
      context.code(beta_interfaces.StatusCode.UNIMPLEMENTED)


  class BetaDataflowExtStub(object):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This class was generated
    only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0."""
    """Optional dataflow service, through which a TA2 system can expose the
    internals of computations outlining a machine learning pipeline.
    """
    def DescribeDataflow(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      """Get a description of the training process as a dataflow.
      """
      raise NotImplementedError()
    DescribeDataflow.future = None
    def GetDataflowResults(self, request, timeout, metadata=None, with_call=False, protocol_options=None):
      """Get the results of each module making up the dataflow, for the most
      recent training run.
      """
      raise NotImplementedError()


  def beta_create_DataflowExt_server(servicer, pool=None, pool_size=None, default_timeout=None, maximum_timeout=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_deserializers = {
      ('DataflowExt', 'DescribeDataflow'): PipelineReference.FromString,
      ('DataflowExt', 'GetDataflowResults'): PipelineReference.FromString,
    }
    response_serializers = {
      ('DataflowExt', 'DescribeDataflow'): DataflowDescription.SerializeToString,
      ('DataflowExt', 'GetDataflowResults'): ModuleResult.SerializeToString,
    }
    method_implementations = {
      ('DataflowExt', 'DescribeDataflow'): face_utilities.unary_unary_inline(servicer.DescribeDataflow),
      ('DataflowExt', 'GetDataflowResults'): face_utilities.unary_stream_inline(servicer.GetDataflowResults),
    }
    server_options = beta_implementations.server_options(request_deserializers=request_deserializers, response_serializers=response_serializers, thread_pool=pool, thread_pool_size=pool_size, default_timeout=default_timeout, maximum_timeout=maximum_timeout)
    return beta_implementations.server(method_implementations, options=server_options)


  def beta_create_DataflowExt_stub(channel, host=None, metadata_transformer=None, pool=None, pool_size=None):
    """The Beta API is deprecated for 0.15.0 and later.

    It is recommended to use the GA API (classes and functions in this
    file not marked beta) for all further purposes. This function was
    generated only to ease transition from grpcio<0.15.0 to grpcio>=0.15.0"""
    request_serializers = {
      ('DataflowExt', 'DescribeDataflow'): PipelineReference.SerializeToString,
      ('DataflowExt', 'GetDataflowResults'): PipelineReference.SerializeToString,
    }
    response_deserializers = {
      ('DataflowExt', 'DescribeDataflow'): DataflowDescription.FromString,
      ('DataflowExt', 'GetDataflowResults'): ModuleResult.FromString,
    }
    cardinalities = {
      'DescribeDataflow': cardinality.Cardinality.UNARY_UNARY,
      'GetDataflowResults': cardinality.Cardinality.UNARY_STREAM,
    }
    stub_options = beta_implementations.stub_options(host=host, metadata_transformer=metadata_transformer, request_serializers=request_serializers, response_deserializers=response_deserializers, thread_pool=pool, thread_pool_size=pool_size)
    return beta_implementations.dynamic_stub(channel, 'DataflowExt', cardinalities, options=stub_options)
except ImportError:
  pass
# @@protoc_insertion_point(module_scope)
