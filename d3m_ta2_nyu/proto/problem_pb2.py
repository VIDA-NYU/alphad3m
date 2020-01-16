# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: problem.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='problem.proto',
  package='',
  syntax='proto3',
  serialized_options=_b('Z\010pipeline'),
  serialized_pb=_b('\n\rproblem.proto\"\\\n\x18ProblemPerformanceMetric\x12\"\n\x06metric\x18\x01 \x01(\x0e\x32\x12.PerformanceMetric\x12\t\n\x01k\x18\x02 \x01(\x05\x12\x11\n\tpos_label\x18\x03 \x01(\t\"f\n\x07Problem\x12#\n\rtask_keywords\x18\x08 \x03(\x0e\x32\x0c.TaskKeyword\x12\x36\n\x13performance_metrics\x18\x07 \x03(\x0b\x32\x19.ProblemPerformanceMetric\"~\n\rProblemTarget\x12\x14\n\x0ctarget_index\x18\x01 \x01(\x05\x12\x13\n\x0bresource_id\x18\x02 \x01(\t\x12\x14\n\x0c\x63olumn_index\x18\x03 \x01(\x05\x12\x13\n\x0b\x63olumn_name\x18\x04 \x01(\t\x12\x17\n\x0f\x63lusters_number\x18\x05 \x01(\x05\"v\n\x15ProblemPrivilegedData\x12\x1d\n\x15privileged_data_index\x18\x01 \x01(\x05\x12\x13\n\x0bresource_id\x18\x02 \x01(\t\x12\x14\n\x0c\x63olumn_index\x18\x03 \x01(\x05\x12\x13\n\x0b\x63olumn_name\x18\x04 \x01(\t\"k\n\x12\x46orecastingHorizon\x12\x13\n\x0bresource_id\x18\x01 \x01(\t\x12\x14\n\x0c\x63olumn_index\x18\x02 \x01(\x05\x12\x13\n\x0b\x63olumn_name\x18\x03 \x01(\t\x12\x15\n\rhorizon_value\x18\x04 \x01(\x01\"\xa6\x01\n\x0cProblemInput\x12\x12\n\ndataset_id\x18\x01 \x01(\t\x12\x1f\n\x07targets\x18\x02 \x03(\x0b\x32\x0e.ProblemTarget\x12/\n\x0fprivileged_data\x18\x03 \x03(\x0b\x32\x16.ProblemPrivilegedData\x12\x30\n\x13\x66orecasting_horizon\x18\x04 \x01(\x0b\x32\x13.ForecastingHorizon\"4\n\x10\x44\x61taAugmentation\x12\x0e\n\x06\x64omain\x18\x01 \x03(\t\x12\x10\n\x08keywords\x18\x02 \x03(\t\"\xe1\x01\n\x12ProblemDescription\x12\x19\n\x07problem\x18\x01 \x01(\x0b\x32\x08.Problem\x12\x1d\n\x06inputs\x18\x02 \x03(\x0b\x32\r.ProblemInput\x12\n\n\x02id\x18\x03 \x01(\t\x12\x0f\n\x07version\x18\x04 \x01(\t\x12\x0c\n\x04name\x18\x05 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x06 \x01(\t\x12\x0e\n\x06\x64igest\x18\x07 \x01(\t\x12,\n\x11\x64\x61ta_augmentation\x18\x08 \x03(\x0b\x32\x11.DataAugmentation\x12\x13\n\x0bother_names\x18\t \x03(\t*\xed\x04\n\x0bTaskKeyword\x12\x1a\n\x16TASK_KEYWORD_UNDEFINED\x10\x00\x12\x12\n\x0e\x43LASSIFICATION\x10\x01\x12\x0e\n\nREGRESSION\x10\x02\x12\x0e\n\nCLUSTERING\x10\x03\x12\x13\n\x0fLINK_PREDICTION\x10\x04\x12\x15\n\x11VERTEX_NOMINATION\x10\x05\x12\x19\n\x15VERTEX_CLASSIFICATION\x10\x06\x12\x17\n\x13\x43OMMUNITY_DETECTION\x10\x07\x12\x12\n\x0eGRAPH_MATCHING\x10\x08\x12\x0f\n\x0b\x46ORECASTING\x10\t\x12\x1b\n\x17\x43OLLABORATIVE_FILTERING\x10\n\x12\x14\n\x10OBJECT_DETECTION\x10\x0b\x12\x12\n\x0eSEMISUPERVISED\x10\x0c\x12\n\n\x06\x42INARY\x10\r\x12\x0e\n\nMULTICLASS\x10\x0e\x12\x0e\n\nMULTILABEL\x10\x0f\x12\x0e\n\nUNIVARIATE\x10\x10\x12\x10\n\x0cMULTIVARIATE\x10\x11\x12\x0f\n\x0bOVERLAPPING\x10\x12\x12\x12\n\x0eNONOVERLAPPING\x10\x13\x12\x0b\n\x07TABULAR\x10\x14\x12\x0e\n\nRELATIONAL\x10\x15\x12\t\n\x05IMAGE\x10\x16\x12\t\n\x05\x41UDIO\x10\x17\x12\t\n\x05VIDEO\x10\x18\x12\n\n\x06SPEECH\x10\x19\x12\x08\n\x04TEXT\x10\x1a\x12\t\n\x05GRAPH\x10\x1b\x12\x0e\n\nMULTIGRAPH\x10\x1c\x12\x0f\n\x0bTIME_SERIES\x10\x1d\x12\x0b\n\x07GROUPED\x10\x1e\x12\x0e\n\nGEOSPATIAL\x10\x1f\x12\x12\n\x0eREMOTE_SENSING\x10 \x12\x08\n\x04LUPI\x10!\x12\x14\n\x10MISSING_METADATA\x10\"*\xad\x03\n\x11PerformanceMetric\x12\x14\n\x10METRIC_UNDEFINED\x10\x00\x12\x0c\n\x08\x41\x43\x43URACY\x10\x01\x12\r\n\tPRECISION\x10\x02\x12\n\n\x06RECALL\x10\x03\x12\x06\n\x02\x46\x31\x10\x04\x12\x0c\n\x08\x46\x31_MICRO\x10\x05\x12\x0c\n\x08\x46\x31_MACRO\x10\x06\x12\x0b\n\x07ROC_AUC\x10\x07\x12\x11\n\rROC_AUC_MICRO\x10\x08\x12\x11\n\rROC_AUC_MACRO\x10\t\x12\x16\n\x12MEAN_SQUARED_ERROR\x10\n\x12\x1b\n\x17ROOT_MEAN_SQUARED_ERROR\x10\x0b\x12\x17\n\x13MEAN_ABSOLUTE_ERROR\x10\x0c\x12\r\n\tR_SQUARED\x10\r\x12!\n\x1dNORMALIZED_MUTUAL_INFORMATION\x10\x0e\x12\x1c\n\x18JACCARD_SIMILARITY_SCORE\x10\x0f\x12\x16\n\x12PRECISION_AT_TOP_K\x10\x11\x12&\n\"OBJECT_DETECTION_AVERAGE_PRECISION\x10\x12\x12\x10\n\x0cHAMMING_LOSS\x10\x13\x12\x08\n\x04RANK\x10\x63\x12\x08\n\x04LOSS\x10\x64\x42\nZ\x08pipelineb\x06proto3')
)

_TASKKEYWORD = _descriptor.EnumDescriptor(
  name='TaskKeyword',
  full_name='TaskKeyword',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='TASK_KEYWORD_UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CLASSIFICATION', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REGRESSION', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='CLUSTERING', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LINK_PREDICTION', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VERTEX_NOMINATION', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VERTEX_CLASSIFICATION', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COMMUNITY_DETECTION', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GRAPH_MATCHING', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FORECASTING', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='COLLABORATIVE_FILTERING', index=10, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBJECT_DETECTION', index=11, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SEMISUPERVISED', index=12, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='BINARY', index=13, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTICLASS', index=14, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTILABEL', index=15, number=15,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='UNIVARIATE', index=16, number=16,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTIVARIATE', index=17, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OVERLAPPING', index=18, number=18,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NONOVERLAPPING', index=19, number=19,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TABULAR', index=20, number=20,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RELATIONAL', index=21, number=21,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='IMAGE', index=22, number=22,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='AUDIO', index=23, number=23,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='VIDEO', index=24, number=24,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SPEECH', index=25, number=25,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TEXT', index=26, number=26,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GRAPH', index=27, number=27,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MULTIGRAPH', index=28, number=28,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='TIME_SERIES', index=29, number=29,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GROUPED', index=30, number=30,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='GEOSPATIAL', index=31, number=31,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='REMOTE_SENSING', index=32, number=32,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LUPI', index=33, number=33,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MISSING_METADATA', index=34, number=34,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1024,
  serialized_end=1645,
)
_sym_db.RegisterEnumDescriptor(_TASKKEYWORD)

TaskKeyword = enum_type_wrapper.EnumTypeWrapper(_TASKKEYWORD)
_PERFORMANCEMETRIC = _descriptor.EnumDescriptor(
  name='PerformanceMetric',
  full_name='PerformanceMetric',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='METRIC_UNDEFINED', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ACCURACY', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRECISION', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RECALL', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='F1', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='F1_MICRO', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='F1_MACRO', index=6, number=6,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ROC_AUC', index=7, number=7,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ROC_AUC_MICRO', index=8, number=8,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ROC_AUC_MACRO', index=9, number=9,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MEAN_SQUARED_ERROR', index=10, number=10,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ROOT_MEAN_SQUARED_ERROR', index=11, number=11,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MEAN_ABSOLUTE_ERROR', index=12, number=12,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='R_SQUARED', index=13, number=13,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='NORMALIZED_MUTUAL_INFORMATION', index=14, number=14,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='JACCARD_SIMILARITY_SCORE', index=15, number=15,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='PRECISION_AT_TOP_K', index=16, number=17,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='OBJECT_DETECTION_AVERAGE_PRECISION', index=17, number=18,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='HAMMING_LOSS', index=18, number=19,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='RANK', index=19, number=99,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='LOSS', index=20, number=100,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1648,
  serialized_end=2077,
)
_sym_db.RegisterEnumDescriptor(_PERFORMANCEMETRIC)

PerformanceMetric = enum_type_wrapper.EnumTypeWrapper(_PERFORMANCEMETRIC)
TASK_KEYWORD_UNDEFINED = 0
CLASSIFICATION = 1
REGRESSION = 2
CLUSTERING = 3
LINK_PREDICTION = 4
VERTEX_NOMINATION = 5
VERTEX_CLASSIFICATION = 6
COMMUNITY_DETECTION = 7
GRAPH_MATCHING = 8
FORECASTING = 9
COLLABORATIVE_FILTERING = 10
OBJECT_DETECTION = 11
SEMISUPERVISED = 12
BINARY = 13
MULTICLASS = 14
MULTILABEL = 15
UNIVARIATE = 16
MULTIVARIATE = 17
OVERLAPPING = 18
NONOVERLAPPING = 19
TABULAR = 20
RELATIONAL = 21
IMAGE = 22
AUDIO = 23
VIDEO = 24
SPEECH = 25
TEXT = 26
GRAPH = 27
MULTIGRAPH = 28
TIME_SERIES = 29
GROUPED = 30
GEOSPATIAL = 31
REMOTE_SENSING = 32
LUPI = 33
MISSING_METADATA = 34
METRIC_UNDEFINED = 0
ACCURACY = 1
PRECISION = 2
RECALL = 3
F1 = 4
F1_MICRO = 5
F1_MACRO = 6
ROC_AUC = 7
ROC_AUC_MICRO = 8
ROC_AUC_MACRO = 9
MEAN_SQUARED_ERROR = 10
ROOT_MEAN_SQUARED_ERROR = 11
MEAN_ABSOLUTE_ERROR = 12
R_SQUARED = 13
NORMALIZED_MUTUAL_INFORMATION = 14
JACCARD_SIMILARITY_SCORE = 15
PRECISION_AT_TOP_K = 17
OBJECT_DETECTION_AVERAGE_PRECISION = 18
HAMMING_LOSS = 19
RANK = 99
LOSS = 100



_PROBLEMPERFORMANCEMETRIC = _descriptor.Descriptor(
  name='ProblemPerformanceMetric',
  full_name='ProblemPerformanceMetric',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='metric', full_name='ProblemPerformanceMetric.metric', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='k', full_name='ProblemPerformanceMetric.k', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pos_label', full_name='ProblemPerformanceMetric.pos_label', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=17,
  serialized_end=109,
)


_PROBLEM = _descriptor.Descriptor(
  name='Problem',
  full_name='Problem',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='task_keywords', full_name='Problem.task_keywords', index=0,
      number=8, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='performance_metrics', full_name='Problem.performance_metrics', index=1,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=111,
  serialized_end=213,
)


_PROBLEMTARGET = _descriptor.Descriptor(
  name='ProblemTarget',
  full_name='ProblemTarget',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='target_index', full_name='ProblemTarget.target_index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resource_id', full_name='ProblemTarget.resource_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='column_index', full_name='ProblemTarget.column_index', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='column_name', full_name='ProblemTarget.column_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clusters_number', full_name='ProblemTarget.clusters_number', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=215,
  serialized_end=341,
)


_PROBLEMPRIVILEGEDDATA = _descriptor.Descriptor(
  name='ProblemPrivilegedData',
  full_name='ProblemPrivilegedData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='privileged_data_index', full_name='ProblemPrivilegedData.privileged_data_index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resource_id', full_name='ProblemPrivilegedData.resource_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='column_index', full_name='ProblemPrivilegedData.column_index', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='column_name', full_name='ProblemPrivilegedData.column_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=343,
  serialized_end=461,
)


_FORECASTINGHORIZON = _descriptor.Descriptor(
  name='ForecastingHorizon',
  full_name='ForecastingHorizon',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='resource_id', full_name='ForecastingHorizon.resource_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='column_index', full_name='ForecastingHorizon.column_index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='column_name', full_name='ForecastingHorizon.column_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='horizon_value', full_name='ForecastingHorizon.horizon_value', index=3,
      number=4, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=463,
  serialized_end=570,
)


_PROBLEMINPUT = _descriptor.Descriptor(
  name='ProblemInput',
  full_name='ProblemInput',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='dataset_id', full_name='ProblemInput.dataset_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='targets', full_name='ProblemInput.targets', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='privileged_data', full_name='ProblemInput.privileged_data', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='forecasting_horizon', full_name='ProblemInput.forecasting_horizon', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=573,
  serialized_end=739,
)


_DATAAUGMENTATION = _descriptor.Descriptor(
  name='DataAugmentation',
  full_name='DataAugmentation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='domain', full_name='DataAugmentation.domain', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='keywords', full_name='DataAugmentation.keywords', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=741,
  serialized_end=793,
)


_PROBLEMDESCRIPTION = _descriptor.Descriptor(
  name='ProblemDescription',
  full_name='ProblemDescription',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='problem', full_name='ProblemDescription.problem', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='ProblemDescription.inputs', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='id', full_name='ProblemDescription.id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='ProblemDescription.version', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='name', full_name='ProblemDescription.name', index=4,
      number=5, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='description', full_name='ProblemDescription.description', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='digest', full_name='ProblemDescription.digest', index=6,
      number=7, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='data_augmentation', full_name='ProblemDescription.data_augmentation', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='other_names', full_name='ProblemDescription.other_names', index=8,
      number=9, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=796,
  serialized_end=1021,
)

_PROBLEMPERFORMANCEMETRIC.fields_by_name['metric'].enum_type = _PERFORMANCEMETRIC
_PROBLEM.fields_by_name['task_keywords'].enum_type = _TASKKEYWORD
_PROBLEM.fields_by_name['performance_metrics'].message_type = _PROBLEMPERFORMANCEMETRIC
_PROBLEMINPUT.fields_by_name['targets'].message_type = _PROBLEMTARGET
_PROBLEMINPUT.fields_by_name['privileged_data'].message_type = _PROBLEMPRIVILEGEDDATA
_PROBLEMINPUT.fields_by_name['forecasting_horizon'].message_type = _FORECASTINGHORIZON
_PROBLEMDESCRIPTION.fields_by_name['problem'].message_type = _PROBLEM
_PROBLEMDESCRIPTION.fields_by_name['inputs'].message_type = _PROBLEMINPUT
_PROBLEMDESCRIPTION.fields_by_name['data_augmentation'].message_type = _DATAAUGMENTATION
DESCRIPTOR.message_types_by_name['ProblemPerformanceMetric'] = _PROBLEMPERFORMANCEMETRIC
DESCRIPTOR.message_types_by_name['Problem'] = _PROBLEM
DESCRIPTOR.message_types_by_name['ProblemTarget'] = _PROBLEMTARGET
DESCRIPTOR.message_types_by_name['ProblemPrivilegedData'] = _PROBLEMPRIVILEGEDDATA
DESCRIPTOR.message_types_by_name['ForecastingHorizon'] = _FORECASTINGHORIZON
DESCRIPTOR.message_types_by_name['ProblemInput'] = _PROBLEMINPUT
DESCRIPTOR.message_types_by_name['DataAugmentation'] = _DATAAUGMENTATION
DESCRIPTOR.message_types_by_name['ProblemDescription'] = _PROBLEMDESCRIPTION
DESCRIPTOR.enum_types_by_name['TaskKeyword'] = _TASKKEYWORD
DESCRIPTOR.enum_types_by_name['PerformanceMetric'] = _PERFORMANCEMETRIC
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ProblemPerformanceMetric = _reflection.GeneratedProtocolMessageType('ProblemPerformanceMetric', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEMPERFORMANCEMETRIC,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ProblemPerformanceMetric)
  ))
_sym_db.RegisterMessage(ProblemPerformanceMetric)

Problem = _reflection.GeneratedProtocolMessageType('Problem', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEM,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:Problem)
  ))
_sym_db.RegisterMessage(Problem)

ProblemTarget = _reflection.GeneratedProtocolMessageType('ProblemTarget', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEMTARGET,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ProblemTarget)
  ))
_sym_db.RegisterMessage(ProblemTarget)

ProblemPrivilegedData = _reflection.GeneratedProtocolMessageType('ProblemPrivilegedData', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEMPRIVILEGEDDATA,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ProblemPrivilegedData)
  ))
_sym_db.RegisterMessage(ProblemPrivilegedData)

ForecastingHorizon = _reflection.GeneratedProtocolMessageType('ForecastingHorizon', (_message.Message,), dict(
  DESCRIPTOR = _FORECASTINGHORIZON,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ForecastingHorizon)
  ))
_sym_db.RegisterMessage(ForecastingHorizon)

ProblemInput = _reflection.GeneratedProtocolMessageType('ProblemInput', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEMINPUT,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ProblemInput)
  ))
_sym_db.RegisterMessage(ProblemInput)

DataAugmentation = _reflection.GeneratedProtocolMessageType('DataAugmentation', (_message.Message,), dict(
  DESCRIPTOR = _DATAAUGMENTATION,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:DataAugmentation)
  ))
_sym_db.RegisterMessage(DataAugmentation)

ProblemDescription = _reflection.GeneratedProtocolMessageType('ProblemDescription', (_message.Message,), dict(
  DESCRIPTOR = _PROBLEMDESCRIPTION,
  __module__ = 'problem_pb2'
  # @@protoc_insertion_point(class_scope:ProblemDescription)
  ))
_sym_db.RegisterMessage(ProblemDescription)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
