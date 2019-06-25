"""Unit tests.

Those are supposed to be run without data or primitives available.
"""

import uuid
import shutil
import tempfile
import unittest
import d3m_ta2_nyu.proto.core_pb2 as pb_core
import d3m_ta2_nyu.proto.value_pb2 as pb_value
import d3m_ta2_nyu.proto.pipeline_pb2 as pb_pipeline
import d3m_ta2_nyu.proto.primitive_pb2 as pb_primitive

from unittest import mock
from d3m_ta2_nyu.ta2 import D3mTa2, Session, TuneHyperparamsJob
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow import convert
from d3m_ta2_nyu.grpc_server import to_timestamp, CoreService


class FakePrimitiveBuilder(object):
    def __init__(self, name):
        self._name = name

    def __call__(self):
        return self

    @property
    def metadata(self):
        return self

    def query(self):
        return {
            'name': self._name.rsplit('.', 1)[-1],
            'id': '%s-mocked' % self._name,
            'digest': '00000000',
            'installation': 'mock',
            'description': "This has been mocked out for unit-testing",
            'version': '0.0',
            'python_path': 'mock'
        }


FakePrimitive = FakePrimitiveBuilder('FakePrimitive')


class TestSession(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self._tmp = tempfile.mkdtemp(prefix='d3m_unittest_')
        self._ta2 = D3mTa2(storage_root=self._tmp,
                           pipelines_root=self._tmp)
        self._problem = {
            'about': {'problemID': 'unittest_problem'},
            'inputs': {
                'data': [
                    {
                        'datasetID': 'unittest_dataset',
                        'targets': [
                            {'resID': '0', 'colName': 'targets'},
                        ],
                    },
                ],
                'performanceMetrics': [{'metric': 'f1Macro'}],
            }
        }

        db = self._ta2.DBSession()
        self._pipelines = []

        for i in range(5):
            pipeline = database.Pipeline(origin="unittest %d" % i,
                                         dataset='file:///data/test.csv')
            db.add(pipeline)
            mod1 = database.PipelineModule(pipeline=pipeline,
                                           name='tests.FakePrimitive',
                                           package='d3m', version='0.0')
            db.add(mod1)
            mod2 = database.PipelineModule(pipeline=pipeline,
                                           name='tests.FakePrimitive',
                                           package='d3m', version='0.0')
            db.add(mod2)
            db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=mod1,
                                               to_module=mod2,
                                               from_output_name='output',
                                               to_input_name='input'))
            self._pipelines.append(pipeline.id)
        db.add(database.CrossValidation(
            pipeline_id=self._pipelines[4],
            scores=[
                database.CrossValidationScore(metric='F1_MACRO',
                                              value=55.0),
                database.CrossValidationScore(metric='EXECUTION_TIME',
                                              value=0.2),
            ],
        ))
        db.commit()
        db.close()

    def tearDown(self):
        shutil.rmtree(self._tmp)

    def test_session(self):
        self._session_test(False)

    def test_session_notuning(self):
        self._session_test(False)

    def _session_test(self, do_tuning):
        db = self._ta2.DBSession()

        def get_job(call):
            assert len(call[-2]) == 1
            assert not call[-1]
            return call[-2][0]

        ta2 = mock.NonCallableMock()
        session = Session(
            ta2,
            self._problem,
            self._ta2.DBSession,
            self._ta2.searched_pipelines,
            self._ta2.scored_pipelines,
            self._ta2.ranked_pipelines
            )
        self._ta2.sessions[session.id] = session

        def compare_scores(ret, expected):
            self.assertEqual(
                [
                    (pipeline.id, score)
                    for (pipeline, score) in ret
                ],
                expected,
            )

        # No pipelines as yet
        self.assertEqual(session.get_top_pipelines(db, 'F1_MACRO'), [])

        # Add scoring pipelines
        session.add_scoring_pipeline(self._pipelines[0])
        session.add_scoring_pipeline(self._pipelines[1])
        session.add_scoring_pipeline(self._pipelines[2])
        session.check_status()
        ta2._run_queue.put.assert_not_called()

        # No pipeline is scored
        compare_scores(session.get_top_pipelines(db, 'F1_MACRO'), [])

        # Pipelines finished scoring
        db.add(database.CrossValidation(
            pipeline_id=self._pipelines[0],
            scores=[
                database.CrossValidationScore(fold=0,
                                              metric='F1_MACRO',
                                              value=41.5),
                database.CrossValidationScore(fold=1,
                                              metric='F1_MACRO',
                                              value=42.5),
                database.CrossValidationScore(fold=0,
                                              metric='EXECUTION_TIME',
                                              value=1.3),
                database.CrossValidationScore(fold=1,
                                              metric='EXECUTION_TIME',
                                              value=1.5),
            ],
        ))
        db.add(database.CrossValidation(
            pipeline_id=self._pipelines[1],
            scores=[
                database.CrossValidationScore(fold=0,
                                              metric='F1_MACRO',
                                              value=16.5),
                database.CrossValidationScore(fold=1,
                                              metric='F1_MACRO',
                                              value=17.5),
                database.CrossValidationScore(fold=0,
                                              metric='EXECUTION_TIME',
                                              value=0.5),
                database.CrossValidationScore(fold=1,
                                              metric='EXECUTION_TIME',
                                              value=0.9),
            ],
        ))
        db.commit()

        # Check scores
        compare_scores(session.get_top_pipelines(db, 'F1_MACRO'),
                       [(self._pipelines[0], 42.0),
                        (self._pipelines[1], 17.0)])
        compare_scores(session.get_top_pipelines(db, 'EXECUTION_TIME'),
                       [(self._pipelines[1], 0.7),
                        (self._pipelines[0], 1.4)])
        compare_scores(session.get_top_pipelines(db, 'ACCURACY'), [])

        # Finish scoring
        ta2._run_queue.put.assert_not_called()
        session.pipeline_scoring_done(self._pipelines[0])
        ta2._run_queue.put.assert_not_called()
        session.tune_when_ready(3 if do_tuning else 0)
        ta2._run_queue.put.assert_not_called()
        session.pipeline_scoring_done(self._pipelines[1])
        ta2._run_queue.put.assert_not_called()
        session.pipeline_scoring_done(self._pipelines[2])

        # Check tuning jobs were submitted
        if do_tuning:
            ta2._run_queue.put.assert_called()
            self.assertTrue(all(type(get_job(c)) is TuneHyperparamsJob
                                for c in ta2._run_queue.put.mock_calls))
            self.assertEqual(
                [get_job(c).pipeline_id
                 for c in ta2._run_queue.put.mock_calls],
                [self._pipelines[0], self._pipelines[1]]
            )

            # Add tuned pipeline scores
            db.add(database.CrossValidation(
                pipeline_id=self._pipelines[2],
                scores=[
                    database.CrossValidationScore(fold=0,
                                                  metric='F1_MACRO',
                                                  value=21.0),
                    database.CrossValidationScore(fold=1,
                                                  metric='F1_MACRO',
                                                  value=22.0),
                    database.CrossValidationScore(fold=0,
                                                  metric='EXECUTION_TIME',
                                                  value=0.9),
                    database.CrossValidationScore(fold=1,
                                                  metric='EXECUTION_TIME',
                                                  value=1.1),
                ],
            ))
            db.commit()

            # Signal tuning is done
            ta2._run_queue.put.reset_mock()
            session.pipeline_tuning_done(self._pipelines[0],
                                         self._pipelines[2])
            session.pipeline_tuning_done(self._pipelines[1])
            ta2._run_queue.put.assert_not_called()
        else:
            ta2._run_queue.put.assert_not_called()

        ta2._run_queue.put.assert_not_called()

        # Get top pipelines
        if do_tuning:
            compare_scores(session.get_top_pipelines(db, 'F1_MACRO'),
                           [(self._pipelines[0], 42.0),
                            (self._pipelines[2], 21.5),
                            (self._pipelines[1], 17.0)])
        else:
            compare_scores(session.get_top_pipelines(db, 'F1_MACRO'),
                           [(self._pipelines[0], 42.0),
                            (self._pipelines[1], 17.0)])


class TestPipelineConversion(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix='d3m_unittest_')
        cls._ta2 = D3mTa2(storage_root=cls._tmp,
                          pipelines_root=cls._tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp)

    def test_convert_classification_template(self):
        def seq_uuid():
            seq_uuid.count += 1
            return uuid.UUID(int=seq_uuid.count)

        seq_uuid.count = 0

        # Build a pipeline using the first template
        func, imputer, classifier = self._ta2.TEMPLATES['CLASSIFICATION'][0]
        with mock.patch('uuid.uuid4', seq_uuid):
            pipeline_id = func(
                self._ta2, imputer, classifier,
                'file:///nonexistent/datasetDoc.json',
                {('0', 'target')}, {('0', 'attr1'), ('0', 'attr2')})

        # Convert it
        db = self._ta2.DBSession()
        pipeline = db.query(database.Pipeline).get(pipeline_id)
        with mock.patch('d3m_ta2_nyu.workflow.convert.get_class',
                        FakePrimitiveBuilder):
            pipeline_json = convert.to_d3m_json(pipeline)

        created = pipeline_json['created']
        self.assertRegex(created,
                         '20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]T'
                         '[0-9][0-9]:[0-9][0-9]:[0-9][0-9]Z')

        # Check output
        self.assertEqual(
            pipeline_json,
            {
                'id': '00000000-0000-0000-0000-000000000001',
                'name': '00000000-0000-0000-0000-000000000001',
                'description': (
                    'classification_template('
                    'imputer=d3m.primitives.data_cleaning.imputer.SKlearn, '
                    'classifier=d3m.primitives.classification.'
                    'random_forest.SKlearn)'
                ),
                'schema': 'https://metadata.datadrivendiscovery.org/schemas/'
                          'v0/pipeline.json',
                'created': created,
                'context': 'TESTING',
                'inputs': [{'name': 'input dataset'}],
                'outputs': [
                    {'data': 'steps.11.produce', 'name': 'predictions'},
                ],
                'steps': [
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'denormalize.Common-mocked',
                            'name': 'Common',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'inputs.0',
                                'type': 'CONTAINER',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'dataset_to_dataframe.Common-mocked',
                            'name': 'Common',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.0.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'column_parser.DataFrameCommon-mocked',
                            'name': 'DataFrameCommon',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.1.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'extract_columns_by_semantic_types.DataFrameCommon-mocked',
                            'name': 'DataFrameCommon',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.2.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'hyperparams': {
                            'semantic_types': {
                                'data': ['https://metadata.datadrivendiscovery'
                                         '.org/types/Attribute'],
                                'type': 'VALUE',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_cleaning.'
                                  'imputer.SKlearn-mocked',
                            'name': 'SKlearn',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.3.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'hyperparams': {
                            'strategy': {
                                'type': 'VALUE',
                                'data': 'most_frequent'
                            }
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'one_hot_encoder.SKlearn-mocked',
                            'name': 'SKlearn',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.4.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'hyperparams': {
                            'handle_unknown': {
                                'type': 'VALUE',
                                'data': 'ignore'
                            }
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'cast_to_type.Common-mocked',
                            'name': 'Common',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.5.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'hyperparams': {
                            'type_to_cast': {
                                'data': 'float',
                                'type': 'VALUE',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'extract_columns_by_semantic_types.DataFrameCommon-mocked',
                            'name': 'DataFrameCommon',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.2.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'hyperparams': {
                            'semantic_types': {
                                'data': ['https://metadata.datadrivendiscovery'
                                         '.org/types/Target'],
                                'type': 'VALUE',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'cast_to_type.Common-mocked',
                            'name': 'Common',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.7.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.classification.'
                                  'random_forest.SKlearn-mocked',
                            'name': 'SKlearn',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.6.produce',
                                'type': 'CONTAINER',
                            },
                            'outputs': {
                                'data': 'steps.8.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'extract_columns_by_semantic_types.DataFrameCommon-mocked',
                            'name': 'DataFrameCommon',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.2.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'hyperparams': {
                            'semantic_types': {
                                'data': [
                                    'https://metadata.datadrivendiscovery.org/'
                                    'types/Target',
                                    'https://metadata.datadrivendiscovery.org/'
                                    'types/PrimaryKey',
                                ],
                                'type': 'VALUE',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                    {
                        'type': 'PRIMITIVE',
                        'primitive': {
                            'id': 'd3m.primitives.data_transformation.'
                                  'construct_predictions.DataFrameCommon-mocked',
                            'name': 'DataFrameCommon',
                            'digest': '00000000',
                            'version': '0.0',
                            'python_path': 'mock',
                        },
                        'arguments': {
                            'inputs': {
                                'data': 'steps.9.produce',
                                'type': 'CONTAINER',
                            },
                            'reference': {
                                'data': 'steps.10.produce',
                                'type': 'CONTAINER',
                            },
                        },
                        'outputs': [{'id': 'produce'}],
                    },
                ],
            },
        )


class TestDescribeSolution(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix='d3m_unittest_')
        cls._ta2 = D3mTa2(storage_root=cls._tmp,
                          pipelines_root=cls._tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp)

    def test_convert_classification_template(self):
        def seq_uuid():
            seq_uuid.count += 1
            return uuid.UUID(int=seq_uuid.count)

        seq_uuid.count = 0

        # Build a pipeline using the first template
        func, imputer, classifier = self._ta2.TEMPLATES['CLASSIFICATION'][0]
        with mock.patch('uuid.uuid4', seq_uuid):
            pipeline_id = func(
                self._ta2, imputer, classifier,
                'file:///nonexistent/datasetDoc.json',
                {('0', 'target')}, {('0', 'attr1'), ('0', 'attr2')})

        # Convert it
        db = self._ta2.DBSession()
        pipeline = db.query(database.Pipeline).get(pipeline_id)

        with mock.patch('d3m_ta2_nyu.workflow.convert.get_class',
                        FakePrimitiveBuilder):
            solution = CoreService(self._ta2).DescribeSolution(pb_core.DescribeSolutionRequest(
                solution_id=pipeline_id.hex
            ), None)

        solution_mock = pb_core.DescribeSolutionResponse(
            pipeline=pb_pipeline.PipelineDescription(
                id=str(pipeline.id),
                name=str(pipeline.id),
                description=pipeline.origin or '',
                created=to_timestamp(pipeline.created_date),
                context=pb_pipeline.TESTING,
                inputs=[
                    pb_pipeline.PipelineDescriptionInput(
                        name="input dataset"
                    )
                ],
                outputs=[
                    pb_pipeline.PipelineDescriptionOutput(
                        name="predictions",
                        data='steps.11.produce'
                    )
                ],
                steps=[
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.denormalize.Common-mocked',
                                version='0.0',
                                python_path='mock',
                                name='Common',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='inputs.0',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={},
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.dataset_to_dataframe.Common-mocked',
                                version='0.0',
                                python_path='mock',
                                name='Common',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.0.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={},
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.column_parser.DataFrameCommon-mocked',
                                version='0.0',
                                python_path='mock',
                                name='DataFrameCommon',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.1.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={},
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.'
                                   'extract_columns_by_semantic_types.DataFrameCommon-mocked',
                                version='0.0',
                                python_path='mock',
                                name='DataFrameCommon',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.2.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={
                                'semantic_types': pb_pipeline.PrimitiveStepHyperparameter(
                                    value=pb_pipeline.ValueArgument(
                                        data=pb_value.Value(
                                            raw=pb_value.ValueRaw(
                                                string=repr(['https://metadata.datadrivendiscovery.org/'
                                                             'types/Attribute'])
                                            )
                                        )
                                    )
                                )
                            }
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id=imputer + '-mocked',
                                version='0.0',
                                python_path='mock',
                                name=imputer.rsplit('.', 1)[-1],
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.3.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={
                                'strategy': pb_pipeline.PrimitiveStepHyperparameter(
                                    value=pb_pipeline.ValueArgument(
                                        data=pb_value.Value(
                                            raw=pb_value.ValueRaw(
                                                string='most_frequent',
                                            ),
                                        ),
                                    ),
                                ),
                            },
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.one_hot_encoder.SKlearn-mocked',
                                version='0.0',
                                python_path='mock',
                                name='SKlearn',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.4.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={
                                'handle_unknown': pb_pipeline.PrimitiveStepHyperparameter(
                                    value=pb_pipeline.ValueArgument(
                                        data=pb_value.Value(
                                            raw=pb_value.ValueRaw(
                                                string='ignore',
                                            ),
                                        ),
                                    ),
                                ),
                            },
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.cast_to_type.Common-mocked',
                                version='0.0',
                                python_path='mock',
                                name='Common',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.5.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={
                                'type_to_cast': pb_pipeline.PrimitiveStepHyperparameter(
                                    value=pb_pipeline.ValueArgument(
                                        data=pb_value.Value(
                                            raw=pb_value.ValueRaw(
                                                string='float',
                                            ),
                                        ),
                                    ),
                                ),
                            },
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.'
                                   'extract_columns_by_semantic_types.DataFrameCommon-mocked',
                                version='0.0',
                                python_path='mock',
                                name='DataFrameCommon',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.2.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={
                                'semantic_types': pb_pipeline.PrimitiveStepHyperparameter(
                                    value=pb_pipeline.ValueArgument(
                                        data=pb_value.Value(
                                            raw=pb_value.ValueRaw(
                                                string=repr(['https://metadata.datadrivendiscovery.org/'
                                                             'types/Target'])
                                            )
                                        )
                                    )
                                )
                            }
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.cast_to_type.Common-mocked',
                                version='0.0',
                                python_path='mock',
                                name='Common',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.7.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={},
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id=classifier + '-mocked',
                                version='0.0',
                                python_path='mock',
                                name=classifier.rsplit('.', 1)[-1],
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.6.produce',
                                    )
                                ),
                                'outputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.8.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={},
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.'
                                   'extract_columns_by_semantic_types.DataFrameCommon-mocked',
                                version='0.0',
                                python_path='mock',
                                name='DataFrameCommon',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.2.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={
                                'semantic_types': pb_pipeline.PrimitiveStepHyperparameter(
                                    value=pb_pipeline.ValueArgument(
                                        data=pb_value.Value(
                                            raw=pb_value.ValueRaw(
                                                string=repr(['https://metadata.datadrivendiscovery.org/'
                                                             'types/Target',
                                                            'https://metadata.datadrivendiscovery.org/'
                                                            'types/PrimaryKey']),
                                            )
                                        )
                                    )
                                )
                            }
                        )
                    ),
                    pb_pipeline.PipelineDescriptionStep(
                        primitive=pb_pipeline.PrimitivePipelineDescriptionStep(
                            primitive=pb_primitive.Primitive(
                                id='d3m.primitives.data_transformation.construct_predictions.DataFrameCommon-mocked',
                                version='0.0',
                                python_path='mock',
                                name='DataFrameCommon',
                                digest='00000000'
                            ),
                            arguments={
                                'inputs': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.9.produce',
                                    )
                                ),
                                'reference': pb_pipeline.PrimitiveStepArgument(
                                    container=pb_pipeline.ContainerArgument(
                                        data='steps.10.produce',
                                    )
                                )
                            },
                            outputs=[
                                pb_pipeline.StepOutput(
                                    id='produce'
                                )
                            ],
                            hyperparams={},
                        )
                    )
                ],
            ),
            steps=[
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
                pb_core.StepDescription(
                    primitive=pb_core.PrimitiveStepDescription(
                        hyperparams={}
                    )
                ),
            ]
        )

        # Check output
        self.assertEqual(solution, solution_mock)


if __name__ == '__main__':
    unittest.main()
