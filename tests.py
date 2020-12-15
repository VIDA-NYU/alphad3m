"""Unit tests.

Those are supposed to be run without data or primitives available.
"""

import shutil
import tempfile
import unittest
from unittest import mock
from d3m_ta2_nyu.ta2 import D3mTa2, Session, TuneHyperparamsJob
from d3m_ta2_nyu.workflow import database
from d3m.metadata.problem import PerformanceMetric


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
        self._ta2 = D3mTa2(self._tmp)
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
            },
            'problem': {'performance_metrics': [{'metric': PerformanceMetric.F1_MACRO}]}
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
            self._ta2.output_folder,
            self._ta2.DBSession
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
        self.assertEqual(session.get_top_pipelines(db, PerformanceMetric.F1_MACRO), [])

        # Add scoring pipelines
        session.add_scoring_pipeline(self._pipelines[0])
        session.add_scoring_pipeline(self._pipelines[1])
        session.add_scoring_pipeline(self._pipelines[2])
        session.check_status()
        ta2._run_queue.put.assert_not_called()

        # No pipeline is scored
        compare_scores(session.get_top_pipelines(db, PerformanceMetric.F1_MACRO), [])

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
        compare_scores(session.get_top_pipelines(db, PerformanceMetric.F1_MACRO),
                       [(self._pipelines[0], 42.0),
                        (self._pipelines[1], 17.0)])
        compare_scores(session.get_top_pipelines(db, PerformanceMetric.ACCURACY), [])

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
            compare_scores(session.get_top_pipelines(db, PerformanceMetric.F1_MACRO),
                           [(self._pipelines[0], 42.0),
                            (self._pipelines[2], 21.5),
                            (self._pipelines[1], 17.0)])
        else:
            compare_scores(session.get_top_pipelines(db, PerformanceMetric.F1_MACRO),
                           [(self._pipelines[0], 42.0),
                            (self._pipelines[1], 17.0)])


if __name__ == '__main__':
    unittest.main()
