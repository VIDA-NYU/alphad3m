"""Unit tests.

Those are supposed to be run without data or primitives available.
"""
import shutil
import tempfile
import unittest
from unittest import mock

from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.ta2 import D3mTa2, Session, TuneHyperparamsJob, TrainJob


class TestSession(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.mkdtemp(prefix='d3m_unittest_')
        cls._ta2 = D3mTa2(storage_root=cls._tmp, logs_root=cls._tmp)
        cls._problem = {
            'about': {'problemID': 'unittest_problem'},
            'inputs': {
            #    'inputs': {
            #        'data': [
            #            {
            #                'targets': [
            #                    {'resID': '0', 'colName': 'targets'},
            #                ],
            #            },
            #        ],
            #    },
                'performanceMetrics': [{'metric': 'f1Macro'}],
            }
        }

        db = cls._ta2.DBSession()
        cls._pipelines = []
        for i in range(4):
            pipeline = database.Pipeline(origin="unittest %d" % i,
                                         dataset='file:///data/test.csv')
            db.add(pipeline)
            mod1 = database.PipelineModule(pipeline=pipeline, name='first',
                                           package='unittest', version='0.0')
            db.add(mod1)
            mod2 = database.PipelineModule(pipeline=pipeline, name='second',
                                           package='unittest', version='0.0')
            db.add(mod2)
            db.add(database.PipelineConnection(pipeline=pipeline,
                                               from_module=mod1,
                                               to_module=mod2,
                                               from_output_name='output',
                                               to_input_name='input'))
            cls._pipelines.append(pipeline)
        db.commit()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp)

    def test_session(self):
        db = self._ta2.DBSession()

        def get_job(call):
            assert len(call[-2]) == 1
            assert not call[-1]
            return call[-2][0]

        ta2 = mock.NonCallableMock()
        session = Session(
            ta2, self._ta2.logs_root,
            self._problem,
            self._ta2.DBSession)
        self._ta2.sessions[session.id] = session

        # No pipelines as yet
        self.assertEqual(session.get_top_pipelines(db, 'F1_MACRO'), [])

        # Add scoring pipelines
        session.add_scoring_pipeline(self._pipelines[0].id)
        session.add_scoring_pipeline(self._pipelines[1].id)
        session.add_scoring_pipeline(self._pipelines[2].id)
        session.check_status()
        ta2._run_queue.put.assert_not_called()

        # No pipeline is scored
        self.assertEqual(session.get_top_pipelines(db, 'F1_MACRO'), [])
        self.assertEqual(session.get_top_pipelines(db, 'F1_MACRO',
                                                   only_trained=False),
                         [])

        # Pipelines finished scoring
        db.add(database.CrossValidation(
            pipeline_id=self._pipelines[0].id,
            scores=[
                database.CrossValidationScore(metric='F1_MACRO',
                                              value=42.0),
                database.CrossValidationScore(metric='EXECUTION_TIME',
                                              value=1.4),
            ],
        ))
        db.add(database.CrossValidation(
            pipeline_id=self._pipelines[1].id,
            scores=[
                database.CrossValidationScore(metric='F1_MACRO',
                                              value=17.0),
                database.CrossValidationScore(metric='EXECUTION_TIME',
                                              value=0.7),
            ],
        ))
        db.commit()

        # Check scores
        self.assertEqual(session.get_top_pipelines(db, 'F1_MACRO',
                                                   only_trained=False),
                         [(self._pipelines[0], 42.0),
                          (self._pipelines[1], 17.0)])
        self.assertEqual(session.get_top_pipelines(db, 'EXECUTION_TIME',
                                                   only_trained=False),
                         [(self._pipelines[1], 0.7),
                          (self._pipelines[0], 1.4)])
        self.assertEqual(session.get_top_pipelines(db, 'ACCURACY',
                                                   only_trained=False),
                         [])

        # Finish scoring
        ta2._run_queue.put.assert_not_called()
        session.pipeline_scoring_done(self._pipelines[0].id)
        ta2._run_queue.put.assert_not_called()
        session.tune_when_ready()
        ta2._run_queue.put.assert_not_called()
        session.pipeline_scoring_done(self._pipelines[1].id)
        ta2._run_queue.put.assert_not_called()
        session.pipeline_scoring_done(self._pipelines[2].id)
        ta2._run_queue.put.assert_called()

        # Check tuning jobs were submitted
        self.assertTrue(all(type(get_job(c)) is TuneHyperparamsJob
                            for c in ta2._run_queue.put.mock_calls))
        self.assertEqual(
            [get_job(c).pipeline_id for c in ta2._run_queue.put.mock_calls],
            [self._pipelines[0].id, self._pipelines[1].id]
        )
        ta2._run_queue.put.reset_mock()

        # Signal tuning is done
        session.pipeline_tuning_done(self._pipelines[0].id)
        ta2._run_queue.put.assert_not_called()
        session.pipeline_tuning_done(self._pipelines[1].id)
        ta2._run_queue.put.assert_called()

        # Check training jobs were submitted
        self.assertTrue(all(type(get_job(c)) is TrainJob
                            for c in ta2._run_queue.put.mock_calls))
        self.assertEqual(
            [get_job(c).pipeline_id for c in ta2._run_queue.put.mock_calls],
            [self._pipelines[0].id, self._pipelines[1].id]
        )
        ta2._run_queue.put.reset_mock()

        # Finish training
        run1 = database.Run(pipeline_id=self._pipelines[1].id,
                            reason='Unittest training',
                            special=False,
                            type=database.RunType.TRAIN)
        db.add(run1)
        db.commit()
        session.pipeline_training_done(self._pipelines[0].id)
        session.pipeline_training_done(self._pipelines[1].id)
        ta2._run_queue.put.assert_not_called()

        # Get top pipelines
        self.assertEqual(session.get_top_pipelines(db, 'F1_MACRO'),
                         [(self._pipelines[1], 17.0)])


if __name__ == '__main__':
    unittest.main()
