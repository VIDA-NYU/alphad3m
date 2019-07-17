import logging
import sys
import pickle
import d3m_ta2_nyu.proto.core_pb2 as pb_core
from d3m import index
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from d3m_ta2_nyu.common import SCORES_RANKING_ORDER
from d3m_ta2_nyu.pipeline_score import evaluate, score, train_test_tabular_split
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.parameter_tuning.primitive_config import is_tunable
from d3m_ta2_nyu.parameter_tuning.bayesian import HyperparameterTuning, hyperparams_from_config


logger = logging.getLogger(__name__)

PRIMITIVES = index.search()


@database.with_db
def tune(pipeline_id, metrics, problem, do_rank, timeout, targets, msg_queue, db):
    # Load pipeline from database
    pipeline = (
        db.query(database.Pipeline)
        .filter(database.Pipeline.id == pipeline_id)
        .options(joinedload(database.Pipeline.modules),
                 joinedload(database.Pipeline.connections))
    ).one()
    dataset_uri = pipeline.dataset  # TODO: Read the dataset as parameter of tune method

    logger.info("About to tune pipeline, id=%s, dataset=%r, timeout=%d secs", pipeline_id, dataset_uri, timeout)

    # TODO: tune all modules, not only the estimator
    tunable_module = None
    for module in pipeline.modules:
        if is_tunable(module.name):
            tunable_module = module

    if not tunable_module:
        logger.info("No module to be tuned for pipeline %s", pipeline_id)
        sys.exit(1)

    logger.info("Tuning single module %s %s %s",
                tunable_module.id,
                tunable_module.name, tunable_module.package)

    dataset = Dataset.load(dataset_uri)
    tuning = HyperparameterTuning([tunable_module.name])

    def evaluate_tune(hyperparameter_configuration):
        hy = hyperparams_from_config(tunable_module.name, hyperparameter_configuration)
        db.add(database.PipelineParameter(
            pipeline=pipeline,
            module_id=tunable_module.id,
            name='hyperparams',
            value=pickle.dumps(hy),
        ))

        scoring_conf = {'shuffle': 'true',
                        #'stratified': 'true',
                        'train_test_ratio': '0.75',
                        'method': pb_core.EvaluationMethod.Value('HOLDOUT')}

        scores = evaluate(pipeline, train_test_tabular_split, dataset, metrics, problem, scoring_conf)
        logger.info("Tuning results:\n%s", scores)

        # Don't store those runs
        db.rollback()

        return scores[0][metrics[0]['metric']] * SCORES_RANKING_ORDER[metrics[0]['metric']]

    # Run tuning, gets best configuration
    best_configuration = tuning.tune(evaluate_tune, wallclock=timeout)

    # Duplicate pipeline in database
    new_pipeline = database.duplicate_pipeline(db, pipeline, "Hyperparameter tuning from pipeline %s" % pipeline_id)

    # TODO: tune all modules, not only the estimator
    tunable_module = None
    for module in new_pipeline.modules:
        if is_tunable(module.name):
            tunable_module = module

    hy = hyperparams_from_config(tunable_module.name, best_configuration)

    db.add(database.PipelineParameter(
        pipeline=new_pipeline,
        module_id=tunable_module.id,
        name='hyperparams',
        value=pickle.dumps(hy),
    ))
    db.commit()

    logger.info("Tuning done, generated new pipeline %s", new_pipeline.id)

    score(new_pipeline.id, dataset_uri, metrics, problem, None, do_rank, False, None,
          db_filename='/output/supporting_files/db.sqlite3')
    # TODO: Change this static string path

    msg_queue.send(('tuned_pipeline_id', new_pipeline.id))
