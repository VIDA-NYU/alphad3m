import logging
import numpy
from sqlalchemy.orm import joinedload
import sys
import pickle
import os
import shutil
from d3m.container import Dataset

from d3m_ta2_nyu.common import SCORES_RANKING_ORDER
from d3m_ta2_nyu.pipeline_score import score
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.parameter_tuning.estimator_config import is_estimator
from d3m_ta2_nyu.parameter_tuning.bayesian import HyperparameterTuning, \
    hyperparams_from_cfg


logger = logging.getLogger(__name__)

RANDOM_SEED = 65682867
MAX_SAMPLE = 100


@database.with_db
def tune(pipeline_id, metrics, problem, targets, results_path, msg_queue, db, timeout=600):
    # Load pipeline from database
    pipeline = (
        db.query(database.Pipeline)
        .filter(database.Pipeline.id == pipeline_id)
        .options(joinedload(database.Pipeline.modules),
                 joinedload(database.Pipeline.connections))
    ).one()
    dataset_uri = pipeline.dataset

    logger.info("About to tune pipeline, id=%s, dataset=%r", pipeline_id, dataset_uri)

    # TODO: tune all modules, not only the estimator
    estimator_module = None
    for module in pipeline.modules:
        if is_estimator(module.name):
            estimator_module = module

    if not estimator_module:
        logger.info("No module to be tuned for pipeline %s", pipeline_id)
        sys.exit(1)

    logger.info("Tuning single module %s %s %s",
                estimator_module.id,
                estimator_module.name, estimator_module.package)

    tuning = HyperparameterTuning([estimator_module.name])

    def evaluate(hyperparameter_configuration):
        hy = hyperparams_from_cfg(estimator_module.name,
                                  hyperparameter_configuration)
        db.add(database.PipelineParameter(
            pipeline=pipeline,
            module_id=estimator_module.id,
            name='hyperparams',
            value=pickle.dumps(hy),
        ))
        scores = score(pipeline_id, dataset_uri, metrics, problem, None, False, False, lambda i: None,
                      db_filename='/output/supporting_files/db.sqlite3')  # TODO Change this static string path

        # Don't store those runs
        db.rollback()

        return scores[metrics[0]['metric']] * SCORES_RANKING_ORDER[metrics[0]['metric']]

    # Run tuning, gets best configuration
    best_configuration = tuning.tune(evaluate, wallclock=timeout)

    # Duplicate pipeline in database
    new_pipeline = database.duplicate_pipeline(
        db, pipeline,
        "Hyperparameter tuning from pipeline %s" % pipeline_id)

    # TODO: tune all modules, not only the estimator
    estimator_module = None
    for module in new_pipeline.modules:
        if is_estimator(module.name):
            estimator_module = module

    hy = hyperparams_from_cfg(estimator_module.name, best_configuration)
    db.add(database.PipelineParameter(
        pipeline=new_pipeline,
        module_id=estimator_module.id,
        name='hyperparams',
        value=pickle.dumps(hy),
    ))
    db.flush()

    logger.info("Tuning done, generated new pipeline %s", new_pipeline.id)
    for f in os.listdir('/tmp'):
        if 'run_1' in f:
            shutil.rmtree(os.path.join('/tmp', f))

    # Score the new pipeline
    scores = score(new_pipeline.id, dataset_uri, metrics, problem, None, False, False, lambda i: None,
                   db_filename='/output/supporting_files/db.sqlite3')  # TODO Change this static string path
    logger.info("Scoring done: %s", ", ".join("%s=%s" % s
                                              for s in scores.items()))

    # Store scores
    scores = [database.CrossValidationScore(metric=metric,
                                            value=numpy.mean(values))
              for metric, values in scores.items()]

    crossval = database.CrossValidation(pipeline_id=new_pipeline.id,
                                        scores=scores)
    db.add(crossval)

    # Store predictions
    if results_path is not None:
        logger.info("Storing predictions at %s", results_path)
        #predictions.sort_index().to_csv(results_path)
    else:
        logger.info("NOT storing predictions")

    db.commit()
    msg_queue.send(('tuned_pipeline_id', new_pipeline.id))
