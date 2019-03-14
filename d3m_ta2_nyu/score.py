import logging
import numpy
import os
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from d3m_ta2_nyu.pipeline_evaluation import cross_validation, holdout
from d3m_ta2_nyu.workflow import database


logger = logging.getLogger(__name__)

SAMPLE_SIZE = 100
FOLDS = 4
if 'TA2_DEBUG_BE_FAST' in os.environ:
    FOLDS = 2


@database.with_db
def score(pipeline_id, dataset, metrics, problem, method_eval, msg_queue, db):
    # Get pipeline from database
    pipeline = (
        db.query(database.Pipeline)
            .filter(database.Pipeline.id == pipeline_id)
            .options(joinedload(database.Pipeline.modules),
                     joinedload(database.Pipeline.connections))
    ).one()

    logger.info("About to score pipeline, id=%s, metrics=%s, dataset=%r",
                pipeline_id, metrics, dataset)

    # Load data
    dataset = Dataset.load(dataset)
    logger.info("Loaded dataset")

    if method_eval is None:  # come from search phase
        for res_id in dataset:
            if ('https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'
                    in dataset.metadata.query([res_id])['semantic_types']):
                break
        else:
            res_id = next(iter(dataset))
        if (hasattr(dataset[res_id], 'columns') and
                len(dataset[res_id]) > SAMPLE_SIZE):
            # Sample the dataset to stay reasonably fast
            logger.info("Sampling down data from %d to %d",
                        len(dataset[res_id]), SAMPLE_SIZE)
            sample = numpy.concatenate(
                [numpy.repeat(True, SAMPLE_SIZE),
                 numpy.repeat(False, len(dataset[res_id]) - SAMPLE_SIZE)])
            numpy.random.RandomState(seed=65682867).shuffle(sample)
            dataset[res_id] = dataset[res_id][sample]

        method_eval = 'HOLDOUT'

    if method_eval == 'K_FOLD':
        scores = cross_validation(pipeline, dataset, metrics, problem, FOLDS)

        # Store scores
        cross_validation_scores = []
        for fold, fold_scores in scores.items():
            for metric, value in fold_scores.items():
                cross_validation_scores.append(
                    database.CrossValidationScore(fold=fold,
                                                  metric=metric,
                                                  value=value)
                )
        crossval_db = database.CrossValidation(pipeline_id=pipeline_id,
                                               scores=cross_validation_scores)
        db.add(crossval_db)

    elif method_eval == 'HOLDOUT':
        scores = holdout(pipeline, dataset, metrics, problem)

        # Store scores
        holdout_scores = []
        for fold, fold_scores in scores.items():
            for metric, value in fold_scores.items():
                holdout_scores.append(
                    database.CrossValidationScore(fold=fold,
                                                  metric=metric,
                                                  value=value)
                )
        holdout_db = database.CrossValidation(pipeline_id=pipeline_id,
                                              scores=holdout_scores)
        db.add(holdout_db)

    db.commit()
