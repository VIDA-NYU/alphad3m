import logging
import os
from sqlalchemy.orm import joinedload

from d3m.container import Dataset

from d3m_ta2_nyu.crossval import cross_validation
from d3m_ta2_nyu.workflow import database


logger = logging.getLogger(__name__)


FOLDS = 4
if 'TA2_DEBUG_BE_FAST' in os.environ:
    FOLDS = 2


@database.with_db
def score(pipeline_id, dataset, metrics, problem, msg_queue, db):
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

    scores = cross_validation(
        pipeline, dataset, metrics, problem,
        FOLDS)

    # Store scores
    cross_validation_scores = []
    for fold, fold_scores in scores.items():
        for metric, value in fold_scores.items():
            cross_validation_scores.append(
                database.CrossValidationScore(fold=fold,
                                              metric=metric,
                                              value=value)
            )
    crossval = database.CrossValidation(pipeline_id=pipeline_id,
                                        scores=cross_validation_scores)
    db.add(crossval)

    db.commit()
