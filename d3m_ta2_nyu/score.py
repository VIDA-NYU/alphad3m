import logging
import numpy
import os

from d3m.container import Dataset

from d3m_ta2_nyu.crossval import cross_validation
from d3m_ta2_nyu.workflow import database


logger = logging.getLogger(__name__)


FOLDS = 4
if 'TA2_DEBUG_BE_FAST' in os.environ:
    FOLDS = 2


@database.with_db
def score(pipeline_id, metrics, targets, results_path, msg_queue, db):
    max_progress = FOLDS

    # Get dataset from database
    dataset, = (
        db.query(database.Pipeline.dataset)
        .filter(database.Pipeline.id == pipeline_id)
        .one()
    )

    logger.info("About to score pipeline, id=%s, dataset=%r",
                pipeline_id, dataset)

    # Load data
    dataset = Dataset.load(dataset)
    logger.info("Loaded dataset")

    scores, predictions = cross_validation(
        pipeline_id, metrics, dataset, targets,
        lambda i: msg_queue.send(('progress', i / max_progress)),
        db, FOLDS)
    logger.info("Scoring done: %s", ", ".join("%s=%s" % s for s in scores.items()))

    # Store scores
    scores = [database.CrossValidationScore(metric=metric,
                                            value=numpy.mean(values))
              for metric, values in scores.items()]
    crossval = database.CrossValidation(pipeline_id=pipeline_id, scores=scores)

    db.add(crossval)

    # Store predictions
    if results_path is not None:
        logger.info("Storing predictions at %s", results_path)
        predictions.sort_index().to_csv(results_path)
    else:
        logger.info("NOT storing predictions")

    db.commit()
