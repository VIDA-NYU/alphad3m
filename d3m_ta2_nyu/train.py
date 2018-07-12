import logging
import numpy
import sys

from d3m.container import Dataset

from d3m_ta2_nyu.crossval import cross_validation
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_train


logger = logging.getLogger(__name__)


FOLDS = 4
RANDOM = 65682867  # The most random of all numbers
MAX_SAMPLE = 50000


@database.with_db
def train(pipeline_id, metrics, targets, results_path, msg_queue, db):
    max_progress = FOLDS + 2.0

    # Get dataset from database
    dataset, = (
        db.query(database.Pipeline.dataset)
        .filter(database.Pipeline.id == pipeline_id)
        .one()
    )

    logger.info("About to run training pipeline, id=%s, dataset=%r",
                pipeline_id, dataset)

    # Load data
    dataset = Dataset.load(dataset)
    logger.info("Loaded dataset")

    if len(dataset['0']) > MAX_SAMPLE:
        # Sample the dataset to stay reasonably fast
        logger.info("Sampling down data from %d to %d",
                    len(dataset['0']), MAX_SAMPLE)
        sample = numpy.concatenate(
            [numpy.repeat(True, MAX_SAMPLE),
             numpy.repeat(False, len(dataset['0']) - MAX_SAMPLE)])
        numpy.random.RandomState(seed=RANDOM).shuffle(sample)
        dataset['0'] = dataset['0'][sample]

    # Scoring step - make folds, run them through the pipeline one by one
    # (set both training_data and test_data),
    # get predictions from OutputPort to get cross validation scores
    scores, predictions = cross_validation(
        pipeline_id, metrics, dataset, targets,
        lambda i: msg_queue.send(('progress', (i + 1.0) / max_progress)),
        db, FOLDS)
    logger.info("Scoring done: %s", ", ".join("%s=%s" % s
                                              for s in scores.items()))

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

    # Training step - run pipeline on full training_data,
    # Persist module set to write
    logger.info("Running training on full data")

    try:
        execute_train(db, pipeline_id, dataset)
    except Exception:
        logger.exception("Error running training on full data")
        sys.exit(1)

    db.commit()
