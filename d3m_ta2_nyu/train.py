import logging
import numpy
import pandas
from sklearn.model_selection import KFold
import sys
import time

from d3m.container import Dataset

from d3m_ta2_nyu.common import SCORES_TO_SKLEARN
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_train, execute_test


logger = logging.getLogger(__name__)


FOLDS = 4
RANDOM = 65682867  # The most random of all numbers

MAX_SAMPLE = 50000


def cross_validation(pipeline, metrics, dataset, targets,
                     progress, db):
    scores = {}

    first_res_id = next(iter(dataset))

    splits = KFold(n_splits=FOLDS, shuffle=True,
                   random_state=RANDOM).split(dataset[first_res_id])

    all_predictions = []

    for i, (train_split, test_split) in enumerate(splits):
        logger.info("Scoring round %d/%d", i + 1, FOLDS)

        progress(i)

        # Do the split
        resources = dict(dataset)
        resources[first_res_id] = resources[first_res_id].iloc[train_split]
        train_data_split = Dataset(resources, dataset.metadata)
        resources = dict(dataset)
        resources[first_res_id] = resources[first_res_id].iloc[test_split]
        test_data_split = Dataset(resources, dataset.metadata)

        start_time = time.time()

        # Run training
        logger.info("Training on fold")
        try:
            train_run, outputs = execute_train(
                db, pipeline, train_data_split,
                crossval=True)
        except Exception:
            logger.exception("Error running training on fold")
            sys.exit(1)
        assert train_run is not None

        # Run prediction
        logger.info("Testing on fold")
        try:
            test_run, outputs = execute_test(
                db, pipeline, test_data_split,
                crossval=True, from_training_run_id=train_run)
        except Exception:
            logger.exception("Error running testing on fold")
            sys.exit(1)

        run_time = time.time() - start_time

        # Get predicted targets
        predictions = next(iter(outputs.values()))['produce']

        # Get expected targets
        test_targets = []
        for resID, col_name in targets:
            test_targets.append(test_data_split[resID].loc[:, col_name])
        test_targets = pandas.concat(test_targets, axis=1)

        # FIXME: Right now pipeline returns a simple array
        # Make it a DataFrame
        predictions = pandas.DataFrame(
            {
                next(iter(targets))[1]: predictions,
                'd3mIndex': test_data_split[next(iter(targets))[0]]['d3mIndex'],
            }
        ).set_index('d3mIndex')

        # Compute score
        for metric in metrics:
            # Special case
            if metric == 'EXECUTION_TIME':
                scores.setdefault(metric, []).append(run_time)
            else:
                score_func = SCORES_TO_SKLEARN[metric]
                scores.setdefault(metric, []).append(
                    score_func(test_targets, predictions))

        # Store predictions
        assert len(predictions.columns) == len(targets)
        predictions.columns = [col_name for resID, col_name in targets]
        all_predictions.append(predictions)

    progress(FOLDS)

    # Aggregate scores over the folds
    scores = {metric: numpy.mean(values) for metric, values in scores.items()}

    # Assemble predictions from each fold
    predictions = pandas.concat(all_predictions, axis=0)

    return scores, predictions


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
        db)
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
        predictions.to_csv(results_path)
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
