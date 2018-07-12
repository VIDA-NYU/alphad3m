import logging
import numpy
import pandas
from sklearn.model_selection import KFold
import sys
import time

from d3m.container import Dataset

from d3m_ta2_nyu.common import SCORES_TO_SKLEARN
from d3m_ta2_nyu.workflow.execute import execute_train, execute_test


logger = logging.getLogger(__name__)


RANDOM = 65682867  # The most random of all numbers


def cross_validation(pipeline, metrics, dataset, targets,
                     progress, db,
                     folds):
    scores = {}

    first_res_id = next(iter(dataset))

    splits = KFold(n_splits=folds, shuffle=True,
                   random_state=RANDOM).split(dataset[first_res_id])

    all_predictions = []

    for i, (train_split, test_split) in enumerate(splits):
        logger.info("Scoring round %d/%d", i + 1, folds)

        progress(i)

        # Do the split
        # FIXME: Use a primitive for this
        resources = dict(dataset)
        resources[first_res_id] = resources[first_res_id].iloc[train_split] \
            .reset_index(drop=True)
        train_data_split = Dataset(resources, dataset.metadata)
        resources = dict(dataset)
        resources[first_res_id] = resources[first_res_id].iloc[test_split] \
            .reset_index(drop=True)
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
        predictions = predictions.set_index('d3mIndex')

        # Get expected targets
        test_targets = [test_data_split['0']['d3mIndex']]
        for resID, col_name in targets:
            test_targets.append(test_data_split[resID].loc[:, col_name])
        test_targets = pandas.concat(test_targets, axis=1) \
            .set_index('d3mIndex')

        assert len(predictions.columns) == len(targets)

        # FIXME: ConstructPredictions doesn't set the right column names
        # https://gitlab.com/datadrivendiscovery/common-primitives/issues/25
        predictions.columns = [col_name for resID, col_name in targets]

        # Compute score
        # FIXME: Use a primitive for this
        for metric in metrics:
            # Special case
            if metric == 'EXECUTION_TIME':
                scores.setdefault(metric, []).append(run_time)
            else:
                score_func = SCORES_TO_SKLEARN[metric]
                scores.setdefault(metric, []).append(
                    score_func(test_targets, predictions))

        # Store predictions
        all_predictions.append(predictions)

    progress(folds)

    # Aggregate scores over the folds
    scores = {metric: numpy.mean(values) for metric, values in scores.items()}

    # Assemble predictions from each fold
    predictions = pandas.concat(all_predictions, axis=0)

    return scores, predictions
