import logging
import numpy
import pandas
import pickle
from sqlalchemy import select
import sys
import time

from d3m.primitives.evaluation import TrainScoreDatasetSplit

from d3m_ta2_nyu.common import SCORES_TO_SKLEARN
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_train, execute_test
from d3m_ta2_nyu.workflow.module_loader import load_dataset

logger = logging.getLogger(__name__)

RANDOM = 65682867  # The most random of all numbers
MAX_SAMPLE = 100

def cross_validation(pipeline, metrics, dataset, targets,
                     progress, db,
                     folds, stratified=False, shuffle=True):
    folds = 1
    scores = {}

    if targets is None:
        # Load targets from database
        module = (
            select([database.PipelineModule.id])
                .where(database.PipelineModule.pipeline_id == pipeline)
                .where(database.PipelineModule.package == 'data')
                .where(database.PipelineModule.name == 'dataset')
        ).as_scalar()
        targets, = (
            db.query(database.PipelineParameter.value)
                .filter(database.PipelineParameter.module_id == module)
                .filter(database.PipelineParameter.name == 'targets')
        ).one()
        targets = pickle.loads(targets)

    # Set correct targets
    dataset = load_dataset(dataset, targets, None) # FIXME is it loaded each time is called?

    if MAX_SAMPLE:
        for res_id in dataset:
            if ('https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'
                    in dataset.metadata.query([res_id])['semantic_types']):
                break
            else:
                res_id = next(iter(dataset))

        if hasattr(dataset[res_id], 'columns') and len(dataset[res_id]) > MAX_SAMPLE:
            logger.info("Sampling down data from %d to %d",
                        len(dataset[res_id]), MAX_SAMPLE)
            sample = numpy.concatenate(
                [numpy.repeat(True, MAX_SAMPLE),
                 numpy.repeat(False, len(dataset[res_id]) - MAX_SAMPLE)])
            numpy.random.RandomState(seed=RANDOM).shuffle(sample)
            dataset[res_id] = dataset[res_id][sample]


    # Do the split
    m = TrainScoreDatasetSplit.metadata.query()['primitive_code']
    SplitHyperparams = m['class_type_arguments']['Hyperparams']
    hyperparams = SplitHyperparams(SplitHyperparams.defaults(),
                                   stratified=stratified,
                                   shuffle=shuffle,
                                   delete_recursive=True)
    kfold = TrainScoreDatasetSplit(hyperparams=hyperparams,
                                   random_seed=RANDOM)
    kfold.set_training_data(dataset=dataset)
    kfold.fit()
    train_splits = kfold.produce(inputs=list(range(folds)))
    assert train_splits.has_finished
    test_splits = kfold.produce_score_data(inputs=list(range(folds)))
    assert test_splits.has_finished
    splits = zip(train_splits.value, test_splits.value)

    all_predictions = []

    for i, (train_split, test_split) in enumerate(splits):
        logger.info("Scoring round %d/%d", i + 1, folds)

        progress(i)

        start_time = time.time()

        # Run training
        logger.info("Training on fold")

        try:
            train_run, outputs = execute_train(
                db, pipeline, train_split,
                crossval=True)

        except Exception:
            logger.exception("Error running training on fold")
            sys.exit(1)
        assert train_run is not None

        # Run prediction
        logger.info("Testing on fold")
        try:
            test_run, outputs = execute_test(
                db, pipeline, test_split,
                crossval=True, from_training_run_id=train_run)
        except Exception:
            logger.exception("Error running testing on fold")
            sys.exit(1)
        run_time = time.time() - start_time

        # Get predicted targets
        predictions = next(iter(outputs.values()))['produce']
        predictions = predictions.set_index('d3mIndex')

        # Get expected targets
        for res_id in dataset:
            if ('https://metadata.datadrivendiscovery.org/'
                'types/DatasetEntryPoint'
                    in dataset.metadata.query([res_id])['semantic_types']):
                break
            else:
                res_id = next(iter(dataset))
        test_targets = [test_split[res_id]['d3mIndex']]

        for resID, col_name in targets:
            test_targets.append(test_split[resID].loc[:, col_name])

        test_targets = pandas.concat(test_targets, axis=1) \
            .set_index('d3mIndex')

        assert len(predictions.columns) == len(targets)

        # FIXME: ConstructPredictions doesn't set the right column names
        # https://gitlab.com/datadrivendiscovery/common-primitives/issues/25
        predictions.columns = [col_name for resID, col_name in targets]

        # print(predictions.columns)

        # Compute score
        # FIXME: Use a primitive for this
        for metric in metrics:
            # Special case
            if metric == 'EXECUTION_TIME':
                scores.setdefault(metric, []).append(run_time)
            else:
                score_func = SCORES_TO_SKLEARN[metric]
                scores.setdefault(metric, []).append(
                    score_func(test_targets.values.flatten(), predictions.values.flatten()))

        # Store predictions
        all_predictions.append(predictions)
        break

    progress(folds)

    # Aggregate scores over the folds
    scores = {metric: numpy.mean(values) for metric, values in scores.items()}

    # Assemble predictions from each fold
    predictions = pandas.concat(all_predictions, axis=0)

    return scores, predictions
