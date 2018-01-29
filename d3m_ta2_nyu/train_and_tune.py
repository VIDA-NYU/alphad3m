import csv
import logging
import numpy
from sqlalchemy.orm import joinedload
import sys
import time
import pickle

from d3m_ta2_nyu.common import SCORES_TO_SKLEARN, SCORES_RANKING_ORDER
from d3m_ta2_nyu.d3mds import D3MDS
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_train, execute_test
from d3m_ta2_nyu.parameter_tuning.estimator_config import ESTIMATORS
from d3m_ta2_nyu.parameter_tuning.bayesian import HyperparameterTuning, estimator_from_cfg


logger = logging.getLogger(__name__)


FOLDS = 4
RANDOM = 65682867  # The most random of all numbers


def cross_validation(pipeline, metrics, data, targets, progress, db):
    scores = {}

    # For a given idx, ``random_sample[idx]`` indicates which fold will use
    # ``data[idx]`` as test row. All the other ones will use it as a train row.
    # data          = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # random_sample = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # train_split0  = [         3, 4, 5, 6, 7, 8]
    # train_split1  = [0, 1, 2,          6, 7, 8]
    # train_split2  = [0, 1, 2, 3, 4, 5         ]
    # test_split0   = [0, 1, 2                  ]
    # test_split1   = [         3, 4, 5         ]
    # test_split2   = [                  6, 7, 8]
    random_sample = numpy.repeat(numpy.arange(FOLDS),
                                 ((len(data) - 1) // FOLDS) + 1)
    numpy.random.RandomState(seed=RANDOM).shuffle(random_sample)
    random_sample = random_sample[:len(data)]

    all_predictions = []

    for i in range(FOLDS):
        logger.info("Scoring round %d/%d", i + 1, FOLDS)

        progress(i)

        # Do the split
        train_data_split = data[random_sample != i]
        test_data_split = data[random_sample == i]

        train_target_split = targets[random_sample != i]
        test_target_split = targets[random_sample == i]

        start_time = time.time()

        # Run training
        logger.info("Training on fold")
        try:
            train_run, outputs = execute_train(
                db, pipeline, train_data_split, train_target_split,
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

        predictions = next(iter(outputs.values()))['predictions']

        # Compute score
        for metric in metrics:
            # Special case
            if metric == 'EXECUTION_TIME':
                scores.setdefault(metric, []).append(run_time)
            else:
                score_func = SCORES_TO_SKLEARN[metric]
                scores.setdefault(metric, []).append(
                    score_func(test_target_split, predictions))

        # Store predictions
        all_predictions.append(predictions)

    progress(FOLDS)

    # Aggregate results over the folds
    return (
        {metric: numpy.mean(values)
         for metric, values in scores.items()},
        numpy.concatenate(all_predictions)
    )


@database.with_db
def tune(pipeline_id, metrics, dataset, problem, results_path, msg_queue, db):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    logger.info("About to run training pipeline, id=%s, dataset=%r, "
                "problem=%r",
                pipeline_id, dataset, problem)

    # Load data
    ds = D3MDS(dataset, problem)
    logger.info("Loaded dataset, columns: %s",
                ", ".join(col['colName']
                          for col in ds.dataset.get_learning_data_columns()))

    data = ds.get_train_data()
    targets = ds.get_train_targets()

    # Load pipeline from database
    pipeline = (
        db.query(database.Pipeline)
        .filter(database.Pipeline.id == pipeline_id)
        .options(joinedload(database.Pipeline.modules),
                 joinedload(database.Pipeline.connections))
    ).one()

    # TODO: tune all modules, not only the estimator
    estimator_module = None
    for module in pipeline.modules:
        if module.name in ESTIMATORS.keys():
            estimator_module = module

    logger.info("Tuning single module %s %s %s",
                estimator_module.id,
                estimator_module.name, estimator_module.package)

    tuning = HyperparameterTuning(estimator_module.name)

    def evaluate(hyperparameter_configuration):
        estimator = estimator_from_cfg(hyperparameter_configuration,estimator_module.name)
        db.add(database.PipelineParameter(
                            pipeline=pipeline,
                            module_id=estimator_module.id,
                            name='hyperparams',
                            value=pickle.dumps(estimator), 
                        )
            )
        scores, _ = cross_validation(
            pipeline, metrics, data, targets,
            lambda i: None,
            db)

        # Don't store those runs
        db.rollback()

        return scores[metrics[0]] * SCORES_RANKING_ORDER[metrics[0]]

    # Run tuning, gets best configuration
    hyperparameter_configuration = tuning.tune(evaluate)

    # Duplicate pipeline in database
    new_pipeline = database.duplicate_pipeline(
        db, pipeline,
        "Hyperparameter tuning from pipeline %s" % pipeline_id)

    # TODO: tune all modules, not only the estimator
    estimator_module = None
    for module in new_pipeline.modules:
        if module.name in ESTIMATORS.keys():
            estimator_module = module

    estimator = estimator_from_cfg(hyperparameter_configuration,estimator_module.name)
    db.add(database.PipelineParameter(
        pipeline=new_pipeline,
        module_id=estimator_module.id,
        name='hyperparams',
        value=pickle.dumps(estimator),
    ))
    db.flush()

    logger.info("Tuning done, generated new pipeline %s.", new_pipeline.id)

    # Scoring step - make folds, run them through the pipeline one by one
    # (set both training_data and test_data),
    # get predictions from OutputPort to get cross validation scores
    scores, predictions = cross_validation(
        new_pipeline, metrics, data, targets,
        lambda i: None,
        db)
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
    with open(results_path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['d3mIndex', ds.problem.get_targets()[0]['colName']])

        for i, o in zip(data.index, predictions):
            writer.writerow([i, o])

    # Training step - run pipeline on full training_data,
    # Persist module set to write
    logger.info("Running training on full data")

    try:
        execute_train(db, new_pipeline, data, targets)
    except Exception:
        logger.exception("Error running training on full data")
        sys.exit(1)

    db.commit()
