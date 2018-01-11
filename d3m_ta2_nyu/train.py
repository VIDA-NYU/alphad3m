import logging
import numpy
import sys
import time

from d3mds import D3MDS

from d3m_ta2_nyu.common import SCORES_TO_SKLEARN
from d3m_ta2_nyu.utils import with_db
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_train, execute_test


logger = logging.getLogger(__name__)


engine, DBSession = database.connect()


FOLDS = 3
SPLIT_RATIO = 0.25


@with_db(DBSession)
def train(pipeline_id, metrics, dataset, problem, msg_queue, db):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")
    max_progress = FOLDS + 2.0

    logger.info("About to run training pipeline, id=%s, dataset=%r",
                pipeline_id, dataset)

    # Load data
    ds = D3MDS(dataset, problem)
    logger.info("Loaded dataset, columns: %r",
                ", ".join(col['colName']
                          for col in ds.dataset.get_learning_data_columns()))

    data = ds.get_train_data()
    targets = ds.get_train_targets()

    # Scoring step - make folds, run them through the pipeline one by one
    # (set both training_data and test_data),
    # get predictions from OutputPort to get cross validation scores
    scores = {}

    for i in range(FOLDS):
        logger.info("Scoring round %d/%d", i + 1, FOLDS)

        msg_queue.put((pipeline_id, 'progress', (i + 1.0) / max_progress))

        # Do the split
        random_sample = numpy.random.rand(len(data)) < SPLIT_RATIO

        train_data_split = data[random_sample]
        test_data_split = data[~random_sample]

        train_target_split = targets[random_sample]
        test_target_split = targets[~random_sample]

        start_time = time.time()

        # Run training
        try:
            train_run, outputs = execute_train(
                db, pipeline_id, train_data_split, train_target_split,
                crossval=True)
        except Exception:
            logger.exception("Error runnin training on fold")
            sys.exit(1)

        # Run prediction
        try:
            test_run, outputs = execute_test(
                db, pipeline_id, test_data_split,
                crossval=True, from_training_run_id=train_run)
        except Exception:
            logger.exception("Error running testing on fold")
            sys.exit(1)

        run_time = time.time() - start_time

        predictions = next(iter(outputs.values()))

        # Compute score
        for metric in metrics:
            # Special case
            if metric == 'EXECUTION_TIME':
                scores.setdefault(metric, []).append(run_time)
            else:
                score_func = SCORES_TO_SKLEARN[metric]
                scores.setdefault(metric, []).append(
                    score_func(test_target_split, predictions))

    msg_queue.put((pipeline_id, 'progress', (FOLDS + 1.0) / max_progress))

    # Aggregate results over the folds
    scores = dict((metric, numpy.mean(values))
                  for metric, values in scores.items())
    msg_queue.put((pipeline_id, 'scores', scores))

    # Training step - run pipeline on full training_data,
    # sink = classifier-sink (the Persist downstream of the classifier),
    # Persist module set to write
    logger.info("Scoring done, running training on full data")

    try:
        execute_train(db, pipeline_id, data, targets)
    except Exception:
        logger.exception("Error running training on full data")
        sys.exit(1)

    db.commit()
