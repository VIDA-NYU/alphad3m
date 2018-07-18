import logging
import pandas
import sys

from d3m.container import Dataset

from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_test


logger = logging.getLogger(__name__)


@database.with_db
def test(pipeline_id, dataset, targets, results_path, msg_queue, db):
    # Load data
    dataset = Dataset.load(dataset)
    logger.info("Loaded dataset")

    # Run prediction
    try:
        test_run, outputs = execute_test(
            db, pipeline_id, dataset)
    except Exception:
        logger.exception("Error running testing")
        sys.exit(1)

    # Get predicted targets
    predictions = next(iter(outputs.values()))['produce']
    predictions = predictions.set_index('d3mIndex')

    assert len(predictions.columns) == len(targets)

    # FIXME: ConstructPredictions doesn't set the right column names
    # https://gitlab.com/datadrivendiscovery/common-primitives/issues/25
    predictions.columns = [col_name for resID, col_name in targets]

    # Store predictions
    if results_path is not None:
        logger.info("Storing predictions at %s", results_path)
        predictions.sort_index().to_csv(results_path)
    else:
        logger.info("NOT storing predictions")

    db.commit()
