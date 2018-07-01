import logging
import pandas
import sys

from d3m.container import Dataset

from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_test


logger = logging.getLogger(__name__)


@database.with_db
def test(pipeline_id, dataset, targets, results_path, db):
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

    # FIXME: Right now pipeline returns a simple array
    # Make it a DataFrame
    predictions = pandas.DataFrame(
        {
            next(iter(targets))[1]: predictions,
            'd3mIndex': dataset[next(iter(targets))[0]]['d3mIndex'],
        }
    ).set_index('d3mIndex')

    assert len(predictions.columns) == len(targets)
    predictions.columns = [col_name for resID, col_name in targets]
    predictions.to_csv(results_path)
