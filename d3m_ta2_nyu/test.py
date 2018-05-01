import logging
import sys

from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_test


logger = logging.getLogger(__name__)


@database.with_db
def test(pipeline_id, dataset, problem, results_path, db):
    # Load data
    ds = D3MDS(dataset, problem)
    logger.info("Loaded dataset, columns: %s",
                ", ".join(col['colName']
                          for col in ds.dataset.get_learning_data_columns()))

    data = ds.get_test_data()

    # Run prediction
    try:
        test_run, outputs = execute_test(
            db, pipeline_id, data)
    except Exception:
        logger.exception("Error running testing")
        sys.exit(1)

    target_names = [t['colName'] for t in ds.problem.get_targets()]
    predictions = next(iter(outputs.values()))['predictions']
    assert len(predictions.columns) == len(target_names)
    predictions.columns = target_names
    predictions.to_csv(results_path)
