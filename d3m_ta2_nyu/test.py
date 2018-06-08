import logging
import sys

from d3m_ta2_nyu.d3mds import D3MDS
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_test


logger = logging.getLogger(__name__)


@database.with_db
def test(pipeline_id, dataset, problem, results_path, db, use_all_rows=False):
    # Load data
    ds = D3MDS(dataset, problem)
    logger.info("Loaded dataset, columns: %s",
                ", ".join(col['colName']
                          for col in ds.dataset.get_learning_data_columns()))

    if not use_all_rows:
        data = ds.get_test_data()
    else:
        logger.info("Ignoring dataSplits, testing on all rows")
        data = ds.dataset.get_learning_data(view=None, problem=ds.problem)
        target_cols = ds._get_target_columns(data)
        data.drop(data.columns[target_cols], axis=1, inplace=True)

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
