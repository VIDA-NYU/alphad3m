import csv
import logging
import sys

from d3mds import D3MDS

from d3m_ta2_nyu.utils import with_db
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.workflow.execute import execute_test


logger = logging.getLogger(__name__)


engine, DBSession = database.connect()


@with_db(DBSession)
def test(pipeline_id, dataset, problem, results_path, db):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    # Load data
    ds = D3MDS(dataset, problem)
    logger.info("Loaded dataset, columns: %r",
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

    predictions = next(iter(outputs.values()))

    with open(results_path, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['d3mIndex', ds.problem.get_targets()[0]['colName']])

        for i, o in zip(data.index, predictions):
            writer.writerow([i, o])
