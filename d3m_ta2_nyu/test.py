import logging
import os
import pickle
from d3m.container import Dataset
from d3m_ta2_nyu.workflow import database

logger = logging.getLogger(__name__)


@database.with_db
def test(pipeline_id, dataset, storage_dir, msg_queue, db):
    # Load data
    dataset = Dataset.load(dataset)
    logger.info("Loaded dataset")

    runtime = None
    with open(os.path.join(storage_dir, 'fitted_solution_%s.pkl' % pipeline_id), 'rb') as fin:
        runtime = pickle.load(fin)

    produce_results = runtime.produce(inputs=[dataset])
    produce_results.check_success()

    print(produce_results.values)
    # Save 'predictions.csv'?

