import logging
import os
import pickle
from d3m.container import Dataset, DataFrame
from d3m_ta2_nyu.workflow import database

logger = logging.getLogger(__name__)


@database.with_db
def test(pipeline_id, dataset, storage_dir, results_path, msg_queue, db):
    dataset = Dataset.load(dataset)
    logger.info('Loaded dataset')

    runtime = None
    with open(os.path.join(storage_dir, 'fitted_solution_%s.pkl' % pipeline_id), 'rb') as fin:
        runtime = pickle.load(fin)

    produce_results = runtime.produce(inputs=[dataset])
    produce_results.check_success()

    if results_path is not None and isinstance(produce_results.values['outputs.0'], DataFrame):
        logger.info('Storing predictions at %s', results_path)
        produce_results.values['outputs.0'].sort_values(by=['d3mIndex']).to_csv(results_path, index=False)
    else:
        logger.info('NOT storing predictions')
