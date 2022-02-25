import logging
import os
import pickle
from os.path import join
from d3m.container import Dataset, DataFrame

logger = logging.getLogger(__name__)


def test(pipeline_id, dataset, storage_dir, steps_to_expose, msg_queue):
    dataset = Dataset.load(dataset)
    logger.info('Loaded dataset')

    runtime = None
    with open(os.path.join(storage_dir, 'fitted_solution_%s.pkl' % pipeline_id), 'rb') as fin:
        runtime = pickle.load(fin)

    results = runtime.produce(inputs=[dataset], outputs_to_expose=steps_to_expose)
    results.check_success()

    logger.info('Storing produce results at %s', storage_dir)
    for step_id in results.values:
        if step_id in steps_to_expose and isinstance(results.values[step_id], DataFrame):
            results.values[step_id].to_csv(join(storage_dir, 'produce_%s_%s.csv' % (pipeline_id, step_id)))
