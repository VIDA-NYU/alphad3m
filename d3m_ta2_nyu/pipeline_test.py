import logging
import os
import subprocess
from d3m_ta2_nyu.workflow import database

logger = logging.getLogger(__name__)


@database.with_db
def test(pipeline_id, dataset, storage_dir, results_path, msg_queue, db):
    logger.info('About to produce pipeline %s', pipeline_id)

    command = [
                'python3', '-m', 'd3m', '--strict-resolving', '--strict-digest',
                'runtime',
                '--volumes', os.environ.get('D3MSTATICDIR', None),
                '--context', 'TESTING',
                '--random-seed', '0',
                'produce',
                '--fitted-pipeline', os.path.join(storage_dir, 'fitted_pipeline_%s.pkl' % pipeline_id),
                '--test-input', dataset[7:],
                '-o', results_path
            ]
    try:
        subprocess.call(command)
        logger.info('Storing produce results at %s', results_path)
    except Exception:
        logger.exception('Error calling produce method for pipeline %s', pipeline_id)
        raise RuntimeError
