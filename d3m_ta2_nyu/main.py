'''Entrypoints for the TA2.

This contains the multiple entrypoints for the TA2, that get called from
different commands. They spin up a D3mTa2 object and use it.
'''

import json
import logging
import os
import sys
import uuid
from d3m_ta2_nyu.ta2 import D3mTa2


logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:TA2:%(name)s:%(message)s')


def main_search():
    setup_logging()
    timeout = None
    storage_root = None
    pipelines_root = None
    executables_root = None

    if 'D3MTIMEOUT' in os.environ:
        timeout = int(os.environ.get('D3MTIMEOUT')) * 60

    if 'D3MOUTPUTDIR' in os.environ:
        pipelines_root = os.environ['D3MOUTPUTDIR']
        storage_root = os.path.join(os.environ['D3MOUTPUTDIR'], 'supporting_files')
        executables_root = os.path.join(os.environ['D3MOUTPUTDIR'], 'executables')

    logger.info('Config loaded from environment variables D3MOUTPUTDIR=%r D3MTIMEOUT=%r',
                os.environ['D3MOUTPUTDIR'], os.environ.get('D3MTIMEOUT'))
    ta2 = D3mTa2(storage_root=storage_root,
                 pipelines_root=pipelines_root,
                 executables_root=executables_root)
    ta2.run_search(dataset='/input/dataset_TRAIN/datasetDoc.json',
                   problem_path='/input/problem_TRAIN',
                   timeout=timeout)


def main_serve():
    setup_logging()

    port = None
    storage_root = None
    pipelines_root = None
    executables_root = None
    predictions_root = None

    if len(sys.argv) == 2:
        port = int(sys.argv[1])

    if 'D3MOUTPUTDIR' in os.environ:
        pipelines_root = os.environ['D3MOUTPUTDIR']
        storage_root = os.path.join(os.environ['D3MOUTPUTDIR'], 'supporting_files')
        predictions_root = os.path.join(os.environ['D3MOUTPUTDIR'], 'predictions')
        executables_root = os.path.join(os.environ['D3MOUTPUTDIR'], 'executables')

    logger.info('Config loaded from environment variables D3MOUTPUTDIR=%r D3MTIMEOUT=%r',
                os.environ['D3MOUTPUTDIR'], os.environ.get('D3MTIMEOUT'))
    ta2 = D3mTa2(storage_root=storage_root,
                 pipelines_root=pipelines_root,
                 predictions_root=predictions_root,
                 executables_root=executables_root)
    ta2.run_server(port)


def main_test():
    setup_logging()

    storage_root = None
    predictions_root = None

    if 'D3MOUTPUTDIR' in os.environ:
        storage_root = os.path.join(os.environ['D3MOUTPUTDIR'], 'supporting_files')
        predictions_root = os.path.join(os.environ['D3MOUTPUTDIR'], 'predictions')

    logger.info('Config loaded from environment variables D3MOUTPUTDIR=%r',
                os.environ.get('D3MOUTPUTDIR'))
    ta2 = D3mTa2(storage_root=storage_root)
    ta2.run_test(dataset='file://%s' % '/input/dataset_TEST/datasetDoc.json',
                 problem_path='/input/problem_TEST',
                 pipeline_id=uuid.UUID(hex=sys.argv[1]),
                 results_root=os.path.join(predictions_root, sys.argv[1]))
