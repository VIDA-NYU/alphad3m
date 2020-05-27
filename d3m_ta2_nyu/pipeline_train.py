import os
import logging
import pickle
import d3m.runtime
import d3m.metadata.base
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from d3m.metadata import base as metadata_base
from d3m_ta2_nyu.workflow import database, convert


logger = logging.getLogger(__name__)


@database.with_db
def train(pipeline_id, dataset, problem, storage_dir, results_path, msg_queue, db):
    # Get pipeline from database
    pipeline = (
        db.query(database.Pipeline)
            .filter(database.Pipeline.id == pipeline_id)
            .options(joinedload(database.Pipeline.modules),
                     joinedload(database.Pipeline.connections))
    ).one()

    logger.info('About to train pipeline, id=%s, dataset=%r',
                pipeline_id, dataset)

    # Load data
    dataset = Dataset.load(dataset)
    logger.info('Loaded dataset')

    # Training step - fit pipeline on training data
    logger.info('Running training')

    d3m_pipeline = d3m.metadata.pipeline.Pipeline.from_json_structure(
        convert.to_d3m_json(pipeline),
    )

    fitted_pipeline, predictions, result = d3m.runtime.fit(d3m_pipeline, [dataset], problem_description=problem,
                                                           context=metadata_base.Context.TESTING,
                                                           volumes_dir=os.environ.get('D3MSTATICDIR', None),
                                                           random_seed=0)

    result.check_success()

    if results_path is not None:
        logger.info('Storing fit results at %s', results_path)
        predictions.to_csv(results_path)
    else:
        logger.info('NOT storing fit results')

    with open(os.path.join(storage_dir, 'fitted_solution_%s.pkl' % pipeline_id), 'wb') as fout:
        pickle.dump(fitted_pipeline, fout)
