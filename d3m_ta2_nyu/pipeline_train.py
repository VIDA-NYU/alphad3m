import os
import logging
import json
import tempfile
import pickle
import d3m.runtime
import d3m.metadata.base
from sqlalchemy.orm import joinedload
from d3m.container import Dataset, DataFrame
from d3m.metadata import base as metadata_base
from d3m.metadata.problem import Problem
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

    d3m_problem = None
    if problem is not None:
        # Convert problem description to core package format
        # FIXME: There isn't a way to parse from JSON data, so write it to a file
        # and read it back
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'problemDoc.json')
            with open(tmp_path, 'w', encoding='utf8') as fin:
                json.dump(problem, fin)
            d3m_problem = Problem.load('file://' + tmp_path)

    runtime = d3m.runtime.Runtime(pipeline=d3m_pipeline, problem_description=d3m_problem,
                                  context=metadata_base.Context.TESTING,
                                  volumes_dir=os.environ.get('D3MSTATICDIR', None))

    # Fitting pipeline on input dataset.
    fit_results = runtime.fit(inputs=[dataset])
    fit_results.check_success()

    if results_path is not None and isinstance(fit_results.values['outputs.0'], DataFrame):
        logger.info('Storing fit results at %s', results_path)
        fit_results.values['outputs.0'].sort_values(by=['d3mIndex']).to_csv(results_path, index=False)
    else:
        logger.info('NOT storing fit results')

    with open(os.path.join(storage_dir, 'fitted_solution_%s.pkl' % pipeline_id), 'wb') as fout:
        pickle.dump(runtime, fout)
