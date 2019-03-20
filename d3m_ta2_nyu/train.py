import logging
import json
import tempfile
import os
import pickle
from sqlalchemy.orm import joinedload

from d3m.container import Dataset
from d3m.metadata import base as metadata_base
import d3m.metadata.base
import d3m.runtime
from d3m.metadata.problem import parse_problem_description
from d3m_ta2_nyu.workflow import database, convert


logger = logging.getLogger(__name__)


class CustomRuntime(d3m.runtime.Runtime):
    def __init__(self, targets, **kwargs):
        super(CustomRuntime, self).__init__(**kwargs)

        self.__targets = targets

    def _mark_columns(self, dataset):
        dataset = dataset.copy()

        # Set suggested target as attribute
        for resID, res in dataset.items():
            length = dataset.metadata.query([
                resID,
                d3m.metadata.base.ALL_ELEMENTS,
            ])['dimension']['length']
            for col_idx in range(length):
                col_selector = [
                    resID,
                    d3m.metadata.base.ALL_ELEMENTS,
                    col_idx,
                ]
                col_meta = dataset.metadata.query(col_selector)
                if ('https://metadata.datadrivendiscovery.org/types/'
                        'SuggestedTarget' in col_meta['semantic_types']):
                    dataset.metadata = dataset.metadata.add_semantic_type(
                        col_selector,
                        'https://metadata.datadrivendiscovery.org/types/'
                        'Attribute',
                    )

        # Mark targets
        for res_id, col_idx in self.__targets:
            dataset.metadata = dataset.metadata.add_semantic_type(
                [res_id, d3m.metadata.base.ALL_ELEMENTS, col_idx],
                'https://metadata.datadrivendiscovery.org/types/Target',
            )
            dataset.metadata = dataset.metadata.add_semantic_type(
                [res_id, d3m.metadata.base.ALL_ELEMENTS, col_idx],
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
            )
            dataset.metadata = dataset.metadata.remove_semantic_type(
                [res_id, d3m.metadata.base.ALL_ELEMENTS, col_idx],
                'https://metadata.datadrivendiscovery.org/types/Attribute',
            )

        return dataset


@database.with_db
def train(pipeline_id, dataset, problem, storage_dir, msg_queue, db):
    # Get pipeline from database
    pipeline = (
        db.query(database.Pipeline)
            .filter(database.Pipeline.id == pipeline_id)
            .options(joinedload(database.Pipeline.modules),
                     joinedload(database.Pipeline.connections))
    ).one()

    logger.info("About to train pipeline, id=%s, dataset=%r",
                pipeline_id, dataset)

    # Load data
    dataset = Dataset.load(dataset)
    logger.info("Loaded dataset")

    # Training step - fit pipeline on training data
    logger.info("Running training")

    d3m_pipeline = d3m.metadata.pipeline.Pipeline.from_json_structure(
        convert.to_d3m_json(pipeline),
    )

    # Convert problem description to core package format
    # FIXME: There isn't a way to parse from JSON data, so write it to a file
    # and read it back
    tmp = tempfile.NamedTemporaryFile('w', encoding='utf-8',
                                      suffix='.json', delete=False)
    try:
        try:
            json.dump(problem, tmp)
        finally:
            tmp.close()
        d3m_problem = parse_problem_description(tmp.name)
    finally:
        os.remove(tmp.name)

    runtime = d3m.runtime.Runtime(pipeline=d3m_pipeline, problem_description=d3m_problem,
                                  context=metadata_base.Context.TESTING)

    # Fitting pipeline on input dataset.
    fit_results = runtime.fit(inputs=[dataset])
    fit_results.check_success()

    with open(os.path.join(storage_dir, 'fitted_solution_%s.pkl' % pipeline_id), 'wb') as fout:
        pickle.dump(runtime, fout)
