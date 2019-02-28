import json
import logging
import os
import pkg_resources
import tempfile

import d3m.metadata.base
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import parse_problem_description
import d3m.runtime

from d3m_ta2_nyu.workflow import convert


logger = logging.getLogger(__name__)


with pkg_resources.resource_stream(
        'd3m_ta2_nyu',
        'pipelines/kfold_tabular_split.yaml') as fp:
    kfold_tabular_split = Pipeline.from_yaml(fp)

with pkg_resources.resource_stream(
        'd3m_ta2_nyu',
        'pipelines/scoring.yaml') as fp:
    scoring = Pipeline.from_yaml(fp)


def cross_validation(pipeline, dataset, metrics, problem,
                     folds, stratified=None, shuffle=None):
    if stratified is not None or shuffle is not None:
        # TODO: Data preparation arguments
        raise NotImplementedError

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

    results = d3m.runtime.evaluate(
        pipeline=d3m_pipeline,
        data_pipeline=kfold_tabular_split,
        scoring_pipeline=scoring,
        problem_description=d3m_problem,
        inputs=[dataset],
        data_params={'number_of_folds': str(folds)},
        metrics=[
            {'metric': d3m.metadata.problem.PerformanceMetric[metric]}
            for metric in metrics
        ],
        volumes_dir=os.environ.get('D3M_PRIMITIVE_STATIC', None),
        context=d3m.metadata.base.Context.TESTING,
        random_seed=0,
    )

    scores = d3m.runtime.combine_folds([fold[0] for fold in results])
    logger.info("Cross-validation results:\n%s", scores)
    return {s['fold']: {s['metric']: s['value']}
            for _, s in scores.iterrows()}
