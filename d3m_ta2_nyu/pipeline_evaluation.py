import json
import logging
import os
import pkg_resources
import tempfile
import d3m.metadata.base
import d3m.runtime
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import Problem
from d3m_ta2_nyu.workflow import convert


logger = logging.getLogger(__name__)
primitives_blacklist = []


with pkg_resources.resource_stream(
        'd3m_ta2_nyu',
        '../resource/pipelines/kfold_tabular_split.yaml') as fp:
    kfold_tabular_split = Pipeline.from_yaml(fp)

with pkg_resources.resource_stream(
        'd3m_ta2_nyu',
        '../resource/pipelines/train-test-tabular-split.yaml') as fp:
    train_test_tabular_split = Pipeline.from_yaml(fp)

with pkg_resources.resource_stream(
        'd3m_ta2_nyu',
        '../resource/pipelines/scoring.yaml') as fp:
    scoring = Pipeline.from_yaml(fp)


def evaluate(pipeline, data_pipeline, dataset, metrics, problem, scoring_conf, resolver):
    json_pipeline = convert.to_d3m_json(pipeline)
    logger.info("Pipeline to be scored:\n%s",
                '\n'.join([x['primitive']['python_path'] for x in json_pipeline['steps']]))

    d3m_pipeline = resolver.get_pipeline(json_pipeline, )

    # Convert problem description to core package format
    # FIXME: There isn't a way to parse from JSON data, so write it to a file
    # and read it back
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, 'problemDoc.json')
        with open(tmp_path, 'w', encoding='utf8') as fin:
            json.dump(problem, fin)
        d3m_problem = Problem.load('file://' + tmp_path)

    formatted_metric = _format_metrics(metrics)

    results = d3m.runtime.evaluate(
        pipeline=d3m_pipeline,
        data_pipeline=data_pipeline,
        scoring_pipeline=scoring,
        problem_description=d3m_problem,
        inputs=[dataset],
        data_params=scoring_conf,
        metrics=formatted_metric,
        volumes_dir=os.environ.get('D3MSTATICDIR', None),
        context=d3m.metadata.base.Context.TESTING,
        scratch_dir=os.path.join(os.environ['D3MOUTPUTDIR'], 'pipeline_runs'),
        random_seed=0,
    )

    results[1][0].check_success()
    scores = d3m.runtime.combine_folds([fold for fold in results[0]])

    return scores


def cross_validation(pipeline, dataset, metrics, problem, scoring_conf, resolver):
    data_pipeline = kfold_tabular_split
    scoring_conf['number_of_folds'] = scoring_conf.pop('folds')
    scores = evaluate(pipeline, data_pipeline, dataset, metrics, problem, scoring_conf, resolver)
    logger.info("Cross-validation results:\n%s", scores)
    results = {}

    for _, row in scores.iterrows():
        if row['fold'] not in results:
            results[row['fold']] = {}
        results[row['fold']][row['metric']] = row['value']

    return results


def holdout(pipeline, dataset, metrics, problem, scoring_conf, resolver):
    data_pipeline = train_test_tabular_split
    scores = evaluate(pipeline, data_pipeline, dataset, metrics, problem, scoring_conf, resolver)
    logger.info("Holdout results:\n%s", scores)
    results = {}

    for _, row in scores.iterrows():
        if row['fold'] not in results:
            results[row['fold']] = {}
        results[row['fold']][row['metric']] = row['value']

    return results


def _format_metrics(metrics):
    formatted_metrics = []

    for metric in metrics:
        formatted_metric = {'metric': d3m.metadata.problem.PerformanceMetric[metric['metric']]}
        if 'params' in metric:
            formatted_metric['params'] = {}
            if 'posLabel' in metric['params']:
                formatted_metric['params']['pos_label'] = metric['params']['posLabel']
            if 'K' in metric['params']:
                formatted_metric['params']['k'] = metric['params']['K']

        formatted_metrics.append(formatted_metric)

    return formatted_metrics
