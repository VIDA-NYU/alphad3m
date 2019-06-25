import logging
import numpy
import json
import os
import pkg_resources
import tempfile
import d3m.metadata.base
import d3m.runtime
import d3m_ta2_nyu.proto.core_pb2 as pb_core
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from d3m_ta2_nyu.workflow import database, convert
from d3m_ta2_nyu.common import normalize_score, SCORES_FROM_SCHEMA
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import Problem

logger = logging.getLogger(__name__)

SAMPLE_SIZE = 100
RANDOM_SEED = 65682867

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


@database.with_db
def score(pipeline_id, dataset_uri, metrics, problem, scoring_conf, do_rank, msg_queue, db):
    if scoring_conf is None:
        scoring_conf = {'train_test_ratio': '0.75',
                        'shuffle': 'true',
                        'stratified': 'false'
        }
        evaluation_method = None
    else:
        evaluation_method = scoring_conf['method']

    # Get pipeline from database
    pipeline = (
        db.query(database.Pipeline)
            .filter(database.Pipeline.id == pipeline_id)
            .options(joinedload(database.Pipeline.modules),
                     joinedload(database.Pipeline.connections))
    ).one()

    logger.info('About to score pipeline, id=%s, metrics=%s, dataset=%r', pipeline_id, metrics, dataset_uri)

    # Load data
    dataset = Dataset.load(dataset_uri)
    logger.info('Loaded dataset')

    if evaluation_method is None:  # It comes from search_solutions, so do the sample and use HOLDOUT
        for res_id in dataset:
            if ('https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint'
                    in dataset.metadata.query([res_id])['semantic_types']):
                break
        else:
            res_id = next(iter(dataset))
        if hasattr(dataset[res_id], 'columns') and len(dataset[res_id]) > SAMPLE_SIZE:
            # Sample the dataset to stay reasonably fast
            logger.info('Sampling down data from %d to %d', len(dataset[res_id]), SAMPLE_SIZE)
            sample = numpy.concatenate([numpy.repeat(True, SAMPLE_SIZE),
                                        numpy.repeat(False, len(dataset[res_id]) - SAMPLE_SIZE)])

            numpy.random.RandomState(seed=RANDOM_SEED).shuffle(sample)
            dataset[res_id] = dataset[res_id][sample]

        evaluation_method = pb_core.EvaluationMethod.Value('HOLDOUT')

    if evaluation_method == pb_core.EvaluationMethod.Value('K_FOLD'):
        scores = cross_validation(pipeline, dataset, metrics, problem, scoring_conf)

        # Store scores
        cross_validation_scores = []
        for fold, fold_scores in scores.items():
            for metric, value in fold_scores.items():
                cross_validation_scores.append(database.CrossValidationScore(fold=fold, metric=metric, value=value))

        crossval_db = database.CrossValidation(pipeline_id=pipeline_id, scores=cross_validation_scores)
        db.add(crossval_db)

    elif evaluation_method == pb_core.EvaluationMethod.Value('HOLDOUT'):
        scores = holdout(pipeline, dataset, metrics, problem, scoring_conf)

        # Store scores
        holdout_scores = []
        for fold, fold_scores in scores.items():
            for metric, value in fold_scores.items():
                holdout_scores.append(database.CrossValidationScore(fold=fold, metric=metric, value=value))

        # TODO Create Holdout table in database
        if not do_rank:
            holdout_db = database.CrossValidation(pipeline_id=pipeline_id, scores=holdout_scores)
            db.add(holdout_db)

    elif evaluation_method == pb_core.EvaluationMethod.Value('RANKING'):
        metrics = []
        for metric in problem['inputs']['performanceMetrics']:
            metric_name = metric['metric']
            try:
                metric_name = SCORES_FROM_SCHEMA[metric_name]
            except KeyError:
                logger.error("Unknown metric %r", metric_name)
                raise ValueError("Unknown metric %r" % metric_name)

            formatted_metric = {'metric': metric_name}

            if len(metric) > 1:  # Metric has parameters
                formatted_metric['params'] = {}
                for param in metric.keys():
                    if param != 'metric':
                        formatted_metric['params'][param] = metric[param]

            metrics.append(formatted_metric)

        scores = holdout(pipeline, dataset, metrics, problem, {'train_test_ratio': '0.75', 'shuffle': 'true'})

        # Store scores
        holdout_scores = []
        for fold, fold_scores in scores.items():
            for metric, current_score in fold_scores.items():
                new_score = normalize_score(metric, current_score, 'desc')
                new_metric = 'RANK'
                holdout_scores.append(database.CrossValidationScore(fold=fold, metric=new_metric, value=new_score))

        # TODO Should results be stored in CrossValidation table?
        holdout_db = database.CrossValidation(pipeline_id=pipeline_id, scores=holdout_scores)
        db.add(holdout_db)

    if do_rank and len(scores) > 0:  # Need to rank too
        logger.info("Calculating RANK in search solution for pipeline %s", pipeline_id)
        dataset = Dataset.load(dataset_uri)  # load complete dataset
        scores = holdout(pipeline, dataset, metrics, problem, {'train_test_ratio': '0.75', 'shuffle': 'true'})
        #holdout_scores = []
        for fold, fold_scores in scores.items():
            for metric, current_score in fold_scores.items():
                new_score = normalize_score(metric, current_score, 'desc')
                new_metric = 'RANK'
                logger.info('Normalizing values: fold=%s, metric=%s, score=%s', fold, new_metric, new_score)
                holdout_scores.append(database.CrossValidationScore(fold=fold, metric=new_metric, value=new_score))

        # TODO Should results be stored in CrossValidation table?
        holdout_db = database.CrossValidation(pipeline_id=pipeline_id, scores=holdout_scores)
        db.add(holdout_db)

    db.commit()


def cross_validation(pipeline, dataset, metrics, problem, scoring_conf):
    data_pipeline = kfold_tabular_split
    scoring_conf['number_of_folds'] = scoring_conf.pop('folds')
    scores = evaluate(pipeline, data_pipeline, dataset, metrics, problem, scoring_conf)
    logger.info("Cross-validation results:\n%s", scores)
    results = {}

    for _, row in scores.iterrows():
        if row['fold'] not in results:
            results[row['fold']] = {}
        results[row['fold']][row['metric']] = row['value']

    return results


def holdout(pipeline, dataset, metrics, problem, scoring_conf):
    data_pipeline = train_test_tabular_split
    scores = evaluate(pipeline, data_pipeline, dataset, metrics, problem, scoring_conf)
    logger.info("Holdout results:\n%s", scores)
    results = {}

    for _, row in scores.iterrows():
        if row['fold'] not in results:
            results[row['fold']] = {}
        results[row['fold']][row['metric']] = row['value']

    return results


def evaluate(pipeline, data_pipeline, dataset, metrics, problem, scoring_conf):
    json_pipeline = convert.to_d3m_json(pipeline)
    #with open(os.path.join(os.path.dirname(__file__),'temporal.json')) as fin:
    #    json_pipeline = json.load(fin)
    logger.info("Pipeline to be scored:\n%s",
                '\n'.join([x['primitive']['python_path'] for x in json_pipeline['steps']]))

    d3m_pipeline = d3m.metadata.pipeline.Pipeline.from_json_structure(json_pipeline, )

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