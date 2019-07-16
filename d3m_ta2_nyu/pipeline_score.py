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
from d3m_ta2_nyu.common import normalize_score, format_metrics
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
def score(pipeline_id, dataset_uri, metrics, problem, scoring_conf, do_rank, do_sample, msg_queue, db):
    #do_sample = False
    if scoring_conf is None:
        scoring_conf = {'shuffle': 'true',
                        'stratified': 'false',
                        'train_test_ratio': '0.75',
                        'method': pb_core.EvaluationMethod.Value('HOLDOUT')}

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

    if do_sample:
        dataset = get_sample(dataset)

    evaluation_method = scoring_conf['method']
    scores_db = []
    scores = {}

    if evaluation_method == pb_core.EvaluationMethod.Value('K_FOLD'):
        scoring_conf['number_of_folds'] = scoring_conf.pop('folds')
        scores = evaluate(pipeline, kfold_tabular_split, dataset, metrics, problem, scoring_conf)
        logger.info("Cross-validation results:\n%s", scores)

    elif evaluation_method == pb_core.EvaluationMethod.Value('HOLDOUT'):
        scores = evaluate(pipeline, train_test_tabular_split, dataset, metrics, problem, scoring_conf)
        logger.info("Holdout results:\n%s", scores)

    elif evaluation_method == pb_core.EvaluationMethod.Value('RANKING'):
        metrics = format_metrics(problem)
        scores = evaluate(pipeline, train_test_tabular_split, dataset, metrics, problem, scoring_conf)
        scores = create_new_metric(scores)
        logger.info("Ranking-D3M results:\n%s", scores)

    scores_db = add_scores_db(scores, scores_db)

    if do_rank and len(scores) > 0:  # Need to rank too
        logger.info("Calculating RANK in search solution for pipeline %s", pipeline_id)
        dataset = Dataset.load(dataset_uri)  # load complete dataset
        scores = evaluate(pipeline, train_test_tabular_split, dataset, metrics, problem, scoring_conf)
        logger.info("Ranking-D3M (whole dataset) results:\n%s", scores)
        scores = create_new_metric(scores)
        logger.info("Ranking-D3M (whole dataset) new metric results:\n%s", scores)
        scores_db = add_scores_db(scores, scores_db)

    # TODO Should we rename CrossValidation table?
    record_db = database.CrossValidation(pipeline_id=pipeline_id, scores=scores_db)  # Store scores
    db.add(record_db)
    db.commit()


def evaluate(pipeline, data_pipeline, dataset, metrics, problem, scoring_conf):
    json_pipeline = convert.to_d3m_json(pipeline)
    #with open(os.path.join(os.path.dirname(__file__), 'pipeline_tmp.json')) as fin:
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

    formatted_metric = format_d3m_metrics(metrics)

    run_scores, run_results = d3m.runtime.evaluate(
        pipeline=d3m_pipeline,
        data_pipeline=data_pipeline,
        scoring_pipeline=scoring,
        problem_description=d3m_problem,
        inputs=[dataset],
        data_params=scoring_conf,
        metrics=formatted_metric,
        volumes_dir=os.environ.get('D3MSTATICDIR', None),
        context=d3m.metadata.base.Context.TESTING,
        random_seed=0,
    )

    run_results.check_success()
    #save_pipeline_runs(run_results.pipelines_runs)  # TODO: It should work, but has some bugs
    combined_folds = d3m.runtime.combine_folds([fold for fold in run_scores])
    scores = {}

    for _, row in combined_folds.iterrows():
        if row['fold'] not in scores:
            scores[row['fold']] = {}
        scores[row['fold']][row['metric']] = row['value']

    return scores


def get_sample(dataset):
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

    return dataset


def create_new_metric(scores):
    scores_tmp = {}

    for fold, fold_scores in scores.items():
        scores_tmp[fold] = {}
        for metric, current_score in fold_scores.items():
            new_score = normalize_score(metric, current_score, 'desc')
            new_metric = 'RANK'
            scores_tmp[fold][new_metric] = new_score

    return scores_tmp


def add_scores_db(scores_dict, scores_db):
    for fold, fold_scores in scores_dict.items():
        for metric, value in fold_scores.items():
            scores_db.append(database.CrossValidationScore(fold=fold, metric=metric, value=value))

    return scores_db


def format_d3m_metrics(metrics):
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


def save_pipeline_runs(pipelines_runs):
    for pipeline_run in pipelines_runs:
        save_run_path = os.path.join(os.environ['D3MOUTPUTDIR'], 'pipeline_runs',
                                     pipeline_run.to_json_structure()['id'] + '.yml')
        with open(save_run_path, 'w') as fin:
            pipeline_run.to_yaml(fin, indent=2)
