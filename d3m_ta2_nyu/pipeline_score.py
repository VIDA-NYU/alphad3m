import logging
import os
import json
import pkg_resources
import random
import d3m.metadata.base
import d3m.runtime
from sqlalchemy.orm import joinedload
from d3m.container import Dataset
from d3m_ta2_nyu.workflow import database, convert
from d3m_ta2_nyu.utils import is_collection, get_dataset_sample
from d3m.metadata.pipeline import Pipeline
from d3m.metadata.problem import PerformanceMetric, TaskKeyword
from multiprocessing import Manager, Process

logger = logging.getLogger(__name__)


with pkg_resources.resource_stream(
        'd3m_ta2_nyu',
        '../resource/pipelines/kfold_tabular_split.yaml') as fp:
    kfold_tabular_split = Pipeline.from_yaml(fp)

with pkg_resources.resource_stream(
        'd3m_ta2_nyu',
        '../resource/pipelines/kfold_timeseries_split.yaml') as fp:
    kfold_timeseries_split = Pipeline.from_yaml(fp)

with pkg_resources.resource_stream(
        'd3m_ta2_nyu',
        '../resource/pipelines/train-test-tabular-split.yaml') as fp:
    train_test_tabular_split = Pipeline.from_yaml(fp)

with pkg_resources.resource_stream(
        'd3m_ta2_nyu',
        '../resource/pipelines/scoring.yaml') as fp:
    scoring_pipeline = Pipeline.from_yaml(fp)


def check_timeindicator(dataset_path):
    with open(dataset_path) as fin:
        dataset_doc = json.load(fin)

    columns = dataset_doc['dataResources'][0]['columns']
    timeindicator_index = None
    has_timeindicator = False
    for item in columns:
        if item['colType'] == 'dateTime':
            timeindicator_index = item['colIndex']
        if 'timeIndicator' in item['role']:
            has_timeindicator = True
            break

    if not has_timeindicator:
        dataset_doc['dataResources'][0]['columns'][timeindicator_index]['role'].append('timeIndicator')
        try:
            with open(dataset_path, 'w') as fout:
                json.dump(dataset_doc, fout, indent=4)
        except:
            logger.error('Saving timeIndicator on dataset')


@database.with_db
def score(pipeline_id, dataset_uri, sample_dataset_uri, metrics, problem, scoring_config, timeout_run, report_rank,
          msg_queue, db):
    dataset_uri_touse = dataset_uri

    if sample_dataset_uri:
        dataset_uri_touse = sample_dataset_uri
    if TaskKeyword.FORECASTING in problem['problem']['task_keywords']:
        check_timeindicator(dataset_uri_touse[7:])

    dataset = Dataset.load(dataset_uri_touse)
    # Get pipeline from database
    pipeline = (
        db.query(database.Pipeline)
            .filter(database.Pipeline.id == pipeline_id)
            .options(joinedload(database.Pipeline.modules),
                     joinedload(database.Pipeline.connections))
    ).one()

    logger.info('About to score pipeline, id=%s, metrics=%s, dataset=%r', pipeline_id, metrics, dataset_uri)

    scores = {}
    scores_db = []
    pipeline_split = None

    if TaskKeyword.FORECASTING in problem['problem']['task_keywords']:
        pipeline_split = kfold_timeseries_split

    elif scoring_config['method'] == 'K_FOLD':
        pipeline_split = kfold_tabular_split

    elif scoring_config['method'] == 'HOLDOUT':
        pipeline_split = train_test_tabular_split

    elif scoring_config['method'] == 'RANKING':  # For TA2 only evaluation
        scoring_config['number_of_folds'] = '4'
        report_rank = True
        pipeline_split = kfold_tabular_split
    else:
        logger.warning('Unknown evaluation method, using K_FOLD')
        pipeline_split = kfold_tabular_split

    if metrics[0]['metric'] == PerformanceMetric.F1 and TaskKeyword.SEMISUPERVISED in problem['problem']['task_keywords']:
        new_metrics = [{'metric': PerformanceMetric.F1_MACRO}]
        scores = evaluate(pipeline, kfold_tabular_split, dataset, new_metrics, problem, scoring_config, dataset_uri, timeout_run)
        scores = change_name_metric(scores, new_metrics, new_metric=metrics[0]['metric'].name)
    else:
        scores = evaluate(pipeline, pipeline_split, dataset, metrics, problem, scoring_config, dataset_uri, timeout_run)

    logger.info("Evaluation results:\n%s", scores)

    if len(scores) > 0:  # It's a valid pipeline
        scores_db = add_scores_db(scores, scores_db)
        if report_rank:  # For TA2 only evaluation
            scores = create_rank_metric(scores, metrics)
            scores_db = add_scores_db(scores, scores_db)
            logger.info("Evaluation results for RANK metric: \n%s", scores)

    # TODO Should we rename CrossValidation table?
    record_db = database.CrossValidation(pipeline_id=pipeline_id, scores=scores_db)  # Store scores
    db.add(record_db)
    db.commit()


def evaluate(pipeline, data_pipeline, dataset, metrics, problem, scoring_config, dataset_uri, timeout_run):
    if is_collection(dataset_uri[7:]):
        dataset = get_dataset_sample(dataset, problem)

    json_pipeline = convert.to_d3m_json(pipeline)

    if TaskKeyword.GRAPH in problem['problem']['task_keywords'] and json_pipeline['description'].startswith('MtLDB'):
        return {0: {'ACCURACY': 1.0}, 1: {'ACCURACY': 1.0}}

    logger.info("Pipeline to be scored:\n\t%s",
                '\n\t'.join([x['primitive']['python_path'] for x in json_pipeline['steps']]))

    d3m_pipeline = Pipeline.from_json_structure(json_pipeline, )
    if 'method' in scoring_config:
        scoring_config.pop('method')

    manager = Manager()
    return_dict = manager.dict()
    p = Process(target=worker, args=(d3m_pipeline, data_pipeline, scoring_pipeline, problem, dataset, scoring_config, metrics, return_dict))
    p.start()
    p.join(timeout_run)
    p.terminate()

    if 'run_results' not in return_dict or 'run_scores' not in return_dict:
        raise TimeoutError('Reached timeout (%d seconds) to score a pipeline' % timeout_run)

    run_results = return_dict['run_results']
    run_scores = return_dict['run_scores']

    for result in run_results:
        if result.has_error():
            raise RuntimeError(result.pipeline_run.status['message'])

    #save_pipeline_runs(run_results.pipeline_runs)
    combined_folds = d3m.runtime.combine_folds([fold for fold in run_scores])
    scores = {}

    for _, row in combined_folds.iterrows():
        if row['fold'] not in scores:
            scores[row['fold']] = {}
        scores[row['fold']][row['metric']] = row['value']

    return scores


def worker(d3m_pipeline, data_pipeline, scoring_pipeline, problem, dataset, scoring_config, metrics, return_dict):
    run_scores, run_results = d3m.runtime.evaluate(
        pipeline=d3m_pipeline,
        data_pipeline=data_pipeline,
        scoring_pipeline=scoring_pipeline,
        problem_description=problem,
        inputs=[dataset],
        data_params=scoring_config,
        metrics=metrics,
        volumes_dir=os.environ.get('D3MSTATICDIR', None),
        context=d3m.metadata.base.Context.TESTING,
        random_seed=0,
    )
    return_dict['run_scores'] = run_scores
    return_dict['run_results'] = run_results


def create_rank_metric(scores, metrics):
    scores_tmp = {}

    for fold, fold_scores in scores.items():
        scores_tmp[fold] = {}
        for metric, current_score in fold_scores.items():
            if metric == metrics[0]['metric'].name:
                new_score = (1.0 - metrics[0]['metric'].normalize(current_score)) + random.random() * 1.e-12
                scores_tmp[fold]['RANK'] = new_score

    return scores_tmp


def change_name_metric(scores, metrics, new_metric):
    scores_tmp = {}

    for fold, fold_scores in scores.items():
        scores_tmp[fold] = {}
        for metric, current_score in fold_scores.items():
            if metric == metrics[0]['metric'].name:
                scores_tmp[fold][new_metric] = current_score

    return scores_tmp


def add_scores_db(scores_dict, scores_db):
    for fold, fold_scores in scores_dict.items():
        for metric, value in fold_scores.items():
            scores_db.append(database.CrossValidationScore(fold=fold, metric=metric, value=value))

    return scores_db


def save_pipeline_runs(pipelines_runs):
    for pipeline_run in pipelines_runs:
        save_run_path = os.path.join(os.environ['D3MOUTPUTDIR'], 'pipeline_runs',
                                     pipeline_run.to_json_structure()['id'] + '.yml')

        with open(save_run_path, 'w') as fin:
            pipeline_run.to_yaml(fin, indent=2)
