import os
import csv
import json
import grpc
import logging
import subprocess
import pandas as pd
import d3m_automl_rpc.core_pb2_grpc as pb_core_grpc
from datetime import datetime
from os.path import join
from d3m.metadata.pipeline import Pipeline
from d3m_ta2_nyu.grpc_api.grpc_logger import LoggingStub
from d3m_automl_rpc.utils import encode_pipeline_description, decode_value
from d3m.metadata.problem import Problem, PerformanceMetric
from d3m.utils import yaml_load_all, fix_uri
from client import do_search, do_score, do_train, do_test, do_export, do_describe, do_save_fitted_solution

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


D3MINPUTDIR = os.environ.get('D3MINPUTDIR')
D3MOUTPUTDIR = os.environ.get('D3MOUTPUTDIR')
D3MSTATICDIR = os.environ.get('D3MSTATICDIR')


def search_pipelines(datasets, time_bound=10, use_template=False):
    search_results_path = join(D3MOUTPUTDIR, 'temp', 'search_results.json')
    search_results = load_search_results(search_results_path)
    channel = grpc.insecure_channel('localhost:45042')
    core = LoggingStub(pb_core_grpc.CoreStub(channel), logger)
    size = len(datasets)
    pipeline_template = None

    if use_template:
        pipeline_template = load_template()

    for i, dataset in enumerate(datasets):
        logger.info('Processing dataset "%s" (%d/%d)' % (dataset, i+1, size))
        start_time = datetime.now()

        dataset_train_path = join(D3MINPUTDIR, dataset, 'TRAIN/dataset_TRAIN/datasetDoc.json')
        problem_path = join(D3MINPUTDIR, dataset, 'TRAIN/problem_TRAIN/problemDoc.json')

        if not os.path.isfile(problem_path):
            logger.error('Problem file (%s) doesnt exist', problem_path)
            continue

        try:
            problem = Problem.load(problem_uri=fix_uri(problem_path))
        except:
            logger.exception('Error parsing problem')
            continue

        task_keywords = '_'.join([x.name for x in problem['problem']['task_keywords']])
        search_id, pipelines = do_search(core, problem, dataset_train_path, time_bound=time_bound, pipelines_limit=0,
                                        pipeline_template=pipeline_template)
        #print(dataset, problem['problem']['performance_metrics'][0]['metric'].name, task_keywords)
        number_pipelines = len(pipelines)
        result = {'search_id': search_id, 'task': task_keywords, 'search_time': str(datetime.now() - start_time), 'pipelines': number_pipelines,
                  'best_time': 'None', 'best_score': 'None', 'all_scores': []}

        if number_pipelines > 0:
            best_time = sorted(pipelines.values(), key=lambda x: x[2])[0][2]
            sorted_pipelines = sorted(pipelines.items(), key=lambda x: x[1][0], reverse=True)
            all_scores = []

            for pipeline_id, (_, pipeline, _) in sorted_pipelines:
                if use_template:  # FIXME: Pipeline score is not calculate when working with fully defined pipeline
                    pipeline_score = 1.0
                else:
                    pipeline_score = decode_value(pipeline[0].scores[0].value)['value']
                all_scores.append({'id': pipeline_id, 'score': pipeline_score})
                #do_score(core, problem, [pipeline_id], dataset_train_path)
                #fitted_pipeline = do_train(core, [pipeline_id], dataset_train_path)
                #do_save_fitted_solution(core, fitted_pipeline)
                #do_test(core, fitted_pipeline, dataset_train_path.replace('TRAIN', 'TEST'))
                #do_export(core, fitted_pipeline)
                #do_describe(core, [pipeline_id])

            result['pipelines'] = number_pipelines
            result['best_time'] = best_time
            result['best_score'] = all_scores[0]['score']
            result['all_scores'] = all_scores

        search_results[dataset] = result

        with open(search_results_path, 'w') as fout:
            json.dump(search_results, fout, indent=4)


def evaluate_pipelines(datasets, top=10):
    statistics_path = join(D3MOUTPUTDIR, 'temp', 'statistics_datasets.csv')
    search_results_path = join(D3MOUTPUTDIR, 'temp', 'search_results.json')
    search_results = load_search_results(search_results_path)
    size = len(datasets)

    for i, dataset in enumerate(datasets):
        if dataset not in search_results:
            continue

        logger.info('Processing dataset "%s" (%d/%d)' % (dataset, i+1, size))
        dataset_train_path = join(D3MINPUTDIR, dataset, 'TRAIN/dataset_TRAIN/datasetDoc.json')
        dataset_test_path = join(D3MINPUTDIR, dataset, 'TEST/dataset_TEST/datasetDoc.json')
        dataset_score_path = join(D3MINPUTDIR, dataset, 'SCORE/dataset_SCORE/datasetDoc.json')
        problem_path = join(D3MINPUTDIR, dataset, 'TRAIN/problem_TRAIN/problemDoc.json')
        performance_top_pipelines = {}
        best_score = metric = best_id = 'None'

        for top_pipeline in search_results[dataset]['all_scores'][:top]:
            top_pipeline_id = top_pipeline['id']
            search_id = search_results[dataset]['search_id']
            logger.info('Scoring top pipeline id=%s' % top_pipeline_id)
            top_pipeline_path = join(D3MOUTPUTDIR, search_id, 'pipelines_searched', '%s.json' % top_pipeline_id)
            #top_pipeline_path = '/usr/src/app/resource/pipelines/example_metalearningdb.json'
            score_pipeline_path = join(D3MOUTPUTDIR, 'temp', 'runtime_output', 'fit_score_%s.csv' % top_pipeline_id)

            command = [
                'python3', '-m', 'd3m',
                'runtime',
                '--volumes', D3MSTATICDIR,
                '--context', 'TESTING',
                '--random-seed', '0',
                'fit-score',
                '--pipeline', top_pipeline_path,
                '--problem', problem_path,
                '--input', dataset_train_path,
                '--test-input', dataset_test_path,
                '--score-input', dataset_score_path,
                '--scores', score_pipeline_path,
                '--output-run', join(D3MOUTPUTDIR, search_id, 'pipeline_runs', 'run_%s.yml' % top_pipeline_id)
                ]
            try:
                subprocess.call(command)
                df = pd.read_csv(score_pipeline_path)
                score = round(df['value'][0], 6)
                metric = df['metric'][0]
                normalized_score = PerformanceMetric[metric].normalize(score)
                performance_top_pipelines[top_pipeline_id] = (score, normalized_score)
                logger.info('Scored top pipeline id=%s, %s=%.6f' % (top_pipeline_id, metric, score))
            except:
                logger.exception('Error calculating test score')

        if len(performance_top_pipelines) > 0:
            best_pipeline = sorted(performance_top_pipelines.items(), key=lambda x: x[1][1], reverse=True)[0]
            best_id = best_pipeline[0]
            best_score = best_pipeline[1][0]
            logger.info('Best pipeline id=%s %s=%.6f' % (best_id, metric, best_score))

        row = [dataset, search_results[dataset]['pipelines'], search_results[dataset]['best_time'],
               search_results[dataset]['search_time'], best_score, metric, search_results[dataset]['task'], best_id]
        save_row(statistics_path, row)
        #create_dupms(search_id, performance_top_pipelines.keys())


def save_row(file_path, row):
    with open(file_path, 'a') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(row)


def load_search_results(file_path):
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as fout:
            json.dump({}, fout)

    with open(file_path) as fin:
        return json.load(fin)


def load_template():
    with open(join(os.path.dirname(__file__), '../resource/pipelines/example_metalearningdb.json')) as fin:
        json_pipeline = json.load(fin)

    d3m_pipeline = Pipeline.from_json_structure(json_pipeline, )
    grpc_pipeline = encode_pipeline_description(d3m_pipeline, ['RAW'], '/tmp')

    return grpc_pipeline


def create_dupms(search_id, top_pipelines):
    pipelines_list = []
    pipeline_runs_list = []

    for top_pipeline in top_pipelines:
        with open(join(D3MOUTPUTDIR, search_id, 'pipeline_runs', 'run_%s.yml' % top_pipeline)) as fin:
            for pipeline_run in yaml_load_all(fin):
                digest = pipeline_run['pipeline']['digest']
                pipeline_runs_list.append(json.dumps(pipeline_run))

        with open(join(D3MOUTPUTDIR, search_id, 'pipelines_searched', '%s.json' % top_pipeline)) as fin:
            pipeline = json.load(fin)
            pipeline['digest'] = digest
            pipeline['source'] = {'name': 'new_nyu'}
            pipelines_list.append(json.dumps(pipeline))

    with open(join(D3MOUTPUTDIR, 'temp', 'pipelines.json'), 'w') as fout:
        fout.write('\n'.join(pipelines_list))

    with open(join(D3MOUTPUTDIR, 'temp', 'pipeline_runs.json'), 'w') as fout:
        fout.write('\n'.join(pipeline_runs_list))


def create_inputs_pofiler(dataset):
    search_results_path = join(D3MOUTPUTDIR, 'temp', 'search_results.json')
    search_results = load_search_results(search_results_path)
    search_id = search_results[dataset]['search_id']
    pipelines = search_results[dataset]['all_scores']
    profiler_inputs = []

    for pipeline_info in pipelines:
        with open(join(D3MOUTPUTDIR, search_id, 'pipelines_searched/%s.json' % pipeline_info['id'])) as fin:
            json_pipeline = json.load(fin)

        score = float(pipeline_info['score'])
        problem = dataset
        start_time = datetime.utcnow().isoformat() + 'Z'
        metric = 'f1'
        end_time = datetime.utcnow().isoformat() + 'Z'
        normalized_score = PerformanceMetric[metric.upper()].normalize(score)
        pipeline_score = [{'metric': {'metric': metric}, 'value': score,
                           'normalized': normalized_score}]
        if 'digest' not in json_pipeline:
            json_pipeline['digest'] = json_pipeline['id']

        profiler_data = {
            'pipeline_id': json_pipeline['id'],
            'inputs': json_pipeline['inputs'],
            'steps': json_pipeline['steps'],
            'outputs': json_pipeline['outputs'],
            'pipeline_digest': json_pipeline['digest'],
            'problem': problem,
            'start': start_time,
            'end': end_time,
            'scores': pipeline_score,
            'pipeline_source': {'name': 'NYU'},
        }
        profiler_inputs.append(profiler_data)

        with open(join(D3MOUTPUTDIR, 'pipeline_profiler.json'), 'w') as fout:
            json.dump(profiler_inputs, fout)


if __name__ == '__main__':
    datasets = sorted([x for x in os.listdir(D3MINPUTDIR) if os.path.isdir(join(D3MINPUTDIR, x))])
    datasets = ['185_baseball_MIN_METADATA']
    search_pipelines(datasets, 5)
    evaluate_pipelines(datasets)

