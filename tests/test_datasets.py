import os
import csv
import json
import grpc
import logging
import subprocess
import pandas as pd
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
from datetime import datetime
from os.path import dirname, join
from d3m.metadata.pipeline import Pipeline
from d3m_ta2_nyu.grpc_logger import LoggingStub
from d3m_ta2_nyu.common import normalize_score
from ta3ta2_api.utils import encode_pipeline_description, ValueType, decode_value
from client import do_search, do_train, do_test, do_export


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


D3MINPUTDIR = os.environ.get('D3MINPUTDIR')
D3MOUTPUTDIR = os.environ.get('D3MOUTPUTDIR')
D3MSTATICDIR = os.environ.get('D3MSTATICDIR')


def search_pipelines(datasets, use_template=False):
    search_results_path = join(dirname(__file__), '../resource/search_results.json')
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

        with open(problem_path) as fin:
            problem = json.load(fin)

        task = get_task(problem)

        pipelines = do_search(core, problem, dataset_train_path, time_bound=10.0, pipelines_limit=0,
                              pipeline_template=pipeline_template)

        number_pipelines = len(pipelines)
        result = {'task': task, 'search_time': str(datetime.now() - start_time), 'pipelines': number_pipelines,
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

            result['pipelines'] = number_pipelines
            result['best_time'] = best_time
            result['best_score'] = all_scores[0]['score']
            result['all_scores'] = all_scores

        search_results[dataset] = result

        with open(search_results_path, 'w') as fout:
            json.dump(search_results, fout, indent=4)


def evaluate_pipelines(datasets, top=5):
    statistics_path = join(dirname(__file__), '../resource/statistics_datasets.csv')
    search_results_path = join(dirname(__file__), '../resource/search_results.json')
    search_results = load_search_results(search_results_path)
    size = len(datasets)

    for i, dataset in enumerate(datasets):
        logger.info('Processing dataset "%s" (%d/%d)' % (dataset, i+1, size))
        dataset_train_path = join(D3MINPUTDIR, dataset, 'TRAIN/dataset_TRAIN/datasetDoc.json')
        dataset_test_path = join(D3MINPUTDIR, dataset, 'TEST/dataset_TEST/datasetDoc.json')
        dataset_score_path = join(D3MINPUTDIR, dataset, 'SCORE/dataset_SCORE/datasetDoc.json')
        problem_path = join(D3MINPUTDIR, dataset, 'TRAIN/problem_TRAIN/problemDoc.json')

        if not os.path.isfile(dataset_score_path):
            dataset_score_path = join(D3MINPUTDIR, dataset, 'SCORE/dataset_TEST/datasetDoc.json')

        performance_top_pipelines = {}
        for top_pipeline in search_results[dataset]['all_scores'][:top]:
            top_pipeline_id = top_pipeline['id']
            logger.info('Scoring top pipeline id=%s' % top_pipeline_id)
            top_pipeline_path = join(D3MOUTPUTDIR, 'pipelines_searched', top_pipeline_id + '.json')
            score_pipeline_path = join(D3MOUTPUTDIR, 'predictions', top_pipeline_id + '_testscore.csv')

            command = [
                'python3', '-m', 'd3m', '--strict-resolving', '--strict-digest',
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
                '--scores', score_pipeline_path
                #'--output-run', os.path.join('/output/pipeline_runs/run.yaml')
                ]
            try:
                subprocess.call(command)
                df = pd.read_csv(score_pipeline_path)
                score = round(df['value'][0], 6)
                metric = df['metric'][0]
                new_score = normalize_score(metric, score, 'asc')
                performance_top_pipelines[top_pipeline_id] = (score, new_score)
                logger.info('Scored top pipeline id=%s, %s=%.6f' % (top_pipeline_id, metric, score))
            except Exception as e:
                logger.error('Error calculating test score')
                logger.error(e)

        best_pipeline = sorted(performance_top_pipelines.items(), key=lambda x: x[1][1], reverse=True)[0]
        best_id = best_pipeline[0]
        best_score = best_pipeline[1][0]
        logger.info('Best pipeline id=%s %s=%.6f' % (best_id, metric, best_score))

        row = [dataset, search_results[dataset]['pipelines'], search_results[dataset]['best_time'],
               search_results[dataset]['search_time'], best_score, metric, search_results[dataset]['task'], best_id]
        save_row(statistics_path, row)


def get_task(problem):
    task_type = problem['about']['taskType'].upper()
    if 'taskSubType' in problem['about']:
        task_type = task_type + '_' + problem['about']['taskSubType'].upper()

    return task_type


def save_row(file_path, row):
    with open(file_path, 'a') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(row)


def load_search_results(file_path):
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as fout:
            json.dump([], fout)

    with open(file_path) as fin:
        return json.load(fin)


def load_template():
    with open(os.path.join(os.path.dirname(__file__), '../resource/pipelines/example_metalearningdb.json')) as fin:
        json_pipeline = json.load(fin)

    d3m_pipeline = Pipeline.from_json_structure(json_pipeline, )
    grpc_pipeline = encode_pipeline_description(d3m_pipeline, [ValueType.RAW], '/tmp')

    return grpc_pipeline


if __name__ == '__main__':
    datasets = sorted([x for x in os.listdir(D3MINPUTDIR) if os.path.isdir(join(D3MINPUTDIR, x))])
    datasets = ['185_baseball']
    search_pipelines(datasets)
    evaluate_pipelines(datasets)
