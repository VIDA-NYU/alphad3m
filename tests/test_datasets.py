import os
import csv
import json
import grpc
import logging
import pandas as pd
import d3m_ta2_nyu.proto.core_pb2_grpc as pb_core_grpc
from datetime import datetime
from os.path import dirname, join
from d3m.metadata.pipeline import Pipeline
from d3m_ta2_nyu.grpc_logger import LoggingStub
from ta3ta2_api.utils import encode_pipeline_description, ValueType
from client import do_search, do_train, do_test, do_export
import subprocess

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


D3MINPUTDIR = os.environ.get('D3MINPUTDIR')
D3MOUTPUTDIR = os.environ.get('D3MOUTPUTDIR')


def run_all_datasets():
    channel = grpc.insecure_channel('localhost:45042')
    core = LoggingStub(pb_core_grpc.CoreStub(channel), logger)
    statistics_path = join(dirname(__file__), '../resource/statistics_datasets.csv')
    datasets = sorted([x for x in os.listdir(D3MINPUTDIR) if os.path.isdir(join(D3MINPUTDIR, x))])
    datasets = ['185_baseball']
    size = len(datasets)
    use_template = True
    pipeline_template = None

    if use_template:
        pipeline_template = load_template()

    for i, dataset in enumerate(datasets):
        logger.info('Processing dataset "%s" (%d/%d)' % (dataset, i+1, size))
        start_time = datetime.now()

        dataset_train_path = join(D3MINPUTDIR, dataset, 'TRAIN/dataset_TRAIN/datasetDoc.json')
        dataset_test_path = join(D3MINPUTDIR, dataset, 'TEST/dataset_TEST/datasetDoc.json')
        dataset_score_path = join(D3MINPUTDIR, dataset, 'SCORE/dataset_TEST/datasetDoc.json')
        problem_path = join(D3MINPUTDIR, dataset, 'TRAIN/problem_TRAIN/problemDoc.json')

        if not os.path.isfile(problem_path):
            logger.error('Problem file (%s) doesnt exist', problem_path)
            continue

        with open(problem_path) as fin:
            problem = json.load(fin)

        task = get_task(problem)
        best_time, score, metric = 'None', 'None', 'None'
        pipelines = do_search(core, problem, dataset_train_path, time_bound=1.0, pipelines_limit=0,
                              pipeline_template=pipeline_template)
        search_time = str(datetime.now() - start_time)
        number_pipelines = len(pipelines)

        if number_pipelines > 0:
            best_time = sorted(pipelines.values(), key=lambda x: x[2])[0][2]
            best_pipeline = sorted(pipelines.items(), key=lambda x: x[1][0])[-1][0]
            logger.info('Best pipeline: solution_id=%s' % best_pipeline)
            best_pipeline_path = join(D3MOUTPUTDIR, 'pipelines_searched', best_pipeline + '.json')
            score_pipeline_path = join(D3MOUTPUTDIR, 'predictions', best_pipeline + '_testscore.csv')

            command = [
                'python3', '-m', 'd3m', '--strict-resolving', '--strict-digest',
                'runtime',
                '--volumes', '"${D3MSTATICDIR}"',
                '--context', 'EVALUATION',
                '--random-seed', '0',
                'fit-score',
                '--pipeline', best_pipeline_path,
                '--problem', problem_path,
                '--input', dataset_train_path,
                '--test-input', dataset_test_path,
                '--score-input', dataset_score_path,
                '--scores', score_pipeline_path]
            try:
                subprocess.call(command)
                df = pd.read_csv(score_pipeline_path)
                score = df['value'][0]
                metric = df['metric'][0]
                logger.info('Best pipeline scored: %s=%.6f' % (metric, score))
            except Exception as e:
                logger.error('Error calculating test score')
                logger.error(e)

        row = [dataset, number_pipelines, best_time, search_time, score, metric, task]
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


def load_template():
    with open(os.path.join(os.path.dirname(__file__), '../resource/pipelines/temporal.json')) as fin:
        json_pipeline = json.load(fin)

    d3m_pipeline = Pipeline.from_json_structure(json_pipeline, )
    grpc_pipeline = encode_pipeline_description(d3m_pipeline, [ValueType.RAW], '/tmp')

    return grpc_pipeline


if __name__ == '__main__':
    run_all_datasets()


