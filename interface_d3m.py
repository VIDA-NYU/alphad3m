import json
import time
import logging
import subprocess
import pandas as pd
from os.path import join, split
from d3m.utils import yaml_load_all
from client import Cliente

logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


class AlphaAutoml:

    def __init__(self, output_folder):
        self.client = Cliente()
        self.output_folder = output_folder
        self.pipelines = []
        self.leaderboard = pd.DataFrame()

    def search_pipelines(self, dataset_path, time_bound=1):
        #start_ta2(dataset_path, self.output_folder)
        time.sleep(5)  # Wait for ta2 start
        dataset_train_path = '/input/dataset/TRAIN/dataset_TRAIN/datasetDoc.json'
        problem_path = join(dataset_path, 'problem_TRAIN/problemDoc.json')

        pipelines = self.client.do_search(dataset_train_path, problem_path, time_bound, pipelines_limit=0)

        for pipeline in pipelines:
            print('Found pipeline, id=%s %s=%s time=%s' % (pipeline['id'],  pipeline['metric'], pipeline['score'],
                                                           pipeline['time']))
            self.pipelines.append(pipeline)

        if len(self.pipelines) > 0:
            leaderboard = []
            sorted_pipelines = sorted(self.pipelines, key=lambda x: x['normalized_score'], reverse=True)
            metric = sorted_pipelines[0]['metric']
            for position, pipeline_data in enumerate(sorted_pipelines, 1):
                leaderboard.append([position, pipeline_data['id'], pipeline_data['score']])

            self.leaderboard = pd.DataFrame(leaderboard, columns=['ranking', 'id', metric]).style.hide_index()

        return self.pipelines

    def train(self, solution_id):
        dataset_train_path = '/input/dataset/TRAIN/dataset_TRAIN/datasetDoc.json'
        solution_ids = {p['id'] for p in self.pipelines}

        if solution_id not in solution_ids:
            print('Pipeline id=%s does not exist' % solution_id)
            return

        print('Training model...')
        fitted_solution_id = self.client.do_train(solution_id, dataset_train_path)
        fitted_solution = None  # TODO: Call to LoadFittedSolution
        model = {fitted_solution_id: fitted_solution}
        print('Training finished!')

        return model

    def test(self, model, test_dataset_path):
        dataset_test_path = '/input/dataset/TEST/dataset_TEST/datasetDoc.json'
        fitted_solution_id = list(model.keys())[0]
        print('Testing model...')
        tested_solution_path = self.client.do_test(fitted_solution_id, dataset_test_path)
        print('Testing finished!')
        tested_solution_path = tested_solution_path.replace('file:///output/', '')
        predictions = pd.read_csv(join(self.output_folder, tested_solution_path))

        return predictions


def start_ta2(dataset_path, output_path):
    dataset_path = split(dataset_path)[0]
    process = subprocess.Popen(
        [
            'docker', 'run', '--rm',
            '-p', '45042:45042',
            '-e', 'D3MRUN=ta2ta3',
            '-e', 'D3MINPUTDIR=/input',
            '-e', 'D3MOUTPUTDIR=/output',
            '-v', '%s:/input/dataset/' % dataset_path,
            '-v', '%s:/output' % output_path,
            'ta2:latest'

        ]
    )
    # process.wait()
    # '--name', 'ta2_container',


def create_dupms(top_pipelines):
    pipelines_list = []
    pipeline_runs_list = []

    for top_pipeline in top_pipelines:
        with open(join(D3MOUTPUTDIR, 'pipeline_runs', 'run_%s.yaml' % top_pipeline)) as fin:
            for pipeline_run in yaml_load_all(fin):
                digest = pipeline_run['pipeline']['digest']
                pipeline_runs_list.append(json.dumps(pipeline_run))

        with open(join(D3MOUTPUTDIR, 'pipelines_searched', '%s.json' % top_pipeline)) as fin:
            pipeline = json.load(fin)
            pipeline['digest'] = digest
            pipeline['source'] = {'name': 'new_nyu'}
            pipelines_list.append(json.dumps(pipeline))

    with open(join(D3MOUTPUTDIR, 'ta2', 'pipelines.json'), 'w') as fout:
        fout.write('\n'.join(pipelines_list))

    with open(join(D3MOUTPUTDIR, 'ta2', 'pipeline_runs.json'), 'w') as fout:
        fout.write('\n'.join(pipeline_runs_list))


if __name__ == '__main__':
    dataset_path = '/Users/rlopez/D3M/datasets/seed_datasets_current/185_baseball/TRAIN'
    search_pipelines(dataset_path, 2)

