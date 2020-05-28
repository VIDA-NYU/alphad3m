import json
import time
import logging
import subprocess
import pandas as pd
from os import listdir
from os.path import join, split, isdir
from d3m.utils import yaml_load_all
from client import BasicTA3

logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


TA2_DOCKER_IMAGES = {'NYU': 'ta2:latest', 'TAMU': 'dmartinez05/tamuta2:latest'}


class Automl:

    def __init__(self, output_folder, ta2='NYU'):
        self.client = None
        self.container = None
        self.output_folder = output_folder
        self.leaderboard = pd.DataFrame()
        self.pipelines = []
        self.ta2 = ta2

    def search_pipelines(self, dataset_path, time_bound=1):
        self.start_ta2(dataset_path, self.output_folder)
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

    def test_score(self, pipeline_id, dataset_path_test):
        #dataset_path = split(dataset_path_test)[0]
        dataset_path = '/input/dataset/'
        dataset_train_path = join(dataset_path, 'TRAIN/dataset_TRAIN/datasetDoc.json')
        dataset_test_path = join(dataset_path, 'TEST/dataset_TEST/datasetDoc.json')
        dataset_score_path = join(dataset_path, 'SCORE/dataset_TEST/datasetDoc.json')
        problem_path = join(dataset_path, 'TRAIN/problem_TRAIN/problemDoc.json')
        #pipeline_path = join(self.output_folder, 'pipelines_searched', '%s.json' % pipeline_id)
        #score_pipeline_path = join(self.output_folder, 'fit_score_%s.csv' % pipeline_id)
        pipeline_path = join('/output/', 'pipelines_searched', '%s.json' % pipeline_id)
        score_pipeline_path = join('/output/', 'fit_score_%s.csv' % pipeline_id)
        metric = None
        score = None

        command = [
            'docker', 'exec', '-it', 'ta2_container',
            'python3', '-m', 'd3m',
            'runtime',
            '--context', 'TESTING',
            '--random-seed', '0',
            'fit-score',
            '--pipeline', pipeline_path,
            '--problem', problem_path,
            '--input', dataset_train_path,
            '--test-input', dataset_test_path,
            '--score-input', dataset_score_path,
            '--scores', score_pipeline_path
        ]
        try:
            process = subprocess.Popen(command)
            process.wait()
            df = pd.read_csv(join(self.output_folder, 'fit_score_%s.csv' % pipeline_id))
            score = round(df['value'][0], 5)
            metric = df['metric'][0].lower()
        except Exception as e:
            print('Error calculating test score', e)

        return (metric, score)

    def create_profiler_inputs(self):
        pipeline_ids = {p['id'] for p in self.pipelines}
        pipeline_runs_folder = join(self.output_folder, 'pipeline_runs')
        profiler_inputs = []
        pipeline_runs = {}

        if not isdir(pipeline_runs_folder):
            print('Folder "%s" does not exist' % pipeline_runs_folder)
            return profiler_inputs

        print('Loading pipeline runs...')
        #  Loading YAML files
        file_names = [f for f in listdir(pipeline_runs_folder) if f.endswith('.yml')]
        for file_name in file_names:
            with open(join(pipeline_runs_folder, file_name)) as fin:
                for pipeline_run in yaml_load_all(fin):
                    pipeline_id = pipeline_run['pipeline']['id']
                    if pipeline_id in pipeline_ids and pipeline_run['run']['phase'] == 'PRODUCE':
                        run_data = {
                            'problem': pipeline_run['problem'],
                            'start': pipeline_run['start'],
                            'end': pipeline_run['end'],
                            'scores': pipeline_run['run']['results']['scores'],
                            'digest': pipeline_run['pipeline']['digest']
                        }
                        if pipeline_id not in pipeline_runs:
                            pipeline_runs[pipeline_id] = []
                        pipeline_runs[pipeline_id].append(run_data)

        #  Loading JSON files
        for pipeline_id in pipeline_ids:
            with open(join(self.output_folder, 'pipelines_searched', '%s.json' % pipeline_id)) as fin:
                pipeline = json.load(fin)
                if len(pipeline_runs[pipeline_id]) > 1:  # There was used cross validation, so calculate avg of scores
                    scores = []
                    for fold_run in pipeline_runs[pipeline_id]:
                        scores.append(fold_run['scores'][0]['value'])
                    avg_score = sum(scores) / len(scores)

                pipeline_run = pipeline_runs[pipeline_id][0]
                pipeline_run['scores'][0]['value'] = avg_score

                if 'normalized' not in pipeline_run['scores'][0]:  # For pipeline_runs without normalized metric
                    pipeline_run['scores'][0]['normalized'] = pipeline_run['scores'][0]['value']

                profiler_data = {
                    'pipeline_id': pipeline['id'],
                    'inputs': pipeline['inputs'],
                    'steps': pipeline['steps'],
                    'outputs': pipeline['outputs'],
                    'pipeline_digest': pipeline_run['digest'],
                    'problem': pipeline_run['problem'],
                    'start': pipeline_run['start'],
                    'end': pipeline_run['end'],
                    'scores': pipeline_run['scores'],
                    'pipeline_source': {'name': self.ta2},
                }
                profiler_inputs.append(profiler_data)

        print('Loading finished!')
        return profiler_inputs

    def start_ta2(self, dataset_path, output_path):
        print('Initializing TA2...')
        dataset_path = split(dataset_path)[0]
        self.container = subprocess.Popen(
            [
                'docker', 'run', '--rm',
                '--name', 'ta2_container',
                '-p', '45042:45042',
                '-e', 'D3MRUN=ta2ta3',
                '-e', 'D3MINPUTDIR=/input',
                '-e', 'D3MOUTPUTDIR=/output',
                '-v', '%s:/input/dataset/' % dataset_path,
                '-v', '%s:/output' % output_path,
                TA2_DOCKER_IMAGES[self.ta2]
            ]
        )
        time.sleep(4)  # Wait for TA2
        while True:
            try:
                self.client = BasicTA3()
                self.client.do_hello()
                print('TA2 initialized!')
                break
            except:
                print('Trying again to initialize TA2...')
                if self.client.channel is not None:
                    self.client.channel.close()
                    self.client = None

                time.sleep(4)

    def end_session(self):
        print('Ending session...')
        if self.container is not None:
            process = subprocess.Popen(['docker', 'stop', 'ta2_container'])
            process.wait()

        print('Session ended!')


if __name__ == '__main__':
    output_path = '/Users/rlopez/D3M/tmp/'
    train_dataset = '/Users/rlopez/D3M/datasets/seed_datasets_current/185_baseball/TRAIN'
    test_dataset = '/Users/rlopez/D3M/datasets/seed_datasets_current/185_baseball/TEST'

    automl = Automl(output_path)
    pipelines = automl.search_pipelines(train_dataset)
    #model = automl.train(pipelines[0]['id'])
    #predictions = automl.test(model, test_dataset)
    #automl.create_profiler_inputs()
    automl.test_score('71c560ae-542f-42e2-93ec-a70374b1dd62', test_dataset)
    #automl.end_session()

