import json
import time
import logging
import subprocess
import pandas as pd
import datetime
from os.path import join, split, isdir
from client import BasicTA3

logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


TA2_DOCKER_IMAGES = {'NYU': 'ta2:latest', 'TAMU': 'dmartinez05/tamuta2:latest'}
IGNORE_PRIMITIVES = {'d3m.primitives.data_transformation.construct_predictions.Common',
                     'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                     'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                     'd3m.primitives.data_transformation.denormalize.Common',
                     'd3m.primitives.data_transformation.column_parser.Common'}


class Automl:

    def __init__(self, output_folder, ta2='NYU'):
        self.client = None
        self.container = None
        self.output_folder = output_folder
        self.leaderboard = pd.DataFrame()
        self.pipelines = []
        self.ta2 = ta2
        self.dataset = None

    def search_pipelines(self, dataset_path, time_bound=10):
        self.start_ta2(dataset_path, self.output_folder)
        self.dataset = dataset_path
        dataset_train_path = '/input/dataset/TRAIN/dataset_TRAIN/datasetDoc.json'
        problem_path = join(dataset_path, 'problem_TRAIN/problemDoc.json')
        start_time = datetime.datetime.utcnow()
        pipelines = self.client.do_search(dataset_train_path, problem_path, time_bound, pipelines_limit=0)

        for pipeline in pipelines:
            end_time = datetime.datetime.utcnow()
            pipeline_json_id = self.client.do_describe(pipeline['id'])

            with open(join(self.output_folder, 'pipelines_searched', '%s.json' % pipeline_json_id)) as fin:
                pipeline_json = json.load(fin)
                summary_pipeline = self.get_summary_pipeline(pipeline_json)
                pipeline['json_representation'] = pipeline_json
                pipeline['summary'] = summary_pipeline
                pipeline['found_time'] = end_time.isoformat() + 'Z'

            duration = str(end_time - start_time)
            print('Found pipeline, id=%s, summary=%s, %s=%s, time=%s' % (pipeline['id'], pipeline['summary'],
                                                                    pipeline['metric'], pipeline['score'], duration))

            self.pipelines.append(pipeline)

        if len(self.pipelines) > 0:
            leaderboard = []
            sorted_pipelines = sorted(self.pipelines, key=lambda x: x['normalized_score'], reverse=True)
            metric = sorted_pipelines[0]['metric']
            for position, pipeline_data in enumerate(sorted_pipelines, 1):
                leaderboard.append([position, pipeline_data['id'], pipeline_data['summary'],  pipeline_data['score']])

            self.leaderboard = pd.DataFrame(leaderboard, columns=['ranking', 'id', 'summary', metric]).style.hide_index()

        return self.pipelines

    def get_summary_pipeline(self, pipeline_json):
        primitives = []
        for primitive in pipeline_json['steps']:
            primitive_name = primitive['primitive']['python_path']
            if primitive_name not in IGNORE_PRIMITIVES:
                primitive_name = '.'.join(primitive_name.split('.')[-2:]).lower()
                primitives.append(primitive_name)

        return ', '.join(primitives)

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
        profiler_inputs = []

        for pipeline in self.pipelines:
            if 'digest' not in pipeline['json_representation']:
                pipeline['json_representation']['digest'] = ''  # TODO: Compute digest

            pipeline['score'] = [{'metric': {'metric': pipeline['metric']}, 'value': pipeline['score'],
                                  'normalized': pipeline['normalized_score']}]

            profiler_data = {
                'pipeline_id': pipeline['json_representation']['id'],
                'inputs': pipeline['json_representation']['inputs'],
                'steps': pipeline['json_representation']['steps'],
                'outputs': pipeline['json_representation']['outputs'],
                'pipeline_digest': pipeline['json_representation']['digest'],
                'problem': self.dataset,
                'start': pipeline['json_representation']['created'],
                'end': pipeline['found_time'],
                'scores': pipeline['score'],
                'pipeline_source': {'name': self.ta2},
            }
            profiler_inputs.append(profiler_data)

        return profiler_inputs

    def start_ta2(self, dataset_path, output_path):
        print('Initializing TA2...')
        # TODO: Verify if other TA2 is running
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

    automl = Automl(output_path, 'TAMU')
    pipelines = automl.search_pipelines(train_dataset, time_bound=1)
    #model = automl.train(pipelines[0]['id'])
    #predictions = automl.test(model, test_dataset)
    #automl.create_profiler_inputs()
    #automl.test_score('71c560ae-542f-42e2-93ec-a70374b1dd62', test_dataset)
    automl.end_session()

