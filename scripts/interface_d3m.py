import pandas
import subprocess
from os.path import join

pandas.set_option('display.max_columns', None)
pandas.set_option('display.width', None)
pandas.set_option('display.max_colwidth', -1)


class AlphaAutoml:

    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.pipelines = []
        self.leaderboard = None

    def search_pipelines(self, train_dataset, target, metric='f1Macro', task=['classification'], timeout=1):
        timeout = str(timeout * 60)
        process = subprocess.Popen(
            [
                'docker', 'run',
                '-v', '%s:/input/training_dataset.csv' % train_dataset,
                '-v', '%s:/output' % self.output_folder,
                '-e', 'D3MOUTPUTDIR=/output',
                'ta2:latest',
                'python3', '/usr/src/app/cli.py', '-o', '/output/', '-d', '/input/training_dataset.csv',  '-t', target,
                '-m', metric, '-b', timeout, '-k', ' '.join(task)
            ],
        )
        process.wait()

        self.pipelines = pandas.read_csv(join(self.output_folder, 'ta2', 'search_results.csv'))
        self.leaderboard = self.pipelines.drop(columns=['pipeline_id']).style.hide_index()

    def train(self, pipeline_index):
        pipeline_id = self.pipelines['pipeline_id'][pipeline_index]
        pipeline_path = join('/output/pipelines_searched', '%s.json' % pipeline_id)

        process = subprocess.Popen(
            [
                'docker', 'run',
                '-v', '%s:/output' % self.output_folder,
                '-e', 'D3MOUTPUTDIR=/output',
                'registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9-20200212-063959',
                'python3', '-m', 'd3m', 'runtime', 'fit', '--pipeline', pipeline_path,
                '--input', '/output/ta2/dataset_d3mformat/dataset/datasetDoc.json',
                '--problem', '/output/ta2/dataset_d3mformat/problem/problemDoc.json',
                '--save', '/output/model.pkl'
            ]
        )
        process.wait()

    def test(self, test_dataset):
        process = subprocess.Popen(
            [
                'docker', 'run',
                '-v', '%s:/output/ta2/dataset_d3mformat/dataset/tables/learningData.csv' % test_dataset,
                '-v', '%s:/output' % self.output_folder,
                'registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9-20200212-063959',
                'python3', '-m', 'd3m', 'runtime', 'produce', '--fitted-pipeline', '/output/model.pkl',
                '--test-input', '/output/ta2/dataset_d3mformat/dataset/datasetDoc.json', '--output', '/output/predictions.csv'
            ]
        )
        process.wait()

        return pandas.read_csv(self.output_folder + 'predictions.csv')


def test_model(model_path, new_dataset_path, result_path):
    process = subprocess.Popen(
                [
                    'docker', 'run',
                    '-v', '%s:/input/model.pkl' % model_path,
                    '-v', '%s:/data' % new_dataset_path,
                    '-v', '%s:/output' % result_path,
                    'registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2020.1.9-20200212-063959'
                    'python3', '-m', 'd3m', 'runtime', 'produce', '--fitted-pipeline', '/input/model.pkl',
                    '--test-input', '/data/datasetDoc.json', '--output', '/output/predictions.csv'
                ]
    )
    process.wait()

    return pandas.read_csv(result_path + 'predictions.csv')
