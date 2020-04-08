import pandas
import subprocess


def test_model(model_path, new_dataset_path, result_path):
    process = subprocess.Popen(
                [
                    'docker', 'run',
                    '-v', '%s:/input/model.pkl' % model_path,
                    '-v', '%s:/data' % new_dataset_path,
                    '-v', '%s:/output' % result_path,
                    'registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10-20191127-050901',
                    'python3', '-m', 'd3m', 'runtime', 'produce', '--fitted-pipeline', '/input/model.pkl',
                    '--test-input', '/data/datasetDoc.json', '--output', '/output/predictions.csv'
                ]
    )
    process.wait()

    return pandas.read_csv(result_path + 'predictions.csv')


def search(csv_path, output_folder):
    process = subprocess.Popen(
                [
                    'docker', 'run',
                    '-v', '%s:/output' % output_folder,
                    'ta2',
                    'python3', '-m', 'd3m', 'runtime', 'produce', '--fitted-pipeline', '/input/model.pkl',
                    '--test-input', '/data/datasetDoc.json', '--output', '/output/predictions.csv'
                ]
    )
    process.wait()