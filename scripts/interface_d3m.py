import os
import pandas


def test_model(model_path, new_dataset_path, result_path):
    os.system('''docker run  -it \
                    -v "%s:/input/model.pkl" \
                    -v "%s:/data" \
                    -v "%s:/output" \
                    registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10-20191127-050901 python3 -m d3m runtime produce --fitted-pipeline /input/model.pkl --test-input /data/datasetDoc.json --output /output/predictions.csv
                ''' % (model_path, new_dataset_path, result_path))

    return pandas.read_csv(result_path + 'predictions.csv')


def test_model_1():
    import subprocess
    subprocess.run(
        [
            'docker', 'run',  '-it',
            '-v', '/Users/rlopez/D3M/examples/input/model.pkl:/input/model.pkl',
            '-v', '/Users/rlopez/D3M/examples/input/new_data:/data',
            '-v', '/Users/rlopez/D3M/examples/output:/output',
            'registry.gitlab.com/datadrivendiscovery/images/primitives:ubuntu-bionic-python36-v2019.11.10-20191127-050901',
            'python3', '-m', 'd3m', 'runtime', 'produce', '--fitted-pipeline', '/input/model.pkl',
            '--test-input', '/data/datasetDoc.json', '--output', '/output/predictions.csv'
        ]
    )


test_model('/Users/rlopez/D3M/examples/input/model.pkl', '/Users/rlopez/D3M/examples/input/new_data', '/Users/rlopez/D3M/examples/output/')