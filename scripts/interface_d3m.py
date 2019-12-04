import os
import pandas


def test_model(model_path, new_dataset_path, result_path):
    os.system('''docker run  -it \
        -v "%s:/input/model.pkl" \
        -v "%s:/data" \
        -v "%s:/output" \
        registry.datadrivendiscovery.org/jpl/docker_images/complete:ubuntu-bionic-python36-v2019.6.7-20190622-073225 python3 -m d3m runtime produce --fitted-pipeline /input/model.pkl --test-input /data/datasetDoc.json --output /output/predictions.csv
        ''' % (model_path, new_dataset_path, result_path))

    return pandas.read_csv(result_path + 'predictions.csv')
