from alphad3m import AutoML
import sys


if len(sys.argv) > 1:
    dataset = sys.argv[1]
else:
    dataset = '185_baseball_MIN_METADATA'  # Default dataset

output_path = '/Users/rlopez/D3M/examples/tmp/'
resource_path = '/Users/rlopez/D3M/static/'
train_dataset = '/Users/rlopez/D3M/datasets/seed_datasets_current/%s/TRAIN' % dataset
test_dataset = '/Users/rlopez/D3M/datasets/seed_datasets_current/%s/TEST' % dataset
score_dataset = '/Users/rlopez/D3M/datasets/seed_datasets_current/%s/SCORE' % dataset


automl = AutoML(output_path, resource_folder=resource_path, verbose=True)
automl.search_pipelines(train_dataset, time_bound=10)
automl.end_session()
