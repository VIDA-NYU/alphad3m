from alphad3m import AutoML

output_path = '/Users/rlopez/D3M/examples/tmp/'
train_dataset = '/Users/rlopez/D3M/datasets/seed_datasets_current/185_baseball_MIN_METADATA/TRAIN'
test_dataset = '/Users/rlopez/D3M/datasets/seed_datasets_current/185_baseball_MIN_METADATA/TEST'
score_dataset = '/Users/rlopez/D3M/datasets/seed_datasets_current/185_baseball_MIN_METADATA/SCORE'


automl = AutoML(output_path, verbose=True)
automl.search_pipelines(train_dataset, time_bound=1)
automl.end_session()
