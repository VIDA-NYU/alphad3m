import logging
import os
import tempfile as tmp

os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer
from alphad3m import AutoML

log = logging.getLogger(os.path.basename(__file__))


def run(dataset, config):
    log.info(f"\n**** Running AlphaD3M ****\n")

    metrics_mapping = {'acc': 'accuracy'}
    task_types_mapping = {'multiclass': 'multiClass', 'binary': 'binary'}
    metric = metrics_mapping.get(config.metric, None)
    task_type = task_types_mapping.get(config.type_, None)

    if metric is None:
        log.warning(f'Performance metric {metric} not supported, defaulting to accuracy')
        metric = 'accuracy'

    train_dataset = dataset.train.path
    test_dataset = dataset.test.path
    target_name = dataset.target.name
    output_path = config.output_dir
    time_bound = 3 #config.max_runtime_seconds

    automl = AutoML(output_path)
    predictions = None

    with Timer() as training:
        automl.search_pipelines(train_dataset, time_bound=time_bound, target=target_name, metric=metric,
                                task_keywords=['classification', task_type, 'tabular'])

        best_pipeline_id = automl.get_best_pipeline_id()
        model_id = automl.train(best_pipeline_id)
        predictions = automl.test(model_id, test_dataset)

    '''return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=len(estimator.estimators_) + 1,
                  training_duration=training.duration)'''


if __name__ == '__main__':
    call_run(run)
