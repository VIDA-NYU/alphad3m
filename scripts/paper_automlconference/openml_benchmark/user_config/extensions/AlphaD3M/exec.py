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
    log.info(f'\n**** Running AlphaD3M ****\n')

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
    time_bound = 1 #config.max_runtime_seconds/60

    log.info(f'Received parameters:\n'
             f'train_dataset: {train_dataset}\n'
             f'test_dataset: {test_dataset}\n'
             f'target_name: {target_name}\n'
             f'time_bound: {time_bound}\n'
             f'metric: {metric}\n'
             f'task_type: {task_type}\n')

    automl = AutoML(output_path)

    with Timer() as training:
        automl.search_pipelines(train_dataset, time_bound=time_bound, target=target_name, metric=metric,
                                task_keywords=['classification', task_type, 'tabular'])

        best_pipeline_id = automl.get_best_pipeline_id()
        model_id = automl.train(best_pipeline_id)
        predictions = automl.test(model_id, test_dataset)
        predictions = predictions[target_name]
        automl.end_session()

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  training_duration=training.duration,
                  target_is_encoded=False)


if __name__ == '__main__':
    call_run(run)
