import logging
import numpy
import sys
import time

from d3m_ta2_nyu.common import SCORES_TO_SKLEARN, read_dataset


logger = logging.getLogger(__name__)


def train(pipeline_id, metrics, dataset, msg_queue):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    FOLDS = 3
    SPLIT_RATIO = 0.25
    TOTAL_PROGRESS = FOLDS + 2.0

    logger.info("About to run training pipeline, file=%r, dataset=%r",
                vt_file, dataset)

    # Load data
    data = read_dataset(dataset)
    logger.info("Loaded dataset, columns: %r", data['trainData']['columns'])

    data_frame = data['trainData']['frame']
    targets = data['trainTargets']['list']

    # Scoring step - make folds, run them through the pipeline one by one
    # (set both training_data and test_data),
    # get predictions from OutputPort to get cross validation scores
    scores = {}

    for i in range(FOLDS):
        logger.info("Scoring round %d/%d", i + 1, FOLDS)

        msg_queue.put((pipeline_id, 'progress', (i + 1.0) / TOTAL_PROGRESS))

        # Do the split
        random_sample = numpy.random.rand(len(data_frame)) < SPLIT_RATIO

        train_data_split = data_frame[random_sample]
        test_data_split = data_frame[~random_sample]

        train_target_split = targets[random_sample]
        test_target_split = targets[~random_sample]

        # Don't persist anything
        persist_config.file_store = None

        # Set input to Internal modules
        Internal.values = {
            get_module(vt_pipeline, 'training_data').id: train_data_split,
            get_module(vt_pipeline, 'training_targets').id: train_target_split,
            get_module(vt_pipeline, 'test_data').id: test_data_split,
        }

        # Select the sink
        sinks = [get_module(vt_pipeline, 'test_targets').id]

        start_time = time.time()

        results, changed = controller.execute_workflow_list([[
            controller.locator,  # locator
            controller.current_version,  # version
            vt_pipeline,  # pipeline
            DummyView(),  # view
            None,  # custom_aliases
            None,  # custom_params
            "Scoring pipeline from d3m_ta2_nyu.train",  # reason
            sinks,  # sinks
            None,  # extra_info
        ]])
        result, = results

        if result.errors:
            logger.error("Errors running pipeline:\n%s",
                         '\n'.join('%d: %s' % p
                                   for p in result.errors.items()))
            sys.exit(1)

        predicted_results = get_module(vt_pipeline, 'test_targets').id
        predicted_results = result.objects[predicted_results]
        predicted_results = predicted_results.get_input('InternalPipe')

        run_time = time.time() - start_time

        # Compute score
        for metric in metrics:
            # Special case
            if metric == 'EXECUTION_TIME':
                scores.setdefault(metric, []).append(run_time)
            else:
                score_func = SCORES_TO_SKLEARN[metric]
                scores.setdefault(metric, []).append(
                    score_func(test_target_split, predicted_results))

        interpreter.flush()

    msg_queue.put((pipeline_id, 'progress', (FOLDS + 1.0) / TOTAL_PROGRESS))

    # Aggregate results over the folds
    scores = dict((metric, numpy.mean(values))
                  for metric, values in scores.items())
    msg_queue.put((pipeline_id, 'scores', scores))

    # Training step - run pipeline on full training_data,
    # sink = classifier-sink (the Persist downstream of the classifier),
    # Persist module set to write
    logger.info("Scoring done, running training on full data")

    # Set input to Internal modules
    Internal.values = {
        get_module(vt_pipeline, 'training_data').id: data_frame,
        get_module(vt_pipeline, 'training_targets').id: targets,
    }

    # Select the sink: all the Persist modules
    registry = get_module_registry()
    descr = registry.get_descriptor_by_name('org.vistrails.vistrails.persist',
                                            'Persist')
    sinks = []
    for module in vt_pipeline.module_list:
        if module.module_descriptor == descr:
            sinks.append(module.id)

    results, changed = controller.execute_workflow_list([[
        controller.locator,  # locator
        controller.current_version,  # version
        vt_pipeline,  # pipeline
        DummyView(),  # view
        None,  # custom_aliases
        None,  # custom_params
        "Training pipeline from d3m_ta2_nyu.train",  # reason
        sinks,  # sinks
        None,  # extra_info
    ]])
    result, = results

    if result.errors:
        logger.error("Errors running pipeline:\n%s",
                     '\n'.join('%d: %s' % p
                               for p in result.errors.items()))
        sys.exit(1)
