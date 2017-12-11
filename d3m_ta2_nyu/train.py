import logging
import numpy
import sys
import time
import vistrails.core.db.io
from vistrails.core.db.locator import BaseLocator
from vistrails.core.interpreter.default import get_default_interpreter
from vistrails.core.modules.module_registry import get_module_registry
from vistrails.core.utils import DummyView
from vistrails.core.vistrail.controller import VistrailController

from d3m_ta2_nyu.common import SCORES_TO_SKLEARN, read_dataset


logger = logging.getLogger(__name__)


# FIXME: Duplicate code
def get_module(pipeline, label):
    for module in pipeline.module_list:
        if '__desc__' in module.db_annotations_key_index:
            name = module.get_annotation_by_key('__desc__').value
            if name == label:
                return module
    return None


def train(vt_file, pipeline, dataset, persist_dir, msg_queue):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    FOLDS = 3
    SPLIT_RATIO = 0.25
    TOTAL_PROGRESS = FOLDS + 2.0

    logger.info("About to run training pipeline, file=%r, dataset=%r",
                vt_file, dataset)

    from userpackages.simple_persist import configuration as persist_config
    from userpackages.simple_persist.init import Internal

    interpreter = get_default_interpreter()

    # Load file
    # Copied from VistrailsApplicationInterface#open_vistrail()
    locator = BaseLocator.from_url(vt_file)
    loaded_objs = vistrails.core.db.io.load_vistrail(locator)
    controller = VistrailController(loaded_objs[0], locator,
                                    *loaded_objs[1:])
    controller.select_latest_version()
    vt_pipeline = controller.current_pipeline

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

        msg_queue.put((pipeline.id, 'progress', (i + 1.0) / TOTAL_PROGRESS))

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
                                   for p in result.errors.iteritems()))
            sys.exit(1)

        predicted_results = get_module(vt_pipeline, 'test_targets').id
        predicted_results = result.objects[predicted_results]
        predicted_results = predicted_results.get_input('InternalPipe')

        run_time = time.time() - start_time

        # Compute score
        for metric in pipeline.metrics:
            # Special case
            if metric == 'EXECUTION_TIME':
                scores.setdefault(metric, []).append(run_time)
            else:
                score_func = SCORES_TO_SKLEARN[metric]
                scores.setdefault(metric, []).append(
                    score_func(test_target_split, predicted_results))

        interpreter.flush()

    msg_queue.put((pipeline.id, 'progress', (FOLDS + 1.0) / TOTAL_PROGRESS))

    # Aggregate results over the folds
    scores = dict((metric, numpy.mean(values))
                  for metric, values in scores.iteritems())
    msg_queue.put((pipeline.id, 'scores', scores))

    # Training step - run pipeline on full training_data,
    # sink = classifier-sink (the Persist downstream of the classifier),
    # Persist module set to write
    logger.info("Scoring done, running training on full data")

    # Persist trained primitives
    persist_config.file_store = persist_dir

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
                               for p in result.errors.iteritems()))
        sys.exit(1)
