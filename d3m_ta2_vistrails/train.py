import logging
import numpy
import sklearn
import vistrails.core.db.io
from vistrails.core.db.locator import BaseLocator
from vistrails.core.vistrail.controller import VistrailController

from d3m_ta2_vistrails.common import read_dataset


logger = logging.getLogger(__name__)


# FIXME: Duplicate code
def get_module(pipeline, label):
    for module in pipeline.module_list:
        if '__desc__' in module.db_annotations_key_index:
            name = module.get_annotation_by_key('__desc__').value
            if name == label:
                return module
    return None


SCORES = dict(
    R_SQUARED=sklearn.metrics.r2_score,
)


def train(vt_file, pipeline, dataset, msg_queue):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

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

    # Scoring step - make folds, run them through the pipeline one by one
    # (set both training_data and test_data),
    # get predictions from OutputPort to get cross validation scores
    FOLDS = 3
    SPLIT_RATIO = 0.25

    scores = {}

    for _ in range(FOLDS):
        data_frame = data['trainData']['frame']
        targets = data['trainTargets']['list']

        random_sample = numpy.random.rand(len(data_frame)) < SPLIT_RATIO

        train_data_split = data_frame[random_sample]
        test_data_split = data_frame[~random_sample]

        train_target_split = targets[random_sample]
        test_target_split = targets[~random_sample]

        # Set input to Internal modules
        from userpackages.simple_persist.init import Internal

        Internal.values[get_module(vt_pipeline, 'training_data').id] = \
            data['trainingData']['frame']

        results, changed = controller.execute_workflow_list([[
            controller.locator,  # locator
            controller.current_version,  # version
            vt_pipeline,  # pipeline
            DummyView(),  # view
            None,  # custom_aliases
            None,  # custom_params
            "Training pipeline from d3m_ta2_vistrails.train",  # reason
            sinks,  # sinks
            None,  # extra_info
        ]])
        result, = results

        # Compute score
        for score in pipeline.metrics:
            score_func = SCORES[score]
            scores.setdefault(score, []).append(
                score_func(test_target_split, predicted_results))

    # Aggregate results over the folds
    scores = dict((metric, mean(values)) for metric, values in scores)

    # Training step - run pipeline on full training_data,
    # sink = classifier-sink (the Persist downstream of the classifier),
    # Persist module set to write
