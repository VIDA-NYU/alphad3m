import signal
import os
import sys
import operator
import logging

# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'
from d3m_ta2_nyu.primitive_loader import D3MPrimitiveLoader
from d3m_ta2_nyu.grammar_loader import format_grammar
from alphaAutoMLEdit.Coach import Coach
from alphaAutoMLEdit.pipeline.PipelineGame import PipelineGame
from alphaAutoMLEdit.pipeline.NNet import NNetWrapper
from .d3mpipeline_builder import *
from d3m_ta2_nyu.metafeature.metafeature_extractor import ComputeMetafeatures
from d3m.metadata.problem import TaskKeyword
from os.path import join
from d3m_ta2_nyu.pipeline_execute import execute
from d3m_ta2_nyu.data_ingestion.data_profiler import profile_data

logger = logging.getLogger(__name__)


config = {
    'PROBLEM_TYPES': {'CLASSIFICATION': 1,
                      'REGRESSION': 2,
                      'CLUSTERING': 3,
                      'TIME_SERIES_FORECASTING': 4,
                      'TIME_SERIES_CLASSIFICATION': 5,
                      'COMMUNITY_DETECTION': 6,
                      'GRAPH_MATCHING': 7,
                      'COLLABORATIVE_FILTERING': 8,
                      'LINK_PREDICTION': 9,
                      'VERTEX_CLASSIFICATION': 10,
                      'OBJECT_DETECTION': 11,
                      'SEMISUPERVISED_CLASSIFICATION': 12,
                      'TEXT_CLASSIFICATION': 13,
                      'IMAGE_CLASSIFICATION': 14,
                      'AUDIO_CLASSIFICATION': 15,
                      'TEXT_REGRESSION': 16,
                      'IMAGE_REGRESSION': 17,
                      'AUDIO_REGRESSION': 18,
                      'LUPI': 19,
                      'NA': 20
                      },

    'DATA_TYPES': {'TABULAR': 1,
                   'GRAPH': 2,
                   'IMAGE': 3},

    'PIPELINE_SIZE': 5,

    'ARGS': {
        'numIters': 25,
        'numEps': 5,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 5,
        'arenaCompare': 40,
        'cpuct': 1,

        'checkpoint': join(os.environ.get('D3MOUTPUTDIR'), 'ta2', 'nn_models'),
        'load_model': False,
        'load_folder_file': (join(os.environ.get('D3MOUTPUTDIR'), 'ta2', 'nn_models'), 'best.pth.tar'),
        'metafeatures_path': '/d3m/data/metafeatures',
        'verbose': True
    }
}


def list_primitives():
    sklearn_primitives = {}
    all_primitives = D3MPrimitiveLoader.get_primitives_by_type()

    for group in list(all_primitives.keys()):
        sklearn_primitives[group] = {}
        for primitive in list(all_primitives[group].keys()):
            if primitive.endswith('.SKlearn'):
                sklearn_primitives[group][primitive] = all_primitives[group][primitive]

    return all_primitives


def generate_by_templates(task_keywords, dataset, search_results, pipeline_template, metrics, problem, targets, features,
                          all_types, inferred_types, timeout, msg_queue, DBSession):
    task_keywords = set(task_keywords)

    if task_keywords & {TaskKeyword.GRAPH_MATCHING, TaskKeyword.LINK_PREDICTION, TaskKeyword.VERTEX_NOMINATION,
                        TaskKeyword.VERTEX_CLASSIFICATION, TaskKeyword.CLUSTERING, TaskKeyword.OBJECT_DETECTION,
                        TaskKeyword.COMMUNITY_DETECTION, TaskKeyword.SEMISUPERVISED, TaskKeyword.LUPI}:
        template_name = 'DEBUG_CLASSIFICATION'
    elif task_keywords & {TaskKeyword.COLLABORATIVE_FILTERING, TaskKeyword.FORECASTING}:
        template_name = 'DEBUG_REGRESSION'
    elif TaskKeyword.REGRESSION in task_keywords:
        template_name = 'REGRESSION'
        if task_keywords & {TaskKeyword.IMAGE, TaskKeyword.TEXT, TaskKeyword.AUDIO}:
            template_name = 'DEBUG_REGRESSION'
    else:
        template_name = 'CLASSIFICATION'
        if task_keywords & {TaskKeyword.IMAGE, TaskKeyword.TEXT, TaskKeyword.AUDIO}:
            template_name = 'DEBUG_CLASSIFICATION'

    logger.info("Creating pipelines from template %s" % template_name)

    # No Augmentation
    templates = BaseBuilder.TEMPLATES.get(template_name, [])
    for imputer, classifier in templates:
        pipeline_id = BaseBuilder.make_template(imputer, classifier, dataset, pipeline_template, targets, features,
                                                all_types, inferred_types, DBSession=DBSession)
        send(msg_queue, pipeline_id)

    # Augmentation
    if search_results and len(search_results) > 0:
        for search_result in search_results:
            templates = BaseBuilder.TEMPLATES_AUGMENTATION.get(template_name, [])
            for datamart, imputer, classifier in templates:
                pipeline_id = BaseBuilder.make_template_augment(datamart, imputer, classifier, dataset,
                                                                pipeline_template, targets, features, all_types,
                                                                search_result, DBSession=DBSession)
                send(msg_queue, pipeline_id)


def send(msg_queue, pipeline_id):
    msg_queue.send(('eval', pipeline_id))
    return msg_queue.recv()


def denormalize_dataset(dataset, targets, features, DBSession):
    new_path = join(os.environ.get('D3MOUTPUTDIR'), 'ta2', 'denormalized_dataset.csv')
    pipeline_id = BaseBuilder.make_denormalize_pipeline(dataset, targets, features, DBSession=DBSession)
    try:
        execute(pipeline_id, dataset, None, new_path, None,
                db_filename=join(os.environ.get('D3MOUTPUTDIR'), 'ta2', 'db.sqlite3'))  # TODO: Change this static path
    except:
        new_path = os.path.dirname(dataset[7:]) + '/tables/learningData.csv'
        logger.exception('Error denormalizing dataset, using only learningData.csv file')

    return new_path


@database.with_sessionmaker
def generate(task_keywords, dataset, search_results, pipeline_template, metrics, problem, targets, features, timeout,
             msg_queue, DBSession):

    import time
    start = time.time()

    with open(dataset[7:]) as fin:
        dataset_doc = json.load(fin)

    index_name = 'd3mIndex'
    target_names = [x[1] for x in targets]
    csv_path = denormalize_dataset(dataset, targets, features, DBSession)
    inferred_types, all_types = profile_data(csv_path, index_name, target_names, dataset_doc)

    if os.environ.get('SKIPTEMPLATES', 'not') == 'not':
        generate_by_templates(task_keywords, dataset, search_results, pipeline_template, metrics, problem, targets,
                              features, all_types, inferred_types, timeout, msg_queue, DBSession)

    if 'TA2_DEBUG_BE_FAST' in os.environ:
        sys.exit(0)

    builder = None
    task_name = 'CLASSIFICATION' if TaskKeyword.CLASSIFICATION in task_keywords else 'REGRESSION'

    def eval_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = builder.make_d3mpipeline(strings, origin, dataset, search_results, pipeline_template, targets,
                                               features, all_types, inferred_types, DBSession=DBSession)
        # Evaluate the pipeline if syntax is correct:
        if pipeline_id:
            msg_queue.send(('eval', pipeline_id))
            return msg_queue.recv()
        else:
            return None

    if TaskKeyword.CLUSTERING in task_keywords:
        task_name = 'CLUSTERING'
        builder = BaseBuilder()
    elif TaskKeyword.SEMISUPERVISED in task_keywords:
        task_name = 'SEMISUPERVISED_CLASSIFICATION'
        builder = BaseBuilder()
    elif TaskKeyword.COLLABORATIVE_FILTERING in task_keywords:
        task_name = 'COLLABORATIVE_FILTERING'
        builder = BaseBuilder()
    elif TaskKeyword.COMMUNITY_DETECTION in task_keywords:
        task_name = 'COMMUNITY_DETECTION'
        builder = CommunityDetectionBuilder()
    elif TaskKeyword.LINK_PREDICTION in task_keywords:
        task_name = 'LINK_PREDICTION'
        builder = LinkPredictionBuilder()
    elif TaskKeyword.OBJECT_DETECTION in task_keywords:
        task_name = 'OBJECT_DETECTION'
        builder = ObjectDetectionBuilder()
    elif TaskKeyword.GRAPH_MATCHING in task_keywords:
        task_name = 'GRAPH_MATCHING'
        builder = GraphMatchingBuilder()
    elif TaskKeyword.FORECASTING in task_keywords:
        task_name = 'TIME_SERIES_FORECASTING'
        builder = TimeseriesForecastingBuilder()
    elif TaskKeyword.TIME_SERIES in task_keywords and TaskKeyword.CLASSIFICATION in task_keywords:
        task_name = 'TIME_SERIES_CLASSIFICATION'
        builder = TimeseriesClassificationBuilder()
    elif TaskKeyword.VERTEX_CLASSIFICATION in task_keywords or TaskKeyword.VERTEX_NOMINATION in task_keywords:
        task_name = 'VERTEX_CLASSIFICATION'
        builder = VertexClassificationBuilder()
    elif TaskKeyword.LUPI in task_keywords:
        task_name = 'LUPI'
        builder = LupiBuilder()
    elif TaskKeyword.TEXT in task_keywords and (
            TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'TEXT_' + task_name
        builder = BaseBuilder()
    elif TaskKeyword.IMAGE in task_keywords and (
            TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'IMAGE_' + task_name
        builder = BaseBuilder()
    elif TaskKeyword.AUDIO in task_keywords and (
            TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'AUDIO_' + task_name
        builder = AudioBuilder()
    elif TaskKeyword.CLASSIFICATION in task_keywords or TaskKeyword.REGRESSION in task_keywords:
        builder = BaseBuilder()
    else:
        logger.warning('Task %s doesnt exist in the grammar, using default NA_TASK' % task_name)
        task_name = 'NA'
        builder = BaseBuilder()

    def update_config(selected_primitves, task_name):
        metafeatures_extractor = ComputeMetafeatures(dataset, targets, features, DBSession)
        config['GRAMMAR'] = format_grammar(task_name + '_TASK', selected_primitves)
        config['PROBLEM'] = task_name
        config['DATA_TYPE'] = 'TABULAR'
        config['METRIC'] = metrics[0]['metric'].name
        config['DATASET_METAFEATURES'] = [0] * 50 #  metafeatures_extractor.compute_metafeatures('AlphaD3M_compute_metafeatures')
        config['DATASET'] = dataset_doc['about']['datasetID']
        config['ARGS']['stepsfile'] = join(os.environ.get('D3MOUTPUTDIR'), 'ta2', config['DATASET'] + '_pipeline_steps.txt')

        return config

    def signal_handler(signal, frame):
        logger.info('Receiving SIGTERM signal')
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    primitives = list_primitives()
    config_updated = update_config(primitives, task_name)
    game = PipelineGame(config_updated, eval_pipeline)
    nnet = NNetWrapper(game)

    if config['ARGS'].get('load_model'):
        model_file = join(config['ARGS'].get('load_folder_file')[0],
                          config['ARGS'].get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(config['ARGS'].get('load_folder_file')[0],
                                 config['ARGS'].get('load_folder_file')[1])

    c = Coach(game, nnet, config['ARGS'])
    c.learn()

    record_bestpipeline(config['DATASET'])

    sys.exit(0)
