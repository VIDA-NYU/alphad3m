import signal
import os
import sys
import operator
import logging
import datamart_profiler
import pandas as pd

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
from d3m.container.dataset import D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES

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
                      'NA': 19
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

        'checkpoint': os.path.join(os.environ.get('D3MOUTPUTDIR'), 'ta2', 'nn_models'),
        'load_model': False,
        'load_folder_file': (os.path.join(os.environ.get('D3MOUTPUTDIR'), 'ta2', 'nn_models'), 'best.pth.tar'),
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


def generate_by_templates(task, dataset, search_results, pipeline_template, metrics, problem, targets, features,
                          all_types, inferred_types, timeout, msg_queue, DBSession):
    logger.info("Creating pipelines from templates...")

    if task in [TaskKeyword.GRAPH_MATCHING, TaskKeyword.LINK_PREDICTION, TaskKeyword.VERTEX_NOMINATION,
                TaskKeyword.VERTEX_CLASSIFICATION, TaskKeyword.CLUSTERING,
                TaskKeyword.OBJECT_DETECTION, TaskKeyword.COMMUNITY_DETECTION, TaskKeyword.TIME_SERIES,
                TaskKeyword.SEMISUPERVISED]:
        template_name = 'DEBUG_CLASSIFICATION'
    elif task in [TaskKeyword.COLLABORATIVE_FILTERING, TaskKeyword.FORECASTING]:
        template_name = 'DEBUG_REGRESSION'
    else:
        template_name = task.name

    if 'TA2_DEBUG_BE_FAST' in os.environ:
        template_name = 'DEBUG_' + task.name

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

    if 'TA2_DEBUG_BE_FAST' in os.environ:
        sys.exit(0)


def send(msg_queue, pipeline_id):
    msg_queue.send(('eval', pipeline_id))
    return msg_queue.recv()


def read_feature_types(dataset_doc):
    feature_types = {}
    try:
        for data_res in dataset_doc['dataResources']:
            if data_res['resID'] == 'learningData':
                for column in data_res['columns']:
                    if 'attribute' in column['role'] and column['colType'] != 'unknown':
                        feature_types[column['colName']] = D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES[column['colType']]
                break
    except:
        logger.exception('Error reading the type of attributes')

    return feature_types


def select_features2identify(csv_path, annotated_features, target_name, index_name):
    all_features = pd.read_csv(csv_path).columns
    features2identify = []

    for feature_name in all_features:
        if feature_name != target_name and feature_name != index_name and feature_name not in annotated_features:
            features2identify.append(feature_name)

    return features2identify


def indentify_feature_types(csv_path, features2identify, target_name, index_name):
    metadata = datamart_profiler.process_dataset(csv_path)
    new_types = {'https://metadata.datadrivendiscovery.org/types/Attribute': []}

    for index, item in enumerate(metadata['columns']):
        column_name = item['name']
        if column_name == index_name:
            new_types['https://metadata.datadrivendiscovery.org/types/PrimaryKey'] = [index]
        elif column_name == target_name:
            new_types['https://metadata.datadrivendiscovery.org/types/TrueTarget'] = [index]
        elif column_name in features2identify:
            semantic_types = item['semantic_types'] if len(item['semantic_types']) > 0 else [item['structural_type']]
            for semantic_type in semantic_types:
                if semantic_type == 'http://schema.org/Enumeration':  # Changing to D3M format
                    semantic_type = 'https://metadata.datadrivendiscovery.org/types/CategoricalData'
                if semantic_type not in new_types:
                    new_types[semantic_type] = []
                new_types[semantic_type].append(index)

            new_types['https://metadata.datadrivendiscovery.org/types/Attribute'].append(index)

    return new_types


@database.with_sessionmaker
def generate(task_keywords, dataset, search_results, pipeline_template, metrics, problem, targets, features, timeout,
             msg_queue,
             DBSession):

    import time
    start = time.time()

    with open(dataset[7:]) as fin:
        dataset_doc = json.load(fin)
        csv_path = os.path.dirname(dataset[7:]) + '/tables/learningData.csv'

    task = task_keywords[0]
    task_name = task.name

    target_name = list(targets)[0][1]
    index_name = 'd3mIndex'
    annotated_types = read_feature_types(dataset_doc)
    features2identify = select_features2identify(csv_path, annotated_types.keys(), target_name, index_name)
    new_types = {}
    if len(features2identify) > 0:
        new_types = indentify_feature_types(csv_path, features2identify, target_name, index_name)
    all_types = list(annotated_types.values()) + list(new_types.keys())
    print('annotated_features', annotated_types)
    print('features2identify', features2identify)
    print('new_types', new_types)
    print('all_types', all_types)
    generate_by_templates(task, dataset, search_results, pipeline_template, metrics, problem, targets, features,
                          all_types, new_types, timeout, msg_queue, DBSession)

    builder = None

    def eval_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = builder.make_d3mpipeline(strings, origin, dataset, search_results, pipeline_template, targets,
                                               features, all_types, new_types, DBSession=DBSession)
        # Evaluate the pipeline if syntax is correct:
        if pipeline_id:
            msg_queue.send(('eval', pipeline_id))
            return msg_queue.recv()
        else:
            return None

    if TaskKeyword.CLUSTERING in task_keywords:
        builder = BaseBuilder()
    if TaskKeyword.SEMISUPERVISED in task_keywords:
        task_name = 'SEMISUPERVISED_CLASSIFICATION'
        builder = BaseBuilder()
    elif TaskKeyword.COLLABORATIVE_FILTERING in task_keywords:
        builder = BaseBuilder()
    elif TaskKeyword.COMMUNITY_DETECTION in task_keywords:
        builder = CommunityDetectionBuilder()
    elif TaskKeyword.LINK_PREDICTION in task_keywords:
        builder = LinkPredictionBuilder()
    elif TaskKeyword.OBJECT_DETECTION in task_keywords:
        builder = ObjectDetectionBuilder()
    elif TaskKeyword.GRAPH_MATCHING in task_keywords:
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
        config['ARGS']['stepsfile'] = os.path.join(os.environ.get('D3MOUTPUTDIR'), 'ta2', config['DATASET'] + '_pipeline_steps.txt')

        return config

    def record_bestpipeline(dataset):
        end = time.time()

        eval_dict = game.evaluations
        eval_times = game.eval_times
        for key, value in eval_dict.items():
            if value == float('inf') and 'error' not in game.metric.lower():
                eval_dict[key] = 0
        evaluations = sorted(eval_dict.items(), key=operator.itemgetter(1))
        if 'error' not in game.metric.lower():
            evaluations.reverse()

        out_p = open(os.path.join(os.environ.get('D3MOUTPUTDIR'), 'ta2', config['DATASET'] + '_best_pipelines.txt'), 'a')
        out_p.write(
            config['DATASET'] + ' ' + evaluations[0][0] + ' ' + str(evaluations[0][1]) + ' ' + str(
                game.steps) + ' ' + str((eval_times[evaluations[0][0]] - start) / 60.0) + ' ' + str(
                (end - start) / 60.0) + '\n')

    def signal_handler(signal, frame):
        logger.info('Receiving SIGTERM signal')
        record_bestpipeline(config['DATASET'])
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    primitives = list_primitives()
    config_updated = update_config(primitives, task_name)
    game = PipelineGame(config_updated, eval_pipeline)
    nnet = NNetWrapper(game)

    if config['ARGS'].get('load_model'):
        model_file = os.path.join(config['ARGS'].get('load_folder_file')[0],
                                  config['ARGS'].get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(config['ARGS'].get('load_folder_file')[0],
                                 config['ARGS'].get('load_folder_file')[1])

    c = Coach(game, nnet, config['ARGS'])
    c.learn()

    record_bestpipeline(config['DATASET'])

    sys.exit(0)
