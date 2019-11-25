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

logger = logging.getLogger(__name__)

    
def get_primitives():
    sklearn_primitives = {}
    all_primitives = D3MPrimitiveLoader.get_primitives_by_type()

    for group in list(all_primitives.keys()):
        sklearn_primitives[group] = {}
        for primitive in list(all_primitives[group].keys()):
            if primitive.endswith('.SKlearn'):
                sklearn_primitives[group][primitive] = all_primitives[group][primitive]

    return all_primitives, sklearn_primitives


ALL_PRIMITIVES, SKLEARN_PRIMITIVES = get_primitives()

input = {
        'PROBLEM_TYPES': {'CLASSIFICATION': 1,
                          'REGRESSION': 2,
                          'CLUSTERING': 3,
                          'TIME_SERIES_FORECASTING': 4,
                          'TIME_SERIES_CLASSIFICATION': 5,
                          'COMMUNITY_DETECTION': 6,
                          'GRAPH_MATCHING': 7,
                          'COLLABORATIVE_FILTERING': 8,
                          'LINK_PREDICTION': 9,
                          'VERTEX_NOMINATION': 10,
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

            'checkpoint': '/output/nn_models',
            'load_model': False,
            'load_folder_file': ('/output/nn_models', 'best.pth.tar'),
            'metafeatures_path': '/d3m/data/metafeatures',
            'verbose': True
        }
    }


def generate_by_templates(task, dataset, search_results, pipeline_template, metrics, problem, targets, features,
                          feature_types, timeout, msg_queue, DBSession):
    logger.info("Creating pipelines from templates...")

    if task in ['TIME_SERIES_CLASSIFICATION', 'GRAPH_MATCHING', 'LINK_PREDICTION', 'VERTEX_NOMINATION', 'CLUSTERING',
                'SEMISUPERVISED_CLASSIFICATION', 'OBJECT_DETECTION', 'VERTEX_CLASSIFICATION', 'COMMUNITY_DETECTION']:
        template_name = 'CLASSIFICATION'
    elif task in ['TIME_SERIES_FORECASTING', 'COLLABORATIVE_FILTERING']:
        template_name = 'REGRESSION'
    else:
        template_name = task
    if 'TA2_DEBUG_BE_FAST' in os.environ:
        template_name = 'DEBUG_' + task

    # No Augmentation
    templates = BaseBuilder.TEMPLATES.get(template_name, [])
    for imputer, classifier in templates:
        pipeline_id = BaseBuilder.make_template(imputer, classifier, dataset, pipeline_template, targets, features,
                                                feature_types, DBSession=DBSession)
        send(msg_queue, pipeline_id)

    # Augmentation
    if search_results and len(search_results) > 0:
        for search_result in search_results:
            templates = BaseBuilder.TEMPLATES_AUGMENTATION.get(template_name, [])
            for datamart, imputer, classifier in templates:
                pipeline_id = BaseBuilder.make_template_augment(datamart, imputer, classifier, dataset,
                                                                pipeline_template, targets, features, feature_types,
                                                                search_result, DBSession=DBSession)
                send(msg_queue, pipeline_id)

    # MeanBaseline pipeline
    pipeline_id = BaseBuilder.make_meanbaseline('MeanBaseline', dataset, DBSession)
    send(msg_queue, pipeline_id)


def send(msg_queue, pipeline_id):
    msg_queue.send(('eval', pipeline_id))
    return msg_queue.recv()


def get_feature_types(dataset_doc):
    feature_types = {}
    try:
        for data_res in dataset_doc['dataResources']:
            if data_res['resID'] == 'learningData':
                for column in data_res['columns']:
                    if 'attribute' in column['role']:
                        if column['colType'] not in feature_types:
                            feature_types[column['colType']] = []
                        feature_types[column['colType']].append(column['colName'])
                break
    except Exception as e:
        logger.error(e)

    return feature_types


@database.with_sessionmaker
def generate(task, dataset, search_results, pipeline_template, metrics, problem, targets, features, timeout, msg_queue,
             DBSession):
    import time
    start = time.time()

    with open(dataset[7:]) as fin:
        dataset_doc = json.load(fin)

    feature_types = get_feature_types(dataset_doc)
    generate_by_templates(task, dataset, search_results, pipeline_template, metrics, problem, targets, features,
                          feature_types, timeout, msg_queue, DBSession)

    builder = None

    def eval_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = builder.make_d3mpipeline(strings, origin, dataset, search_results, pipeline_template, targets,
                                               features, DBSession=DBSession)
        # Evaluate the pipeline if syntax is correct:
        if pipeline_id:
            msg_queue.send(('eval', pipeline_id))
            return msg_queue.recv()
        else:
            return None

    data_types = []
    for data_res in dataset_doc['dataResources']:
        data_types.append(data_res['resType'])

    if 'CLUSTERING' in task:
        builder = BaseBuilder()
    if 'SEMISUPERVISED_CLASSIFICATION' in task:
        builder = BaseBuilder()
    elif 'COLLABORATIVE_FILTERING' in task:
        builder = BaseBuilder()
    elif 'COMMUNITY_DETECTION' in task:
        builder = CommunityDetectionBuilder()
    elif 'LINK_PREDICTION' in task:
        builder = LinkPredictionBuilder()
    elif 'OBJECT_DETECTION' in task:
        builder = ObjectDetectionBuilder()
    elif 'GRAPH_MATCHING' in task:
        builder = GraphMatchingBuilder()
    elif 'TIME_SERIES_FORECASTING' in task:
        builder = TimeseriesForecastingBuilder()
    elif 'timeseries' in data_types and 'CLASSIFICATION' in task:
        task = 'TIME_SERIES_CLASSIFICATION'
        builder = TimeseriesClassificationBuilder()
    elif 'VERTEX_NOMINATION' in task or 'VERTEX_CLASSIFICATION' in task:
        task = 'VERTEX_NOMINATION'
        builder = VertexNominationBuilder()
    elif 'text' in data_types and ('REGRESSION' in task or 'CLASSIFICATION' in task):
        task = 'TEXT_' + task
        builder = BaseBuilder()
    elif 'image' in data_types and ('REGRESSION' in task or 'CLASSIFICATION' in task):
        task = 'IMAGE_' + task
        builder = BaseBuilder()
    elif 'audio' in data_types and ('REGRESSION' in task or 'CLASSIFICATION' in task):
        task = 'AUDIO_' + task
        builder = AudioBuilder()
    elif 'CLASSIFICATION' in task or 'REGRESSION' in task:
        builder = BaseBuilder()
    else:
        logger.warning('Task %s doesnt exist in the grammar, using default NA_TASK' % task)
        task = 'NA'
        builder = BaseBuilder()

    def create_input(selected_primitves):
        task_name = task + '_TASK'
        metafeatures_extractor = ComputeMetafeatures(dataset, targets, features, DBSession)
        input['GRAMMAR'] = format_grammar(task_name, selected_primitves)
        input['PROBLEM'] = task
        input['DATA_TYPE'] = 'TABULAR'
        input['METRIC'] = metrics[0]['metric']
        input['DATASET_METAFEATURES'] = metafeatures_extractor.compute_metafeatures('AlphaD3M_compute_metafeatures')
        input['DATASET'] = dataset_doc['about']['datasetName']
        input['ARGS']['stepsfile'] = os.path.join('/output', input['DATASET'] + '_pipeline_steps.txt')

        return input

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

        out_p = open(os.path.join('/output', input['DATASET'] + '_best_pipelines.txt'), 'a')
        out_p.write(
            dataset_doc['about']['datasetName'] + ' ' + evaluations[0][0] + ' ' + str(evaluations[0][1]) + ' ' + str(
                game.steps) + ' ' + str((eval_times[evaluations[0][0]] - start) / 60.0) + ' ' + str(
                (end - start) / 60.0) + '\n')

    def signal_handler(signal, frame):
        logger.info('Receiving SIGTERM signal')
        record_bestpipeline(input['DATASET'])
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)

    input_all = create_input(ALL_PRIMITIVES)
    game = PipelineGame(input_all, eval_pipeline)
    nnet = NNetWrapper(game)

    if input['ARGS'].get('load_model'):
        model_file = os.path.join(input['ARGS'].get('load_folder_file')[0],
                                  input['ARGS'].get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(input['ARGS'].get('load_folder_file')[0],
                                 input['ARGS'].get('load_folder_file')[1])

    c = Coach(game, nnet, input['ARGS'])
    c.learn()

    record_bestpipeline(input['DATASET'])

    sys.exit(0)
