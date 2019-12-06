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

    if task in [TaskKeyword.GRAPH_MATCHING, TaskKeyword.LINK_PREDICTION, TaskKeyword.VERTEX_NOMINATION, TaskKeyword.VERTEX_CLASSIFICATION, TaskKeyword.CLUSTERING,
                TaskKeyword.OBJECT_DETECTION, TaskKeyword.COMMUNITY_DETECTION, TaskKeyword.TIME_SERIES, TaskKeyword.SEMISUPERVISED]:  # 'TIME_SERIES_CLASSIFICATION', 'SEMISUPERVISED_CLASSIFICATION'
        template_name = 'DEBUG_CLASSIFICATION'
    elif task in [TaskKeyword.COLLABORATIVE_FILTERING, TaskKeyword.FORECASTING]:  # 'TIME_SERIES_FORECASTING']
        template_name = 'DEBUG_REGRESSION'
    else:
        template_name = task.name

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
    #pipeline_id = BaseBuilder.make_meanbaseline('MeanBaseline', dataset, DBSession)
    #send(msg_queue, pipeline_id)


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
def generate(task_keywords, dataset, search_results, pipeline_template, metrics, problem, targets, features, timeout, msg_queue,
             DBSession):
    import time
    start = time.time()

    with open(dataset[7:]) as fin:
        dataset_doc = json.load(fin)

    task = task_keywords[0]
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

    task_name = task.name

    if TaskKeyword.CLUSTERING in task_keywords:
        builder = BaseBuilder()
    if TaskKeyword.SEMISUPERVISED in task_keywords:  # to review
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
    elif TaskKeyword.FORECASTING in task_keywords:  # to review
        builder = TimeseriesForecastingBuilder()
    elif TaskKeyword.TIME_SERIES in task_keywords:  # to review
        task_name = 'TIME_SERIES_CLASSIFICATION'
        builder = TimeseriesClassificationBuilder()
    elif TaskKeyword.VERTEX_NOMINATION in task_keywords or TaskKeyword.VERTEX_CLASSIFICATION in task_keywords:
        task_name = 'VERTEX_NOMINATION'
        builder = VertexNominationBuilder()
    elif 'text' in data_types and (TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'TEXT_' + task_name
        builder = BaseBuilder()
    elif 'image' in data_types and (TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'IMAGE_' + task_name
        builder = BaseBuilder()
    elif 'audio' in data_types and (TaskKeyword.REGRESSION in task_keywords or TaskKeyword.CLASSIFICATION in task_keywords):
        task_name = 'AUDIO_' + task_name
        builder = AudioBuilder()
    elif TaskKeyword.CLASSIFICATION in task_keywords or TaskKeyword.REGRESSION in task_keywords:
        builder = BaseBuilder()
    else:
        logger.warning('Task %s doesnt exist in the grammar, using default NA_TASK' % task)
        task_name = 'NA'
        builder = BaseBuilder()

    def create_input(selected_primitves, task_name):
        metafeatures_extractor = ComputeMetafeatures(dataset, targets, features, DBSession)
        input['GRAMMAR'] = format_grammar(task_name + '_TASK', selected_primitves)
        input['PROBLEM'] = task_name
        input['DATA_TYPE'] = 'TABULAR'
        input['METRIC'] = metrics[0]['metric'].name
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

    input_all = create_input(ALL_PRIMITIVES, task_name)
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
