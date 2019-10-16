import signal
import json
import os
import sys
import operator
import logging
import multiprocessing

# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.primitive_loader import D3MPrimitiveLoader
from d3m_ta2_nyu.grammar_loader import format_grammar
from alphaAutoMLEdit.Coach import Coach
from alphaAutoMLEdit.pipeline.PipelineGame import PipelineGame
from alphaAutoMLEdit.pipeline.NNet import NNetWrapper
from .d3mpipeline_generator import D3MPipelineGenerator, D3MPipelineGenerator_TimeseriesClass, D3MPipelineGenerator_TimeseriesFore
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
                          'TIME_SERIES_FORECASTING': 3,
                          'CLUSTERING': 4,
                          'TIME_SERIES_CLASSIFICATION': 5},

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

process_sklearn = None


def generate_by_templates(task, dataset, search_results, pipeline_template, metrics, problem, targets, features,
                          timeout, msg_queue, DBSession):
    logger.info("Creating pipelines from templates...")

    if task in ['GRAPH_MATCHING', 'LINK_PREDICTION', 'VERTEX_NOMINATION', 'OBJECT_DETECTION', 'CLUSTERING',
                'SEMISUPERVISED_CLASSIFICATION']:
        template_name = 'CLASSIFICATION'
    elif task in ['TIME_SERIES_FORECASTING', 'COLLABORATIVE_FILTERING']:
        template_name = 'REGRESSION'
    else:
        template_name = task
    if 'TA2_DEBUG_BE_FAST' in os.environ:
        template_name = 'DEBUG_' + task

    # No Augmentation
    templates = D3MPipelineGenerator.TEMPLATES.get(template_name, [])
    for imputer, classifier in templates:
        pipeline_id = D3MPipelineGenerator.make_template(imputer, classifier, dataset, pipeline_template, targets,
                                                         features, DBSession=DBSession)

        send(msg_queue, pipeline_id)

    # Augmentation
    if search_results and len(search_results) > 0:
        for search_result in search_results:
            templates = D3MPipelineGenerator.TEMPLATES_AUGMENTATION.get(template_name, [])
            for datamart, imputer, classifier in templates:
                pipeline_id = D3MPipelineGenerator.make_template_augment(datamart, imputer, classifier, dataset,
                                                                        pipeline_template, targets, features,
                                                                        search_result, DBSession=DBSession)

                send(msg_queue, pipeline_id)


def send(msg_queue, pipeline_id):
    msg_queue.send(('eval', pipeline_id))
    return msg_queue.recv()

@database.with_sessionmaker
def generate(task, dataset, search_results, pipeline_template, metrics, problem, targets, features, timeout, msg_queue, DBSession):
    #generate_by_templates(task, dataset, search_results, pipeline_template, metrics, problem, targets, features, timeout, msg_queue, DBSession)

    import time
    start = time.time()
    # FIXME: don't use 'problem' argument
    compute_metafeatures = ComputeMetafeatures(dataset, targets, features, DBSession)
    obj = D3MPipelineGenerator()

    def eval_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = obj.make_pipeline_from_strings(strings, origin, dataset, search_results,
                                                                      pipeline_template, targets, features,
                                                                      DBSession=DBSession)

        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_text_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_text_pipeline_from_strings(strings, origin, dataset, targets, features,
                                                                           DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_image_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_image_pipeline_from_strings(strings, origin, dataset, targets, features,
                                                                            DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_audio_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_audio_pipeline_from_strings(origin, dataset, targets, features,
                                                                            DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_object_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_objectdetection_pipeline_from_strings(origin, dataset, targets,
                                                                                      features, DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_graphmatch_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_graphmatching_pipeline_from_strings(origin, dataset, targets, features,
                                                                                    DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_communitydetection_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_communitydetection_pipeline_from_strings(origin, dataset, targets,
                                                                                         features, DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_linkprediction_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_linkprediction_pipeline_from_strings(origin, dataset, targets, features,
                                                                                     DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_vertexnomination_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_vertexnomination_pipeline_from_strings(origin, dataset, targets,
                                                                                       features, DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_semisupervised_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_semisupervised_pipeline_from_strings(origin, dataset, targets,
                                                                                     features, DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_collaborativefiltering_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = D3MPipelineGenerator.make_collaborativefiltering_pipeline_from_strings(origin, dataset, targets,
                                                                                        features, DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    dataset_path = os.path.dirname(dataset[7:])
    f = open(os.path.join(dataset_path, 'datasetDoc.json'))
    dataset_doc = json.load(f)
    data_resources = dataset_doc['dataResources']
    data_types = []
    for data_res in data_resources:
        data_types.append(data_res['resType'])

    function_name = eval_pipeline

    if 'COLLABORATIVE_FILTERING' in task:
        eval_collaborativefiltering_pipeline('ALPHAD3M')
        return
    elif 'OBJECT_DETECTION' in task:
        eval_object_pipeline('ALPHAD3M')
        return
    elif 'GRAPH_MATCHING' in task:
        eval_graphmatch_pipeline('ALPHAD3M')
        return
    elif 'SEMISUPERVISED_CLASSIFICATION' in task:
        eval_semisupervised_pipeline('ALPHAD3M')
        return
    elif 'COMMUNITY_DETECTION' in task:
        eval_communitydetection_pipeline('ALPHAD3M')
        return
    elif 'LINK_PREDICTION' in task:
        eval_linkprediction_pipeline('ALPHAD3M')
        return
    elif 'VERTEX_NOMINATION' in task or 'VERTEX_CLASSIFICATION' in task:
        eval_vertexnomination_pipeline('ALPHAD3M')
        return
    elif 'TIME_SERIES_FORECASTING' in task:
        obj = D3MPipelineGenerator_TimeseriesFore()
    elif 'timeseries' in data_types and 'CLASSIFICATION' in task:
        task = 'TIME_SERIES_CLASSIFICATION'
        obj = D3MPipelineGenerator_TimeseriesClass()
    elif 'image' in data_types and ('REGRESSION' in task or 'CLASSIFICATION' in task):
        function_name = eval_image_pipeline
    elif 'text' in data_types and ('REGRESSION' in task or 'CLASSIFICATION' in task):
        function_name = eval_text_pipeline
    elif 'audio' in data_types:
        eval_audio_pipeline('ALPHAD3M')
        return
    else:
        logger.warning('Task %s doesnt exist in the grammar, using default NA_TASK' % task)
        task = 'NA_TASK'

    def create_input(selected_primitves):
        task_name = task + '_TASK'
        GRAMMAR = format_grammar(task_name, selected_primitves)
        input['GRAMMAR'] = GRAMMAR
        input['PROBLEM'] = task
        input['DATA_TYPE'] = 'TABULAR'
        input['METRIC'] = metrics[0]['metric']
        input['DATASET_METAFEATURES'] = compute_metafeatures.compute_metafeatures('AlphaD3M_compute_metafeatures')
        input['DATASET'] = dataset_doc['about']['datasetName']
        input['ARGS']['stepsfile'] = os.path.join('/output', input['DATASET'] + '_pipeline_steps.txt')

        return input

    def record_bestpipeline(dataset):
        end = time.time()

        eval_dict = game.evaluations
        eval_times = game.eval_times
        for key, value in eval_dict.items():
            if value == float('inf') and not 'error' in game.metric.lower():
                eval_dict[key] = 0
        evaluations = sorted(eval_dict.items(), key=operator.itemgetter(1))
        if 'error' not in game.metric.lower():
            evaluations.reverse()

        out_p = open(os.path.join('/output', input['DATASET'] + '_best_pipelines.txt'), 'a')
        out_p.write(
            dataset_doc['about']['datasetName'] + ' ' + evaluations[0][0] + ' ' + str(evaluations[0][1]) + ' ' + str(
                game.steps) + ' ' + str((eval_times[evaluations[0][0]] - start) / 60.0) + ' ' + str(
                (end - start) / 60.0) + '\n')

    global process_sklearn
    input_sklearn = create_input(SKLEARN_PRIMITIVES)
    timeout_sklearn = int(timeout * 0.4)

    def run_sklearn_primitives():
        logger.info('Starting evaluation Scikit-learn primitives, timeout is %s', timeout_sklearn)
        game = PipelineGame(input_sklearn, function_name)
        nnet = NNetWrapper(game)
        c = Coach(game, nnet, input_sklearn['ARGS'])
        c.learn()

    def signal_handler(signal, frame):
        logger.info('Receiving SIGTERM signal')
        #record_bestpipeline(input['DATASET'])
        if process_sklearn.is_alive():
            process_sklearn.terminate()
        sys.exit(0)
    # TODO Not use multiprocessing to prioritize sklearn primitives
    signal.signal(signal.SIGTERM, signal_handler)

    process_sklearn = multiprocessing.Process(target=run_sklearn_primitives)
    process_sklearn.daemon = True
    process_sklearn.start()
    process_sklearn.join(timeout_sklearn)

    if process_sklearn.is_alive():
        process_sklearn.terminate()
        logger.info('Finished evaluation Scikit-learn primitives')

    input_all = create_input(ALL_PRIMITIVES)
    game = PipelineGame(input_all, function_name)
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

