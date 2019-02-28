import signal
import json
import os
import sys
import operator
import pickle
import logging
import multiprocessing

# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.multiprocessing import Receiver
from d3m_ta2_nyu.d3m_primitives import D3MPrimitives
from alphaAutoMLEdit.Coach import Coach
from alphaAutoMLEdit.pipeline.PipelineGame import PipelineGame
from alphaAutoMLEdit.pipeline.pytorch.NNet import NNetWrapper
from .GenerateD3MPipelines import GenerateD3MPipelines
from d3m_ta2_nyu.metafeatures.dataset import ComputeMetafeatures

logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:TA2:%(name)s:%(message)s")

    
def getPrimitives():
    installed_primitives_file = '/output/installed_primitives.pkl'
    installed_primitives_file_path = Path(installed_primitives_file)
    sklearn_primitives = {}

    if installed_primitives_file_path.is_file():
        fp = open(installed_primitives_file, 'rb')
        logger.info('Loading primitives from file')
        all_primitives = pickle.load(fp)
    else:
        all_primitives = D3MPrimitives.get_primitives_dict()
        fp = open(installed_primitives_file, 'wb')
        pickle.dump(all_primitives, fp)
        logger.info('Loading primitives from D3M index')

    for group in list(all_primitives.keys()):
        sklearn_primitives[group] = {}
        for primitive in list(all_primitives[group].keys()):
            if primitive.startswith('d3m.primitives.sklearn_wrap.'):
                sklearn_primitives[group][primitive] = all_primitives[group][primitive]
            if 'jhu_primitives' in primitive: # FIXME New version of jhu_primitives doesn't have the R's infinity loop error
                del all_primitives[group][primitive]

    return all_primitives, sklearn_primitives


ALL_PRIMITIVES, SKLEARN_PRIMITIVES = getPrimitives()

GRAMMAR = {
    'NON_TERMINALS': {
        'S': 1,
        'DATA_CLEANING':2,
        'DATA_TRANSFORMATION':3,
        'ESTIMATORS':4
    },
    'START': 'S->S'
}


def getTerminals(non_terminals, primitives, task):
    terminals = {}
    count = len(GRAMMAR['NON_TERMINALS'])+1
    for non_terminal in non_terminals:
        if non_terminal == 'S':
            continue
        if non_terminal == 'ESTIMATORS':
            non_terminal = task.upper()
        for terminal in primitives[non_terminal]:
            terminals[terminal] = count
            count += 1
    terminals['E'] = 0

    return terminals


def getRules(non_terminals, primitives, task):
    rules = { 'S->ESTIMATORS':1,
              'S->DATA_CLEANING ESTIMATORS':2,
              'S->DATA_TRANSFORMATION ESTIMATORS':3,
              'S->DATA_CLEANING DATA_TRANSFORMATION ESTIMATORS': 4
    }
    
    rules_lookup = {'S': list(rules.keys())}
    count = len(rules)+1
    for non_terminal in non_terminals:
        if non_terminal == 'S':
            continue

        rules_lookup[non_terminal] = []
        if non_terminal == 'ESTIMATORS':
            terminals = primitives[task.upper()]
            for terminal in terminals:
                rule = non_terminal+'->'+terminal
                rules[rule] = count
                count += 1
                rules_lookup[non_terminal].append(rule)
            continue

        terminals = primitives[non_terminal]

        for terminal in terminals:
            rule = non_terminal+'->'+terminal+' ' +non_terminal 
            rules[rule] = count
            count += 1
            rules_lookup[non_terminal].append(rule)
            rule = non_terminal+'->'+terminal
            rules[rule] = count
            count += 1
            rules_lookup[non_terminal].append(rule)

        rule = non_terminal+'->E'
        rules[rule] = count
        count += 1
        rules_lookup[non_terminal].append(rule)
        
    return rules, rules_lookup


input = {
        'PROBLEM_TYPES': {'CLASSIFICATION': 1,
                          'REGRESSION': 2,
                          'TIME_SERIES_FORECASTING': 3,
                          'CLUSTERING': 4},

        'DATA_TYPES': {'TABULAR': 1,
                       'GRAPH': 2,
                       'IMAGE': 3},

        'PIPELINE_SIZE': 4,

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


@database.with_sessionmaker
def generate(task, dataset, metrics, problem, targets, features, timeout, msg_queue, DBSession):
    import time
    start = time.time()
    # FIXME: don't use 'problem' argument
    compute_metafeatures = ComputeMetafeatures(dataset, targets, features, DBSession)

    def eval_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_pipeline_from_strings(strings, origin,
                                                                      dataset, targets, features,
                                                                      DBSession=DBSession)

        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_audio_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_audio_pipeline_from_strings(origin,
                                                                            dataset,
                                                                            targets, features, DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_graphMatch_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_graphMatching_pipeline_from_strings(origin,
                                                                                    dataset,
                                                                                    targets, features,
                                                                                    DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_communityDetection_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_communityDetection_pipeline_from_strings(origin,
                                                                                         dataset,
                                                                                         targets, features,
                                                                                         DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_linkprediction_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_linkprediction_pipeline_from_strings(origin,
                                                                                     dataset,
                                                                                     targets, features,
                                                                                     DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_vertexnomination_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_vertexnomination_pipeline_from_strings(origin,
                                                                                     dataset,
                                                                                     targets, features,
                                                                                     DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_image_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_image_pipeline_from_strings(origin,
                                                                            dataset,
                                                                            targets, features,
                                                                            DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    def eval_timeseries_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_timeseries_pipeline_from_strings(origin,
                                                                            dataset,
                                                                            targets, features,
                                                                            DBSession=DBSession)
        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    
    dataset_path = os.path.dirname(dataset[7:])
    f = open(os.path.join(dataset_path, 'datasetDoc.json'))
    datasetDoc = json.load(f)
    data_resources = datasetDoc["dataResources"]
    data_types = []
    for data_res in data_resources:
        data_types.append(data_res["resType"])

    unsupported_problems = ["TIME_SERIES_FORECASTING", "COLLABORATIVE_FILTERING", 'OBJECT_DETECTION']
    print('>>>>>>', task, data_types)
    if task in unsupported_problems:
        logger.error("%s Not Supported", task)
        sys.exit(148)
    
    if "text" in data_types:
        logger.error("Text Datatype Not Supported")
        sys.exit(148)

    if "audio" in data_types:
        eval_audio_pipeline("ALPHAD3M")
        return

    if "graph" in data_types:
        if "GRAPH_MATCHING" in task:
            eval_graphMatch_pipeline("ALPHAD3M")
            return
        elif "COMMUNITY_DETECTION" in task:
            eval_communityDetection_pipeline("ALPHAD3M")
            return
        elif "LINK_PREDICTION" in task:
            eval_linkprediction_pipeline("ALPHAD3M")
            return
        elif "VERTEX_NOMINATION" in task:
            eval_vertexnomination_pipeline("ALPHAD3M")
            return
        logger.error("%s Not Supported", task)
        sys.exit(148)

    if "image" in data_types:
        if "REGRESSION" in task:
            eval_image_pipeline("ALPHAD3M")
            return
        logger.error("%s Not Supported", task)
        sys.exit(148)

    if "timeseries" in data_types:
        if "CLASSIFICATION" in task:
            eval_timeseries_pipeline("ALPHAD3M")
            return
        logger.error("%s Not Supported", task)
        sys.exit(148)

    def create_input(selected_primitves):
        GRAMMAR['TERMINALS'] = getTerminals(GRAMMAR['NON_TERMINALS'], selected_primitves, task)
        r, l = getRules(GRAMMAR['NON_TERMINALS'], selected_primitves, task)

        GRAMMAR['RULES'] = r
        GRAMMAR['RULES_LOOKUP'] = l

        input['GRAMMAR'] = GRAMMAR
        input['PROBLEM'] = task
        input['DATA_TYPE'] = 'TABULAR'
        input['METRIC'] = metrics[0]
        input['DATASET_METAFEATURES'] = compute_metafeatures.compute_metafeatures('AlphaD3M_compute_metafeatures')
        input['DATASET'] = datasetDoc['about']['datasetName']
        input['ARGS']['stepsfile'] = os.path.join('/output', input['DATASET']+'_pipeline_steps.txt')

        return input

    input_sklearn = create_input(SKLEARN_PRIMITIVES)
    timeout_sklearn = int(timeout * 0.4)

    def run_sklearn_primitives():
        logger.info('Starting evaluation Scikit-learn primitives, timeout is %s', timeout_sklearn)
        game = PipelineGame(input_sklearn, eval_pipeline)
        nnet = NNetWrapper(game)
        c = Coach(game, nnet, input_sklearn['ARGS'])
        c.learn()

    process_sklearn = multiprocessing.Process(target=run_sklearn_primitives)
    process_sklearn.daemon = True
    process_sklearn.start()
    process_sklearn.join(timeout_sklearn)

    if process_sklearn.is_alive():
        process_sklearn.terminate()
        logger.info('Finished evaluation Scikit-learn primitives')

    input_all = create_input(ALL_PRIMITIVES)

    game = PipelineGame(input_all, eval_pipeline)
    nnet = NNetWrapper(game)

    def signal_handler(signal, frame):
        record_bestpipeline(input['DATASET'])
        sys.exit(0)

    def record_bestpipeline(dataset):
        end = time.time()

        eval_dict = game.evaluations
        eval_times = game.eval_times
        for key, value in eval_dict.items():
            if value == float('inf') and not 'error' in game.metric.lower():
                eval_dict[key] = 0
        evaluations = sorted(eval_dict.items(), key=operator.itemgetter(1))
        if not 'error' in game.metric.lower():
            evaluations.reverse()

        out_p = open(os.path.join('/output', input['DATASET'] + '_best_pipelines.txt'), 'a')
        out_p.write(
            datasetDoc['about']['datasetName'] + ' ' + evaluations[0][0] + ' ' + str(evaluations[0][1]) + ' ' + str(
                game.steps) + ' ' + str((eval_times[evaluations[0][0]] - start) / 60.0) + ' ' + str(
                (end - start) / 60.0) + '\n')

    signal.signal(signal.SIGTERM, signal_handler)

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


def main(dataset, problem_path, output_path):
    setup_logging()

    import time
    start = time.time()
    import tempfile
    from d3m_ta2_nyu.ta2 import D3mTa2

    pipelines = {}
    if os.path.isfile('pipelines.txt'):
        with open('pipelines.txt') as f:
            pipelines_list = [line.strip() for line in f.readlines()]
            for pipeline in pipelines_list:
                fields = pipeline.split(' ')
                pipelines[fields[0]] = fields[1].split(',')

    problem = {}
    with open(os.path.join(problem_path, 'problemDoc.json')) as fp:
        problem = json.load(fp)
    task = problem['about']['taskType']
    
    metrics = []
    for metric in problem['inputs']['performanceMetrics']:
        metrics.append(metric['metric'])

    storage = tempfile.mkdtemp(prefix='d3m_pipeline_eval_')
    ta2 = D3mTa2(storage_root=storage,
        pipelines_considered_root=os.path.join(storage, 'pipelines_considered'),
        executables_root=os.path.join(storage, 'executables'))

    session_id = ta2.new_session(args['problem'])
    session = ta2.sessions[session_id]
    msg_queue = Receiver()

    generate(task, dataset, metrics, problem, session.targets, session.features, msg_queue, ta2.DBSession)


if __name__ == '__main__':
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    main(sys.argv[1], sys.argv[2], output_path)
