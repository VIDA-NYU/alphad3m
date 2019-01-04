import signal
import json
import os
import sys
import operator
import pickle
import logging

# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'

from d3m_ta2_nyu.workflow import database
from d3m_ta2_nyu.multiprocessing import Receiver
from d3m_ta2_nyu.d3m_primitives import D3MPrimitives
from d3m.container import Dataset

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
    import pickle
    from pathlib import Path
    installed_primitives_file = '/output/installed_primitives.pkl'
    installed_primitives_file_path = Path(installed_primitives_file)
    if installed_primitives_file_path.is_file():
        fp =  open(installed_primitives_file, 'rb')
        logger.info('Loading primitives from file')
        return pickle.load(fp)
    else:
        fp =  open(installed_primitives_file, 'wb')
        installed_primitives = D3MPrimitives.get_primitives_dict()
        pickle.dump(installed_primitives, fp)
        logger.info('Loading primitives from D3M index')
        return installed_primitives

PRIMITIVES = getPrimitives()

GRAMMAR = {
    'NON_TERMINALS': {
        'DATA_CLEANING':1,
        'DATA_TRANSFORMATION':2,
        'ESTIMATORS':3
        #'T': 4
    },
    'START': 'S->DATA_CLEANING DATA_TRANSFORMATION ESTIMATORS'
}

def getTerminals(non_terminals, primitives, task):
    terminals = {}
    count = len(GRAMMAR['NON_TERMINALS'])+1
    for non_terminal in non_terminals:
        if non_terminal == 'T':
            continue
        if non_terminal == 'ESTIMATORS':
            non_terminal = task.upper()
        for terminal in PRIMITIVES[non_terminal]:
            terminals[terminal] = count
            count += 1
    terminals['E'] = 0        
    return terminals



def getRules(non_terminals, primitives, task):
    # rules = {'T->DATA_CLEANING T DATA_TRANSFORMATION':1,
    #          'T->E':2
    #          }
    # rules_lookup = {'T': rules.keys()}

    rules = {}
    rules_lookup = {}
    count = len(rules)+1
    for non_terminal in non_terminals:
        rules_lookup[non_terminal] = []
        if non_terminal == 'ESTIMATORS':
            terminals = primitives[task.upper()]
            for terminal in terminals:
                rule = non_terminal+'->'+terminal
                rules[rule]=count
                count += 1
                rules_lookup[non_terminal].append(rule)
            continue
        
        terminals = primitives[non_terminal]
        for terminal in terminals:
            rule = non_terminal+'->'+terminal+' ' +non_terminal 
            rules[rule]=count
            count += 1
            rules_lookup[non_terminal].append(rule)
            rule = non_terminal+'->'+terminal
            rules[rule]=count
            count += 1
            rules_lookup[non_terminal].append(rule)

        rule = non_terminal+'->E'
        rules[rule] = count
        count += 1
        rules_lookup[non_terminal].append(rule)
        
    return rules, rules_lookup


input = {
    
    'PROBLEM_TYPES': {'CLASSIFICATION': 1,
                     'REGRESSION': 2},
    
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
        'metafeatures_path': '/d3m/data/metafeatures'
    }
}



@database.with_sessionmaker
def generate(task, dataset, metrics, problem, targets, features, msg_queue, DBSession):
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
        pipeline_id = GenerateD3MPipelines.make_audio_pipeline_from_strings(strings, origin,
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

    def eval_image_pipeline(estimator, origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_image_pipeline_from_strings(estimator,
                                                                            origin,
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

    unsupported_problems = ["timeSeriesForecasting", "collaborativeFiltering"]

    if task in unsupported_problems:
        logger.error("%s Not Supported", task)
        sys.exit(148)
    
    if "text" in data_types:
        logger.error("Text Datatype Not Supported")
        sys.exit(148)

    if "audio" in data_types:
        primitives = ["d3m.primitives.bbn.time_series.ChannelAverager",
                          "d3m.primitives.bbn.time_series.SignalDither", "d3m.primitives.bbn.time_series.SignalFramer",  "d3m.primitives.bbn.time_series.SignalMFCC",
                          "d3m.primitives.bbn.time_series.UniformSegmentation", "d3m.primitives.bbn.time_series.SegmentCurveFitter", "d3m.primitives.bbn.time_series.ClusterCurveFittingKMeans",
                          "d3m.primitives.bbn.time_series.SignalFramer", "d3m.primitives.bbn.time_series.SequenceToBagOfTokens", "d3m.primitives.bbn.time_series.BBNTfidfTransformer",
                          "d3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier"]
        eval_audio_pipeline(primitives, "ALPHAD3M")

        primitives = ["d3m.primitives.bbn.time_series.ChannelAverager",
                          "d3m.primitives.bbn.time_series.SignalDither", "d3m.primitives.bbn.time_series.SignalFramer",  "d3m.primitives.bbn.time_series.SignalMFCC",
                          "d3m.primitives.bbn.time_series.IVectorExtractor", 
                          "d3m.primitives.bbn.sklearn_wrap.BBNMLPClassifier"]
        eval_audio_pipeline(primitives, "ALPHAD3M")
        return

    if "graph" in data_types:
        if "graphMatching" in task:
            eval_graphMatch_pipeline("ALPHAD3M")
            return
        elif "communityDetection" in task:
            eval_communityDetection_pipeline("ALPHAD3M")
            return
        elif "linkPrediction" in task:
            eval_linkprediction_pipeline("ALPHAD3M")
            return
        elif "vertexNomination" in task:
            eval_vertexnomination_pipeline("ALPHAD3M")
            return
        logger.error("%s Not Supported", task)
        sys.exit(148)


    estimator = 'd3m.primitives.sklearn_wrap.SKRandomForestClassifier'
    if "image" in data_types:
        if "regression" in task:
            estimator = 'd3m.primitives.sklearn_wrap.SKLasso'
            eval_image_pipeline(estimator, "ALPHAD3M")
            return
        logger.error("%s Not Supported", task)
        sys.exit(148)


    if "timeseries" in data_types:
        if "classification" in task:
            eval_timeseries_pipeline("ALPHAD3M")
            return
        logger.error("%s Not Supported", task)
        sys.exit(148)

    GRAMMAR['TERMINALS'] = getTerminals(GRAMMAR['NON_TERMINALS'], PRIMITIVES, task)
    r, l = getRules(GRAMMAR['NON_TERMINALS'], PRIMITIVES, task)
    GRAMMAR['RULES'] = r
    GRAMMAR['RULES_LOOKUP'] = l

    input['GRAMMAR'] = GRAMMAR
    input['PROBLEM'] = task
    input['DATA_TYPE'] = 'TABULAR'
    input['METRIC'] = metrics[0]
    input['DATASET_METAFEATURES'] = compute_metafeatures.compute_metafeatures('AlphaD3M_compute_metafeatures')
    input['DATASET'] = datasetDoc['about']['datasetName']
    game = PipelineGame(input, eval_pipeline)
    nnet = NNetWrapper(game)

    def record_bestpipeline():
        end = time.time()
        
        eval_dict = game.evaluations
        for key, value in eval_dict.items():
            if value == float('inf') and not 'error' in game.metric.lower():
                eval_dict[key] = 0
        evaluations = sorted(eval_dict.items(), key=operator.itemgetter(1))
        if not 'error' in game.metric.lower():
            evaluations.reverse()

        out_p = open(os.path.join('/output', 'best_pipelines.txt'), 'a')
        out_p.write(datasetDoc['about']['datasetName']+' '+evaluations[0][0] + ' ' + str(evaluations[0][1])+ ' ' + str(game.steps) + ' ' + str((end-start)/60.0) + '\n')
        
    def signal_handler(signal, frame):
        record_bestpipeline()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    
    if input['ARGS'].get('load_model'):
        model_file = os.path.join(input['ARGS'].get('load_folder_file')[0],
                                  input['ARGS'].get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(input['ARGS'].get('load_folder_file')[0],
                                 input['ARGS'].get('load_folder_file')[1])

    c = Coach(game, nnet, input['ARGS'])
    c.learn()
    
    record_bestpipeline()

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
