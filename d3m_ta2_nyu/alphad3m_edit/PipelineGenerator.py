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

estimators = {
    'CLASSIFICATION': {
        'd3m.primitives.sklearn_wrap.SKRandomForestClassifier':3,
        'd3m.primitives.sklearn_wrap.SKDecisionTreeClassifier':4,
        'd3m.primitives.featuretools_ta1.SKRFERandomForestClassifier':5,
        'd3m.primitives.classifier.RandomForest':6,
        'd3m.primitives.sklearn_wrap.SKMultinomialNB':7,
        'd3m.primitives.common_primitives.BayesianLogisticRegression': 9,
        'd3m.primitives.dsbox.CorexSupervised': 11,
        'd3m.primitives.lupi_svm': 13,
        'd3m.primitives.realML.TensorMachinesBinaryClassification': 14,
        'd3m.primitives.sklearn_wrap.SKPassiveAggressiveClassifier': 15,
        'd3m.primitives.sklearn_wrap.SKQuadraticDiscriminantAnalysis': 16,
        'd3m.primitives.sklearn_wrap.SKSGDClassifier': 18,
        'd3m.primitives.sklearn_wrap.SKSVC': 19
    },
    'REGRESSION': {
        'd3m.primitives.cmu.autonlab.find_projections.SearchNumeric': 20,
        'd3m.primitives.cmu.autonlab.find_projections.SearchHybridNumeric': 21,
        'd3m.primitives.featuretools_ta1.SKRFERandomForestRegressor':22,
        'd3m.primitives.sklearn_wrap.SKARDRegression': 25,
        'd3m.primitives.sklearn_wrap.SKDecisionTreeRegressor': 26,
        'd3m.primitives.sklearn_wrap.SKExtraTreesRegressor': 27,
        'd3m.primitives.sklearn_wrap.SKGaussianProcessRegressor': 28,
        'd3m.primitives.sklearn_wrap.SKLars': 31,
        'd3m.primitives.sklearn_wrap.SKLasso': 32,
        'd3m.primitives.sklearn_wrap.SKLassoCV': 33,
        'd3m.primitives.sklearn_wrap.SKLinearSVR': 34,
        'd3m.primitives.sklearn_wrap.SKPassiveAggressiveRegressor': 35,
        'd3m.primitives.sklearn_wrap.SKSGDRegressor': 36,
        'd3m.primitives.sklearn_wrap.SKRidge':39
    }
}


input = {

    'PIPELINE_SIZES': {
        'DATA_CLEANING': 1,
        'DATA_PREPROCESSING': 1,
        'ESTIMATORS': 1
    },
    
    'PROBLEM_TYPES': {'CLASSIFICATION': 1,
                     'REGRESSION': 2},
    
    'DATA_TYPES': {'TABULAR': 1,
                  'GRAPH': 2,
                  'IMAGE': 3},
    
    'PRIMITIVES': {
        'DATA_CLEANING':{
            'd3m.primitives.dsbox.MeanImputation': 2
        },
        'DATA_PREPROCESSING': {
            'd3m.primitives.dsbox.Encoder': 1,
        }
    },

    'ARGS': {
        'numIters': 3,
        'numEps': 100,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 25,
        'arenaCompare': 40,
        'cpuct': 1,
        
        'checkpoint': './temp/',
        'load_model': False,
        'load_folder_file': ('./temp/', 'best.pth.tar'),
        'metafeatures_path': '/d3m/data/metafeatures'
    }
}

@database.with_sessionmaker
def generate(task, dataset, metrics, problem, targets, features, msg_queue, DBSession):
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

    input['PROBLEM'] = task
    input['DATA_TYPE'] = 'TABULAR'
    input['METRIC'] = metrics[0]
    input['DATASET_METAFEATURES'] = compute_metafeatures.compute_metafeatures('AlphaD3M_compute_metafeatures')
    input['PRIMITIVES']['ESTIMATORS'] = estimators[task]
    
    game = PipelineGame(None, input, eval_pipeline)
    nnet = NNetWrapper(game)

    if input['ARGS'].get('load_model'):
        model_file = os.path.join(input['ARGS'].get('load_folder_file')[0],
                                  input['ARGS'].get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(input['ARGS'].get('load_folder_file')[0],
                                 input['ARGS'].get('load_folder_file')[1])

    c = Coach(game, nnet, input['ARGS'])
    c.learn()


def main(dataset, problem_path, output_path):
    setup_logging()

    import time
    start = time.time()
    import tempfile
    from d3m_ta2_nyu.ta2 import D3mTa2

    p_enum = {
                 'dsbox.datapreprocessing.cleaner.Encoder': 1,
                 'dsbox.datapreprocessing.cleaner.KNNImputation': 2,
                 'sklearn.linear_model.SGDClassifier':17,
            	 'sklearn.linear_model.SGDRegressor': 35     
             }

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
    output_path = '.'
    if len(sys.argv) > 3:
       output_path = sys.argv[3]
    main(sys.argv[1], sys.argv[2], output_path)
