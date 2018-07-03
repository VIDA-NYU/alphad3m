import json
import os
import sys
import operator
import pickle
import logging

# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'

from d3m_ta2_nyu.workflow import database
from d3m.container import Dataset

from .Coach import Coach
from .pipeline.PipelineGame import PipelineGame
from .pipeline.pytorch.NNet import NNetWrapper
from .GenerateD3MPipelines import GenerateD3MPipelines

from d3m_ta2_nyu.metafeatures.dataset import ComputeMetafeatures

logger = logging.getLogger(__name__)

ARGS = {
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

    args = dict(ARGS)
    args['dataset'] = dataset.split('/')[-1].replace('_dataset','')
    assert dataset.startswith('file://')
    args['dataset_path'] = os.path.dirname(dataset[7:])
    args['problem'] = problem

    taskType = self.args['problem']['about']['taskType'].upper()

    game = PipelineGame(args, None, eval_pipeline, compute_metafeatures)
    nnet = NNetWrapper(game)

    if args.get('load_model'):
        model_file = os.path.join(args.get('load_folder_file')[0],
                                  args.get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(args.get('load_folder_file')[0],
                                 args.get('load_folder_file')[1])

    c = Coach(game, nnet, args)
    c.learn()


def main(dataset_uri, problem_path, output_path):
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
    args = dict(ARGS)

    storage = tempfile.mkdtemp(prefix='d3m_pipeline_eval_')
    ta2 = D3mTa2(storage_root=storage,
             logs_root=os.path.join(storage, 'logs'),
             executables_root=os.path.join(storage, 'executables'))

    session_id = ta2.new_session(problem_path)
    session = ta2.sessions[session_id]
    compute_metafeatures = ComputeMetafeatures(os.path.join(dataset_uri, 'datasetDoc.json'), session.targets, session.features, ta2.DBSession)

    def eval_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_pipeline_from_strings(strings, origin, os.path.join(dataset_uri, 'datasetDoc.json'),
                                                 session.targets, session.features, ta2.DBSession)
        # Evaluate the pipeline
        return ta2.run_pipeline(session_id, pipeline_id)

    def eval_audio_pipeline(origin):
        # Create the pipeline in the database
        pipeline_id = GenerateD3MPipelines.make_audio_pipeline_from_strings(origin,
                                                                      os.path.join(dataset_uri, 'datasetDoc.json'),
                                                                      session.targets, session.features, ta2.DBSession)
        # Evaluate the pipeline
        return ta2.run_pipeline(session_id, pipeline_id)

    pipeline = None
    if pipelines:
        pipeline = [p_enum[primitive] for primitive in pipelines[dataset]]
    logger.info(pipeline)
    args['dataset'] = dataset_uri.split('/')[-1].replace('_dataset', '')
    assert dataset_uri.startswith('file://')
    args['dataset_path'] = dataset_uri[7:]
    print(dataset_uri)
    print(args['dataset_path'])
    with open(os.path.join(problem_path, 'problemDoc.json')) as fp:
        args['problem'] = json.load(fp)

    f = open(os.path.join(args['dataset_path'], 'datasetDoc.json'))
    datasetDoc = json.load(f)
    data_resources = datasetDoc["dataResources"]
    data_types = []
    for data_res in data_resources:
        data_types.append(data_res["resType"])

    if "audio" in data_types:
        eval_audio_pipeline("ALPHAD3M")
        return

    game = PipelineGame(args, pipeline, eval_pipeline, compute_metafeatures)
    nnet = NNetWrapper(game)

    if args.get('load_model'):
       model_file = os.path.join(args.get('load_folder_file')[0],
                              args.get('load_folder_file')[1])
       if os.path.isfile(model_file):
            nnet.load_checkpoint(args.get('load_folder_file')[0],
                             args.get('load_folder_file')[1])

    c = Coach(game, nnet, args)
    c.learn()
    eval_dict = game.evaluations
    for key, value in eval_dict.items():
        if value == float('inf') and not 'error' in game.metric.lower():
           eval_dict[key] = 0
    evaluations = sorted(eval_dict.items(), key=operator.itemgetter(1))
    if not 'error' in game.metric.lower():
        evaluations.reverse()
    end = time.time()
    out_p = open(os.path.join(output_path, args['dataset']+'_best_pipelines.txt'), 'w')
    out_p.write(args['dataset']+' '+evaluations[0][0] + ' ' + str(evaluations[0][1])+ ' ' + str(game.steps) + ' ' + str((end-start)/60.0) + '\n')

if __name__ == '__main__':
    output_path = '.'
    if len(sys.argv) > 3:
       output_path = sys.argv[3]
    main(sys.argv[1], sys.argv[2], output_path)
