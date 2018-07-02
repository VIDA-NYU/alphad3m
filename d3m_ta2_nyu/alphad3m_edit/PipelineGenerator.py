import json
import os
import sys
import operator
import pickle

# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'

from d3m_ta2_nyu.workflow import database
from d3m.container import Dataset

from .Coach import Coach
from .pipeline.PipelineGame import PipelineGame
from .pipeline.pytorch.NNet import NNetWrapper
from d3m_ta2_nyu.metafeatures.dataset import ComputeMetafeatures

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


def make_pipeline_from_strings(primitives, origin, dataset, targets=None, features=None, DBSession=None):
    db = DBSession()


    pipeline = database.Pipeline(
        origin=origin,
        dataset=dataset)

    def make_module(package, version, name):
        pipeline_module = database.PipelineModule(
            pipeline=pipeline,
            package=package, version=version, name=name)
        db.add(pipeline_module)
        return pipeline_module

    def make_data_module(name):
        return make_module('data', '0.0', name)

    def make_primitive_module(name):
        if name[0] == '.':
            name = 'd3m.primitives' + name
        return make_module('d3m', '2018.4.18', name)

    def connect(from_module, to_module,
                from_output='produce', to_input='inputs'):
        db.add(database.PipelineConnection(pipeline=pipeline,
                                           from_module=from_module,
                                           to_module=to_module,
                                           from_output_name=from_output,
                                           to_input_name=to_input))

    try:
        #                data
        #                  |
        #              Denormalize
        #                  |
        #           DatasetToDataframe
        #                  |
        #             ColumnParser
        #                /     \
        # ExtractAttributes  ExtractTargets
        #         |               |
        #     CastToType      CastToType
        #         |               |
        #     [imputer]           |
        #            \            /
        #             [classifier]
        # TODO: Use pipeline input for this
        # TODO: Have execution set metadata from problem, don't hardcode
        input_data = make_data_module('dataset')
        db.add(database.PipelineParameter(
            pipeline=pipeline, module=input_data,
            name='targets', value=pickle.dumps(targets),
        ))
        db.add(database.PipelineParameter(
            pipeline=pipeline, module=input_data,
            name='features', value=pickle.dumps(features),
        ))

        step0 = make_primitive_module('.datasets.Denormalize')
        connect(input_data, step0, from_output='dataset')

        step1 = make_primitive_module('.datasets.DatasetToDataFrame')
        connect(step0, step1)

        step2 = make_primitive_module('.data.ExtractAttributes')
        connect(step1, step2)

        step3 = make_primitive_module('.data.ColumnParser')
        connect(step2, step3)

        step4 = make_primitive_module('.data.CastToType')
        connect(step3, step4)

        step = prev_step = step4
        preprocessors = []
        if len(primitives) > 1:
            preprocessors = primitives[0:len(primitives)-2]
        classifier = primitives[len(primitives)-1]
        for preprocessor in preprocessors:
            step = make_primitive_module(preprocessor)
            connect(prev_step, step)
            prev_step = step

        step6 = make_primitive_module('.data.ExtractTargets')
        connect(step1, step6)

        step7 = make_primitive_module('.data.CastToType')
        connect(step6, step7)

        step8 = make_primitive_module(classifier)
        connect(step, step8)
        connect(step7, step8, to_input='outputs')

        db.add(pipeline)
        db.commit()
        print('PIPELINE ID: ', pipeline.id)
        return pipeline.id
    finally:
        db.close()


@database.with_sessionmaker
def generate(task, dataset, metrics, problem, targets, features, msg_queue, DBSession):
    # FIXME: don't use 'problem' argument
    compute_metafeatures = ComputeMetafeatures(dataset, targets, features, DBSession)
    def eval_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = make_pipeline_from_strings(strings, origin,
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


def main(dataset, output_path):
    import time
    start = time.time()
    import tempfile
    from d3m_ta2_nyu.main import setup_logging
    from d3m_ta2_nyu.ta2 import D3mTa2

    p_enum = {
                 'dsbox.datapreprocessing.cleaner.Encoder': 1,
                 'dsbox.datapreprocessing.cleaner.KNNImputation': 2,
                 'sklearn.linear_model.SGDClassifier':17,
            	 'sklearn.linear_model.SGDRegressor': 35     
             }
    pipelines_list = []
    pipelines = {}
    if os.path.isfile('pipelines.txt'):
        with open('pipelines.txt') as f:
            pipelines_list = [line.strip() for line in f.readlines()]
            for pipeline in pipelines_list:
                fields = pipeline.split(' ')
                pipelines[fields[0]] = fields[1].split(',')
    datasets_path = '/d3m/data'
    dataset_names = pipelines.keys()
    args = dict(ARGS)

    print(dataset)

    if 'LL0' in dataset:
       dataset_path = os.path.join(datasets_path, 'LL0', dataset)
    else:
       dataset_path = os.path.join(datasets_path, 'LL1', dataset)

    dataset_name = os.path.join(dataset_path, dataset+'_dataset')
    problem_name = os.path.join(dataset_path, dataset+'_problem')

    storage = tempfile.mkdtemp(prefix='d3m_pipeline_eval_')
    ta2 = D3mTa2(storage_root=storage,
             logs_root=os.path.join(storage, 'logs'),
             executables_root=os.path.join(storage, 'executables'))
    setup_logging()
    session_id = ta2.new_session(problem_name)
    session = ta2.sessions[session_id]
    compute_metafeatures = ComputeMetafeatures('file://'+dataset_name+'/datasetDoc.json', session.targets, session.features, ta2.DBSession)

    def eval_pipeline(strings, origin):
        print('CALLING EVAL PIPELINE')
        # Create the pipeline in the database
        pipeline_id = make_pipeline_from_strings(strings, origin, 'file://'+dataset_name+'/datasetDoc.json',
                                                 session.targets, session.features, ta2.DBSession)
        # Evaluate the pipeline
        return ta2.run_pipeline(session_id, pipeline_id)
    pipeline = None
    if pipelines:
        pipeline = [p_enum[primitive] for primitive in pipelines[dataset]]
    print(pipeline)
    args['dataset'] = dataset
    args['dataset_path'] = dataset_name
    with open(os.path.join(problem_name, 'problemDoc.json')) as fp:
        args['problem'] = json.load(fp)
    args['metric'] = problem_name
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
    out_p = open(os.path.join(output_path, dataset+'_best_pipelines.txt'), 'w')
    out_p.write(dataset+' '+evaluations[0][0] + ' ' + str(evaluations[0][1])+ ' ' + str(game.steps) + ' ' + str((end-start)/60.0) + '\n')

if __name__ == '__main__':
    output_path = '.'
    if len(sys.argv) > 2:
       output_path = sys.argv[2]
    main(sys.argv[1], output_path)
