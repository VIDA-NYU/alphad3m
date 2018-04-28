import os
import operator

# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'

from d3m_ta2_nyu.workflow import database

from .Coach import Coach
from .pipeline.PipelineGame import PipelineGame
from .pipeline.pytorch.NNet import NNetWrapper


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
    'load_folder_file': ('./temp/', 'best.pth.tar')
}


def make_pipeline_from_strings(strings, origin, dataset, DBSession):
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

    def connect(from_module, to_module,
                from_output='data', to_input='data'):
        db.add(database.PipelineConnection(pipeline=pipeline,
                                           from_module=from_module,
                                           to_module=to_module,
                                           from_output_name=from_output,
                                           to_input_name=to_input))

    try:
        data = make_module('data', '0.0', 'data')
        targets = make_module('data', '0.0', 'targets')

        # Assuming a simple linear pipeline
        imputer_name, encoder_name, classifier_name = strings

        # This will use sklearn directly, and others through the TA1 interface
        def make_primitive(name):
            if name.startswith('sklearn.'):
                return make_module('sklearn-builtin', '0.0', name)
            else:
                return make_module('primitives', '0.0', name)

        imputer = make_primitive(imputer_name)
        encoder = make_primitive(encoder_name)
        classifier = make_primitive(classifier_name)

        connect(data, imputer)
        connect(imputer, encoder)
        connect(encoder, classifier)
        connect(targets, classifier, 'targets', 'targets')

        db.add(pipeline)
        db.commit()
        return pipeline.id
    finally:
        db.close()


@database.with_sessionmaker
def generate(task, dataset, metrics, problem, msg_queue, DBSession):
    def eval_pipeline(strings, origin):
        # Create the pipeline in the database
        pipeline_id = make_pipeline_from_strings(strings, origin, dataset,
                                                 DBSession)

        # Evaluate the pipeline
        msg_queue.send(('eval', pipeline_id))
        return msg_queue.recv()

    args = dict(ARGS)
    args['dataset_path'] = dataset
    args['problem_path'] = problem
    args['metric'] = problem
    game = PipelineGame(3, args, eval_pipeline)
    nnet = NNetWrapper(game)

    if args.get('load_model'):
        model_file = os.path.join(args.get('load_folder_file')[0],
                                  args.get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(args.get('load_folder_file')[0],
                                 args.get('load_folder_file')[1])

    c = Coach(game, nnet, args)
    c.learn()


def main():
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
    with open('pipelines.txt') as f:
        pipelines_list = [line.strip() for line in f.readlines()]
        for pipeline in pipelines_list:
            fields = pipeline.split(' ')
            pipelines[fields[0]] = fields[1].split(',')
    datasets_path = '/Users/yamuna/D3M/data'
    dataset_names = pipelines.keys()
    out_p = open('best_pipelines.txt', 'w')
    for dataset in dataset_names:
        dataset = 'LL0_21_car'
        print(dataset)
        out_str = dataset + ', , \n'
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

        def eval_pipeline(strings, origin):
            # Create the pipeline in the database
            pipeline_id = make_pipeline_from_strings(strings, origin, dataset_name,
                                                 ta2.DBSession)

            # Evaluate the pipeline
            return ta2.run_pipeline(session_id, pipeline_id)
        pipeline = [p_enum[primitive] for primitive in pipelines[dataset]]
        print(pipeline)
        args = dict(ARGS)
        args['dataset_path'] = dataset_name
        args['problem_path'] = problem_name
        args['metric'] = problem_name
        game = PipelineGame(args, pipeline, eval_pipeline)
        nnet = NNetWrapper(game)

        if args.get('load_model'):
           model_file = os.path.join(args.get('load_folder_file')[0],
                                  args.get('load_folder_file')[1])
           if os.path.isfile(model_file):
                nnet.load_checkpoint(args.get('load_folder_file')[0],
                                 args.get('load_folder_file')[1])

        c = Coach(game, nnet, args)
        c.learn()
        evaluations = sorted(game.evaluations.items(), key=operator.itemgetter(1)).values()
        out_p.write(dataset+' '+evaluations[0] + ' ' + evaluations[1])
        break

if __name__ == '__main__':
    main()
