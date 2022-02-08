import os
import sys
import operator
import json
import logging
import pandas as pd
import numpy as np
from pprint import pprint

# Use a headless matplotlib backend
os.environ['MPLBACKEND'] = 'Agg'


from ..Coach import Coach
from ..pipeline.PipelineGame import PipelineGame
from alphaautoml.alphaAutoMLEdit.pipeline.NNet import NNetWrapper
from .ComputeMetafeatures import ComputeMetafeatures

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")


#************* INPUTS REQUIRED FOR PIPELINE GENERATION ******************

def getPrimitives():
    import pickle
    from pathlib import Path
    installed_primitives_file = '/home/ubuntu/temp/sklearn_primitives.pkl'
    installed_primitives_file_path = Path(installed_primitives_file)
    if installed_primitives_file_path.is_file():
        fp =  open(installed_primitives_file, 'rb')
        return pickle.load(fp)

PRIMITIVES = getPrimitives()

GRAMMAR = {
    'NON_TERMINALS': {
        'S':1,
        'DATA_CLEANING':2,
        'DATA_TRANSFORMATION':3,
        'ESTIMATORS':4,
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
        for terminal in PRIMITIVES[non_terminal]:
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

        if rules_lookup.get(non_terminal) is None:
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

    'PIPELINE_SIZE': 3,

    'ARGS': {
        'numIters': 25,
        'numEps': 5,
        'tempThreshold': 15,
        'updateThreshold': 0.6,
        'maxlenOfQueue': 200000,
        'numMCTSSims': 5,
        'arenaCompare': 40,
        'cpuct': 1,

        'checkpoint': '/home/ubuntu/temp/nn_models_test',
        'load_model': False,
        'load_folder_file': ('/home/ubuntu/temp/nn_models_test', 'best.pth.tar'),
        'metafeatures_path': '/d3m/data/metafeatures',
        'verbose': True
    }
}


#****************************************************************************

def impute_categorical(data):
    data = data.replace('?', np.nan)
    cat_cols = data.select_dtypes(include=object)
    #print('Cat Cols Data')
    #pprint(cat_cols)
    cat_col_names = cat_cols.columns.values
    print('Categorical Columns')
    pprint(cat_col_names)
    partial_data = data.drop(columns=cat_col_names)

    from sklearn_pandas import CategoricalImputer
    ci = CategoricalImputer()
    for col in cat_col_names:
        try:
            col_data=ci.fit_transform(cat_cols[col].values)
            partial_data = pd.concat([partial_data, pd.DataFrame(col_data, dtype=object)], axis=1)
            #pprint(partial_data)
        except:
            partial_data = pd.concat([partial_data, cat_cols[col]], axis=1)
    return partial_data

def extract_data(dataset, target_col):

    df = pd.read_csv(dataset)
    # Get the target as panda series
    target = df.loc[:, target_col]

    data = df.drop(columns=[target_col, 'd3mIndex'], errors='ignore')

    return data, target

def generate(dataset_path, dataset_name, problem, target_col, metric, input_args):

    X_full, y_full = extract_data(dataset_path, target_col)
    # Compute dataset metafeatures
    dataset_metafeatures = list(ComputeMetafeatures().compute_metafeatures(X_full, y_full).values())

    X_full = impute_categorical(X_full)
    print('Columns missing values')
    pprint(X_full.columns[X_full.isnull().any()])

    from importlib import import_module
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split

    # Function to execute pipeline generated by AlphaAutoMLEdit
    def eval_pipeline(strings, origin):
        steps = []
        logger.info('PIPELINE: %s', ",".join(strings))
        for string in strings:
            fields = string.split('.')
            module_name = ".".join(fields[0:len(fields)-1])
            class_name = fields[-1]
            module = import_module(module_name)
            class_ = getattr(module, class_name)
            instance = class_()

            if 'OneHotEncoder' in class_name:
                instance = class_(handle_unknown='ignore')
            if 'SimpleImputer' in class_name:
                instance = class_(strategy='most_frequent')

            steps.append(instance)

        #mean_scores = float('inf')
        if steps:
            try:
                estimator = make_pipeline(*steps)
                X_train, X_test, y_train, y_test = train_test_split(X_full.values, y_full.values, test_size=0.33, random_state=0)
                clf = estimator.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                score = None
                logger.info('METRIC %s', input['METRIC'])
                if input['METRIC']  == 'mean_squared_error':
                    from sklearn.metrics import mean_squared_error
                    score = mean_squared_error(y_test, y_pred)
                elif input['METRIC']  == 'f1_macro':
                    from sklearn.metrics import f1_score
                    score =  f1_score(y_test, y_pred, average='macro')
                elif input['METRIC']  == 'accuracy':
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_test, y_pred)

                logger.info('SCORE: %s', score)
                return score
                #mean_scores = cross_val_score(estimator, X_full.values, y_full.values,
                #                              scoring=input['METRIC'],
                #                              cv=5)
            except Exception as e:
                logger.info('%s', e)
               	logger.info('EXCEPTION: %s', str(e))
                logger.info('Could not execute %s', strings)
                return None

        #return sum(mean_scores)/len(mean_scores)


    # *************** Steps for initializing and running AlphaAutoMLEdit*********************

    input['DATASET_METAFEATURES'] = dataset_metafeatures
    GRAMMAR['TERMINALS'] = getTerminals(GRAMMAR['NON_TERMINALS'], PRIMITIVES, problem)
    r, l = getRules(GRAMMAR['NON_TERMINALS'], PRIMITIVES, problem)
    GRAMMAR['RULES'] = r
    GRAMMAR['RULES_LOOKUP'] = l

    pprint(GRAMMAR)

    input['GRAMMAR'] = GRAMMAR
    input['PROBLEM'] = problem
    input['DATA_TYPE'] = 'TABULAR'
    input['METRIC'] = metric
    input['DATASET'] = dataset_name
    if not input_args is None:
        f = open(input_args)
        input['ARGS'] = json.load(f)['ARGS']
    input['ARGS']['stepsfile'] = os.path.join(input['ARGS']['stepsfile'], dataset_name+'_pipeline_steps.txt')
    logger.info('ARGS %s', input['ARGS'])

    game = PipelineGame(input, eval_pipeline)
    nnet = NNetWrapper(game)

    if input['ARGS'].get('load_model'):
        model_file = os.path.join(input['ARGS'].get('load_folder_file')[0],
                                  input['ARGS'].get('load_folder_file')[1])
        if os.path.isfile(model_file):
            nnet.load_checkpoint(input['ARGS'].get('load_folder_file')[0],
                                 input['ARGS'].get('load_folder_file')[1])

    c = Coach(game, nnet, input['ARGS'])
    c.learn()

    # **************************************************************************************

    return game


def main():
    setup_logging()

    output_path = '/home/ubuntu/temp/results/classification'
    dataset_path = "./alphaAutoMLEdit/test/data/learningData.csv"
    dataset_name = 'TEST_DATA'
    problem = "CLASSIFICATION"
    target_col = "class"
    metric = 'f1_macro'
    input_args = None

    args = sys.argv

    arg_count = len(args)

    for i in range(1, arg_count):
        if i == 1:
            dataset_path = args[i]
        if i== 2:
            dataset_name = args[i]
        if i == 3:
            problem = args[i].upper()
        if i == 4:
            target_col = args[i]
        if i == 5:
            metric = args[i]
        if i == 6:
            output_path = args[i]
        if i == 7:
            input_args = args[i]


    if problem not in ['CLASSIFICATION', 'REGRESSION']:
        logger.info('%s NOT SUPPORTED', problem)
        sys.exit(0)

    import time
    start = time.time()

    game = generate(dataset_path, dataset_name, problem, target_col, metric, input_args)

    end = time.time()

    eval_dict = game.evaluations
    eval_times = game.eval_times
    for key, value in eval_dict.items():
        if value == float('inf') and not 'error' in game.metric.lower():
           eval_dict[key] = 0

    total_valid_pipelines = str(np.where(np.asarray(list(eval_dict.values())) != float('inf'))[0].size) if 'error' in metric else  str(np.where(np.asarray(list(eval_dict.values()))>0)[0].size)
    evaluations = sorted(eval_dict.items(), key=operator.itemgetter(1))
    if not 'error' in game.metric.lower():
        evaluations.reverse()
    logger.info('GENERATED PIPELINES %s', evaluations)
    out_p = open(os.path.join(output_path, dataset_name+'_best_pipelines.txt'), 'a')
    out_p.write(dataset_name+' '+evaluations[0][0] + ' ' + str(evaluations[0][1])+ ' ' +  str((eval_times[evaluations[0][0]]-start)/60.0)  + ' ' + ' ' + str((end-start)/60.0) + ' ' + str(game.steps) + ' ' + total_valid_pipelines + '\n')
    out_p.close()
    out_p = open(os.path.join(output_path, dataset_name+'_pipeline_times.txt'), 'a')
    evaluations = sorted(eval_times.items(), key=operator.itemgetter(1))
    for i in range(0, len(evaluations)):
        out_p.write(evaluations[i][0]+' '+str(eval_dict[evaluations[i][0]])+' ' + str((evaluations[i][1]-start)/60.0)+'\n')
    out_p.close()


if __name__ == '__main__':
    main()
