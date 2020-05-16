import os
import json
import logging
from os.path import join
from collections import OrderedDict

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


IGNORE_PRIMITIVES = {'d3m.primitives.data_transformation.construct_predictions.Common',
                     'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                     'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                     'd3m.primitives.data_transformation.denormalize.Common',
                     'd3m.primitives.data_transformation.column_parser.Common',
                     'd3m.primitives.schema_discovery.profiler.DSBOX'}


def merge_pipeline_files(pipelines_file, pipeline_runs_file, problems_file, n=-1, verbose=False):
    print("Adding pipelines to lookup table...")
    pipelines = {}
    with open(pipelines_file, "r") as f:
        for line in f:
            pipeline = json.loads(line)
            pipelines[pipeline["digest"]] = pipeline

    print("Adding problems to lookup table...")
    problems = {}
    with open(problems_file, "r") as f:
        for line in f:
            problem = json.loads(line)
            problems[problem["digest"]] = problem["problem"]
            problems[problem["digest"]]["id"] = problem["id"]

    print("Merging pipeline information with pipeline_runs_file (this might take a while)...")
    merged = []
    with open(pipeline_runs_file, "r") as f:
        for line in f:
            if len(merged) == n:
                break
            try:
                run = json.loads(line)
                if run['run']['phase'] != 'PRODUCE':
                    continue
                pipeline = pipelines[run["pipeline"]["digest"]]
                problem = problems[run["problem"]["digest"]]
                data = {
                    'pipeline_id': pipeline['id'],
                    'pipeline_digest': pipeline['digest'],
                    'pipeline_source': pipeline['source'],
                    'inputs': pipeline['inputs'],
                    'outputs': pipeline['outputs'],
                    'problem': problem,
                    'start': run['start'],
                    'end': run['end'],
                    'steps': pipeline['steps'],
                    'scores': run['run']['results']['scores']
                }
                merged.append(json.dumps(data))
            except Exception as e:
                if (verbose):
                    print(problem['id'], repr(e))
        print("Done.")

    with open(join(os.path.dirname(__file__), '../resource/metalearningdb.json'), 'w') as fout:
        fout.write('\n'.join(merged))


def load_metalearningdb(task):
    primitives_by_name = load_primitives_by_name()
    primitive_ids = set(primitives_by_name.values())
    ignore_primitives_ids = set()
    all_pipelines = []
    task_pipelines = []

    logger.info('Loading pipelines from metalearning database...')

    with open(join(os.path.dirname(__file__), '../resource/metalearningdb.json')) as fin:
        for line in fin:
            all_pipelines.append(json.loads(line))

    for ignore_primitive in IGNORE_PRIMITIVES:
        ignore_primitives_ids.add(primitives_by_name[ignore_primitive])

    for pipeline_run in all_pipelines:
        pipeline_primitives = pipeline_run['steps']
        if is_target_task(pipeline_run['problem'], task) and is_available_primitive(pipeline_primitives, primitive_ids):
            primitives = filter_primitives(pipeline_primitives, ignore_primitives_ids)
            if len(primitives) > 0:
                score = pipeline_run['scores'][0]['value']
                task_pipelines.append((primitives, score))

    logger.info('Found %d pipelines for task %s', len(task_pipelines), task)

    return task_pipelines


def create_vectors_from_metalearningdb(task, current_primitives, current_primitive_types):
    pipelines_metalearningdb = load_metalearningdb(task)
    primitives_by_type = load_primitives_by_type()
    primitives_by_name = load_primitives_by_name()
    current_primitive_ids = {}
    vectors = []

    for primitive_name in current_primitives:
        if primitive_name != 'E':  # Special symbol for empty primitive
            current_primitive_ids[primitives_by_name[primitive_name]] = current_primitives[primitive_name]

    for pipeline, score in pipelines_metalearningdb:
        primitive_types = [primitives_by_type[p] for p in pipeline]
        size_vector = len(current_primitives) + len(current_primitive_types)
        primitives_vector = [0] * size_vector
        primitive_types_vector = [0] * size_vector

        for primitive_id in pipeline.keys():
            primitives_vector[current_primitive_ids[primitive_id]] = 1

        for primitive_type in primitive_types:
            primitive_types_vector[current_primitive_types[primitive_type]] = 1

    return vectors


def create_grammar_from_metalearningdb(task):
    pipelines = load_metalearningdb(task)
    patterns = extract_patterns(pipelines)
    # TODO: Convert patterns to grammar format


def extract_patterns(pipelines):
    primitives_by_type = load_primitives_by_type()
    patterns = {}

    for pipeline, score in sorted(pipelines, key=lambda x: x[1], reverse=True):
        primitive_types = [primitives_by_type[p] for p in pipeline]
        pattern_id = ' '.join(primitive_types)
        if pattern_id not in patterns:
            patterns[pattern_id] = primitive_types

    logger.info('Found %d different patterns:\n%s', len(patterns), '\n'.join([str(x) for x in patterns.values()]))

    return list(patterns.values())


def is_available_primitive(pipeline_primitives, current_primitives):
    for primitive in pipeline_primitives:
        if primitive['primitive']['id'] not in current_primitives:
            logger.warning('Primitive %s is not longer available' % primitive['primitive']['python_path'])
            return False
    return True


def is_target_task(problem, task):
    problem_task = None
    if 'task_type' in problem:
        problem_task = [problem['task_type']]
    elif 'task_keywords' in problem:
        if 'CLASSIFICATION' in problem['task_keywords'] and 'TABULAR' in problem['task_keywords']:
            problem_task = 'CLASSIFICATION'

    if task == problem_task:
        return True

    return False


def filter_primitives(pipeline_steps, ignore_primitives):
    primitives = OrderedDict()

    for pipeline_step in pipeline_steps:
        if pipeline_step['primitive']['id'] not in ignore_primitives:
                primitives[pipeline_step['primitive']['id']] = pipeline_step['primitive']['python_path']

    return primitives


def load_primitives_by_name():
    primitives_by_name = {}

    with open(join(os.path.dirname(__file__), '../resource/primitives_by_name.json')) as fin:
        primitives = json.load(fin)

    for primitive in primitives:
        primitives_by_name[primitive['python_path']] = primitive['id']

    return primitives_by_name


def load_primitives_by_type():
    primitives_by_type = {}
    primitives_by_name = load_primitives_by_name()

    with open(join(os.path.dirname(__file__), '../resource/primitives_by_type.json')) as fin:
        primitives = json.load(fin)

    for primitive_type in primitives:
        primitive_names = primitives[primitive_type].keys()
        for primitive_name in primitive_names:
            primitives_by_type[primitives_by_name[primitive_name]] = primitive_type

    return primitives_by_type


if __name__ == '__main__':
    task = 'CLASSIFICATION'
    pipelines_file = '/Users/rlopez/Downloads/metalearningdb_dump_20200304/pipelines-1583354358.json'
    pipeline_runs_file = '/Users/rlopez/Downloads/metalearningdb_dump_20200304/pipeline_runs-1583354387.json'
    problems_file = '/Users/rlopez/Downloads/metalearningdb_dump_20200304/problems-1583354357.json'
    #merge_pipeline_files(pipelines_file, pipeline_runs_file, problems_file)
    #create_grammar_from_metalearningdb(task)
    non_terminals = {x:i+1 for i, x in enumerate(set(load_primitives_by_type().values()))}
    terminals = {x:len(non_terminals)+ i for i, x in enumerate(load_primitives_by_name().keys())}
    terminals['E'] = 0
    create_vectors_from_metalearningdb(task, terminals, non_terminals)
    '''create_vectors_from_metalearningdb(task, {'d3m.primitives.classification.bagging.SKlearn': 3,
                                             'd3m.primitives.classification.random_forest.SKlearn': 4,
                                             'd3m.primitives.classification.decision_tree.SKlearn': 5,
                                             'd3m.primitives.data_cleaning.imputer.SKlearn': 6,
                                             'E': 0}, {'CLASSIFICATION': 1, 'DATA_CLEANING':2})
    '''
