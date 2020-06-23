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
                     'd3m.primitives.schema_discovery.profiler.DSBOX',
                     'd3m.primitives.data_cleaning.column_type_profiler.Simon'}


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


def create_vectors_from_metalearningdb(task, current_primitives, current_primitive_types, rules):
    pipelines_metalearningdb = load_metalearningdb(task)
    primitives_by_type = load_primitives_by_type()
    primitives_by_name = load_primitives_by_name()
    current_primitive_ids = {}
    train_examples = []

    for primitive_name in current_primitives:
        if primitive_name != 'E':  # Special symbol for empty primitive
            current_primitive_ids[primitives_by_name[primitive_name]] = current_primitives[primitive_name]

    primitives_distribution = analyze_distribution(pipelines_metalearningdb)
    action_probabilities = {}
    actions = [i for i, j in sorted(rules.items(), key=lambda x: x[1])]
    for primitive_type, primitives_info in primitives_distribution.items():
        for primitive, distribution in primitives_info.items():
            action = primitive_type + ' -> ' + primitive
            action_probabilities[action] = distribution

    for pipeline, score in pipelines_metalearningdb:
        if all(primitive in current_primitive_ids for primitive in pipeline):
            size_vector = len(current_primitives) + len(current_primitive_types)
            primitives_vector = [0] * size_vector
            primitive_types_vector = [0] * size_vector
            action_vector = [0] * len(actions)

            for primitive_id in pipeline:
                primitive_type = primitives_by_type[primitive_id]
                primitives_vector[current_primitive_ids[primitive_id]] = 1
                primitive_types_vector[current_primitive_types[primitive_type]] = 1
                # TODO: check if should be current_primitive_ids[primitive_id] - 1

            for index, action in enumerate(actions):
                if action in action_probabilities:
                    action_vector[index] = action_probabilities[action]
            # TODO: we send always the same action_vector, should they be different?

            #print('<<<<<<<<<<')
            #print([primitives_by_type[p] for p in pipeline], score)
            #print(primitive_types_vector, score)
            #print(list(pipeline.values()), score)
            #print(action_vector)
            #print(actions)
            #print(primitives_vector, score)
            #print('>>>>>>>>>>>>')
            metafeature_vector = [0] * 50 + [1, 1]  # problem (classification) and datatype (tabular)
            train_example = (metafeature_vector + primitive_types_vector, action_vector, score)
            train_examples.append(train_example)
            train_example = (metafeature_vector + primitives_vector, action_vector, score)
            train_examples.append(train_example)

    logger.info('Found %d training examples for task %s', len(train_examples)/2, task)

    return train_examples


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


def analyze_distribution(pipelines_metalearningdb):
    primitives_by_type = load_primitives_by_type()
    primitives_by_id = load_primitives_by_id()
    primitive_frequency = {}
    primitive_distribution = {}

    for pipeline, score in pipelines_metalearningdb:
        for primitive_id in pipeline:
            primitive_type = primitives_by_type[primitive_id]
            if primitive_type not in primitive_frequency:
                primitive_frequency[primitive_type] = {'primitives': {}, 'total': 0}
            primitive_name = primitives_by_id[primitive_id]
            if primitive_name not in primitive_frequency[primitive_type]['primitives']:
                primitive_frequency[primitive_type]['primitives'][primitive_name] = 0
            primitive_frequency[primitive_type]['primitives'][primitive_name] += 1
            primitive_frequency[primitive_type]['total'] += 1

    for primitive_type, primitives_info in primitive_frequency.items():
        if primitive_type not in primitive_distribution:
            primitive_distribution[primitive_type] = OrderedDict()
        for primitive, frequency in sorted(primitives_info['primitives'].items(), key=lambda x: x[1], reverse=True):
            distribution = frequency / primitives_info['total']
            primitive_distribution[primitive_type][primitive] = distribution
        print(primitive_type)
        print(['%s %s' % (k, round(v, 4)) for k, v in primitive_distribution[primitive_type].items()])

    return primitive_distribution


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
    available_primitives = set()

    with open(join(os.path.dirname(__file__), '../resource/primitives_by_type.json')) as fin:
        for primitive_type, primitive_names in json.load(fin).items():
            for primitive_name in primitive_names:
                available_primitives.add(primitive_name)

    with open(join(os.path.dirname(__file__), '../resource/primitives_by_name.json')) as fin:
        primitives = json.load(fin)

    for primitive in primitives:
        if primitive['python_path'] in available_primitives:
            primitives_by_name[primitive['python_path']] = primitive['id']

    return primitives_by_name


def load_primitives_by_id():
    primitives_by_id = {}
    available_primitives = set()

    with open(join(os.path.dirname(__file__), '../resource/primitives_by_type.json')) as fin:
        for primitive_type, primitive_names in json.load(fin).items():
            for primitive_name in primitive_names:
                available_primitives.add(primitive_name)

    with open(join(os.path.dirname(__file__), '../resource/primitives_by_name.json')) as fin:
        primitives = json.load(fin)

    for primitive in primitives:
        if primitive['python_path'] in available_primitives:
            primitives_by_id[primitive['id']] = primitive['python_path']

    return primitives_by_id


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
    analyze_distribution(load_metalearningdb(task))
    non_terminals = {x:i+1 for i, x in enumerate(set(load_primitives_by_type().values()))}
    terminals = {x:len(non_terminals)+ i for i, x in enumerate(load_primitives_by_name().keys())}
    terminals['E'] = 0
    rules = {'S -> DATA_CLEANING GROUP_PREPROCESSING FEATURE_SELECTION CLASSIFICATION': 1, 'GROUP_PREPROCESSING -> DATA_PREPROCESSING': 2, 'DATA_PREPROCESSING -> E': 3, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.binarizer.SKlearn': 4, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.count_vectorizer.SKlearn': 5, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.feature_agglomeration.SKlearn': 6, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.max_abs_scaler.SKlearn': 7, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.min_max_scaler.SKlearn': 8, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.normalizer.SKlearn': 9, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.nystroem.SKlearn': 10, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.polynomial_features.SKlearn': 11, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.quantile_transformer.SKlearn': 12, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.random_trees_embedding.SKlearn': 13, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.rbf_sampler.SKlearn': 14, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.robust_scaler.SKlearn': 15, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.standard_scaler.SKlearn': 16, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.tfidf_vectorizer.SKlearn': 17, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.truncated_svd.SKlearn': 18, 'DATA_PREPROCESSING -> d3m.primitives.classification.ensemble_voting.DSBOX': 19, 'DATA_PREPROCESSING -> d3m.primitives.column_parser.preprocess_categorical_columns.Cornell': 20, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.audio_reader.Common': 21, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.audio_reader.DistilAudioDatasetLoader': 22, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.csv_reader.Common': 23, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.data_cleaning.DistilTimeSeriesFormatter': 24, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX': 25, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.dataset_sample.Common': 26, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.dataset_text_reader.DatasetTextReader': 27, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.datetime_range_filter.Common': 28, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.do_nothing.DSBOX': 29, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.do_nothing_for_dataset.DSBOX': 30, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.encoder.DSBOX': 31, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.flatten.DataFrameCommon': 32, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.greedy_imputation.DSBOX': 33, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.horizontal_concat.DSBOX': 34, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.image_reader.Common': 35, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.iterative_regression_imputation.DSBOX': 36, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.label_decoder.Common': 37, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.label_encoder.Common': 38, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.largest_connected_component.JHU': 39, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.low_rank_imputer.Cornell': 40, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.lupi_mfa.lupi_mfa.LupiMFA': 41, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.mean_imputation.DSBOX': 42, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.numeric_range_filter.Common': 43, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.one_hot_encoder.MakerCommon': 44, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.one_hot_encoder.PandasCommon': 45, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.random_sampling_imputer.BYU': 46, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.regex_filter.Common': 47, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.splitter.DSBOX': 48, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.term_filter.Common': 49, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.text_reader.Common': 50, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.time_series_to_list.DSBOX': 51, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.unary_encoder.DSBOX': 52, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.vertical_concatenate.DSBOX': 53, 'DATA_PREPROCESSING -> d3m.primitives.data_preprocessing.video_reader.Common': 54, 'DATA_PREPROCESSING -> d3m.primitives.feature_selection.mutual_info_classif.DistilMIRanking': 55, 'DATA_CLEANING -> E': 56, 'DATA_CLEANING -> d3m.primitives.data_cleaning.imputer.SKlearn': 57, 'DATA_CLEANING -> d3m.primitives.data_cleaning.missing_indicator.SKlearn': 58, 'DATA_CLEANING -> d3m.primitives.data_cleaning.string_imputer.SKlearn': 59, 'DATA_CLEANING -> d3m.primitives.data_cleaning.clean_augmentation.AutonBox': 60, 'DATA_CLEANING -> d3m.primitives.data_cleaning.cleaning_featurizer.DSBOX': 61, 'DATA_CLEANING -> d3m.primitives.data_cleaning.column_type_profiler.Simon': 62, 'DATA_CLEANING -> d3m.primitives.data_cleaning.data_cleaning.Datacleaning': 63, 'DATA_CLEANING -> d3m.primitives.data_cleaning.geocoding.Goat_forward': 64, 'DATA_CLEANING -> d3m.primitives.data_cleaning.geocoding.Goat_reverse': 65, 'DATA_CLEANING -> d3m.primitives.data_cleaning.label_encoder.DSBOX': 66, 'DATA_CLEANING -> d3m.primitives.data_cleaning.tabular_extractor.Common': 67, 'DATA_CLEANING -> d3m.primitives.data_cleaning.text_summarization.Duke': 68, 'FEATURE_SELECTION -> E': 69, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.generic_univariate_select.SKlearn': 70, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.select_fwe.SKlearn': 71, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.select_percentile.SKlearn': 72, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.variance_threshold.SKlearn': 73, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.joint_mutual_information.AutoRPI': 74, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.pca_features.Pcafeatures': 75, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.rffeatures.Rffeatures': 76, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.score_based_markov_blanket.RPI': 77, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.simultaneous_markov_blanket.AutoRPI': 78, 'FEATURE_SELECTION -> d3m.primitives.feature_selection.skfeature.TAMU': 79, 'CLASSIFICATION -> d3m.primitives.classification.ada_boost.SKlearn': 80, 'CLASSIFICATION -> d3m.primitives.classification.bagging.SKlearn': 81, 'CLASSIFICATION -> d3m.primitives.classification.bernoulli_naive_bayes.SKlearn': 82, 'CLASSIFICATION -> d3m.primitives.classification.decision_tree.SKlearn': 83, 'CLASSIFICATION -> d3m.primitives.classification.dummy.SKlearn': 84, 'CLASSIFICATION -> d3m.primitives.classification.extra_trees.SKlearn': 85, 'CLASSIFICATION -> d3m.primitives.classification.gaussian_naive_bayes.SKlearn': 86, 'CLASSIFICATION -> d3m.primitives.classification.gradient_boosting.SKlearn': 87, 'CLASSIFICATION -> d3m.primitives.classification.k_neighbors.SKlearn': 88, 'CLASSIFICATION -> d3m.primitives.classification.linear_discriminant_analysis.SKlearn': 89, 'CLASSIFICATION -> d3m.primitives.classification.linear_svc.SKlearn': 90, 'CLASSIFICATION -> d3m.primitives.classification.logistic_regression.SKlearn': 91, 'CLASSIFICATION -> d3m.primitives.classification.mlp.SKlearn': 92, 'CLASSIFICATION -> d3m.primitives.classification.multinomial_naive_bayes.SKlearn': 93, 'CLASSIFICATION -> d3m.primitives.classification.nearest_centroid.SKlearn': 94, 'CLASSIFICATION -> d3m.primitives.classification.passive_aggressive.SKlearn': 95, 'CLASSIFICATION -> d3m.primitives.classification.quadratic_discriminant_analysis.SKlearn': 96, 'CLASSIFICATION -> d3m.primitives.classification.random_forest.SKlearn': 97, 'CLASSIFICATION -> d3m.primitives.classification.sgd.SKlearn': 98, 'CLASSIFICATION -> d3m.primitives.classification.svc.SKlearn': 99, 'CLASSIFICATION -> d3m.primitives.classification.bert_classifier.DistilBertPairClassification': 100, 'CLASSIFICATION -> d3m.primitives.classification.cover_tree.Fastlvm': 101, 'CLASSIFICATION -> d3m.primitives.classification.gaussian_classification.JHU': 102, 'CLASSIFICATION -> d3m.primitives.classification.general_relational_dataset.GeneralRelationalDataset': 103, 'CLASSIFICATION -> d3m.primitives.classification.light_gbm.Common': 104, 'CLASSIFICATION -> d3m.primitives.classification.lstm.DSBOX': 105, 'CLASSIFICATION -> d3m.primitives.classification.lupi_rf.LupiRFClassifier': 106, 'CLASSIFICATION -> d3m.primitives.classification.lupi_rfsel.LupiRFSelClassifier': 107, 'CLASSIFICATION -> d3m.primitives.classification.lupi_svm.LupiSvmClassifier': 108, 'CLASSIFICATION -> d3m.primitives.classification.random_classifier.Test': 109, 'CLASSIFICATION -> d3m.primitives.classification.random_forest.Common': 110, 'CLASSIFICATION -> d3m.primitives.classification.search.Find_projections': 111, 'CLASSIFICATION -> d3m.primitives.classification.search_hybrid.Find_projections': 112, 'CLASSIFICATION -> d3m.primitives.classification.text_classifier.DistilTextClassifier': 113, 'CLASSIFICATION -> d3m.primitives.classification.tree_augmented_naive_bayes.BayesianInfRPI': 114, 'CLASSIFICATION -> d3m.primitives.classification.xgboost_dart.Common': 115, 'CLASSIFICATION -> d3m.primitives.classification.xgboost_gbtree.Common': 116}
    create_vectors_from_metalearningdb(task, terminals, non_terminals, rules)
