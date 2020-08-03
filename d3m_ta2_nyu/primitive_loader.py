import os
import logging
import json
from d3m import index

logger = logging.getLogger(__name__)


PRIMITIVES_BY_NAME_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_by_name.json')
PRIMITIVES_BY_TYPE_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_by_type.json')

INSTALLED_PRIMITIVES = index.search()

BLACK_LIST = {
    'd3m.primitives.classification.canonical_correlation_forests.UBC',
    'd3m.primitives.classification.global_causal_discovery.ClassifierRPI',
    'd3m.primitives.classification.inceptionV3_image_feature.Gator',
    'd3m.primitives.classification.tree_augmented_naive_bayes.BayesianInfRPI'
}


def get_primitive_class(name):
    return index.get_primitive(name)


def get_primitive_family(name):
    return get_primitive_class(name).metadata.to_json_structure()['primitive_family']


def get_primitive_algorithms(name):
    return get_primitive_class(name).metadata.to_json_structure()['algorithm_types']


def get_primitive_info(name):
    primitive_dict =  get_primitive_class(name).metadata.to_json_structure()

    return {
            'id': primitive_dict['id'],
            'name': primitive_dict['name'],
            'version': primitive_dict['version'],
            'python_path': primitive_dict['python_path'],
            'digest': primitive_dict['digest']
    }


def get_primitives_by_type():
    if os.path.isfile(PRIMITIVES_BY_TYPE_PATH):
        with open(PRIMITIVES_BY_TYPE_PATH) as fin:
            primitives = json.load(fin)
        logger.info('Loading primitives info from file')

        return primitives

    primitives = {}
    count = 1
    for primitive_name in INSTALLED_PRIMITIVES:
        if primitive_name not in BLACK_LIST:
            try:
                family = get_primitive_family(primitive_name)
                algorithm_types = get_primitive_algorithms(primitive_name)
            except:
                logger.error('Loading metadata about primitive %s', primitive_name)
                continue
            #  Use the algorithm types as families since they are more descriptive
            if family in {'DATA_TRANSFORMATION', 'DATA_PREPROCESSING', 'DATA_CLEANING'}:
                family = algorithm_types[0]

            # Changing the primitive families using some predefined rules
            if primitive_name in {'d3m.primitives.feature_construction.corex_text.DSBOX',
                                  'd3m.primitives.data_transformation.encoder.DistilTextEncoder',
                                  'd3m.primitives.data_preprocessing.tfidf_vectorizer.SKlearn',
                                  'd3m.primitives.data_preprocessing.count_vectorizer.SKlearn'}:
                family = 'TEXT_ENCODER'
            if primitive_name in {'d3m.primitives.data_transformation.data_cleaning.DistilEnrichDates'}:
                family = 'DATETIME_ENCODER'
            if family == 'ENCODE_ONE_HOT':
                family = 'CATEGORICAL_ENCODER'

            if family not in primitives:
                primitives[family] = {}
            primitives[family][primitive_name] = count
            count += 1

    with open(PRIMITIVES_BY_TYPE_PATH, 'w') as fout:
        json.dump(primitives, fout, indent=4)
    logger.info('Loading primitives info from D3M index')

    return primitives


def get_primitives_by_name():
    if os.path.isfile(PRIMITIVES_BY_NAME_PATH):
        with open(PRIMITIVES_BY_NAME_PATH) as fin:
            primitives = json.load(fin)
        logger.info('Loading primitives info from file')

        return primitives

    primitives = []

    for primitive_name in INSTALLED_PRIMITIVES:
        if primitive_name not in BLACK_LIST:
            try:
                primitive_info = get_primitive_info(primitive_name)
            except:
                logger.error('Loading metadata about primitive %s', primitive_name)
                continue
            primitives.append(primitive_info)

    with open(PRIMITIVES_BY_NAME_PATH, 'w') as fout:
        json.dump(primitives, fout, indent=4)
    logger.info('Loading primitives info from D3M index')

    return primitives
