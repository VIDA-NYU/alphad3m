import logging
import datamart_profiler
import pandas as pd
from d3m.container.dataset import D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES

logger = logging.getLogger(__name__)


def read_annotated_feature_types(dataset_doc):
    feature_types = {}
    try:
        for data_res in dataset_doc['dataResources']:
            if data_res['resType'] == 'table':
                for column in data_res['columns']:
                    if 'attribute' in column['role'] and column['colType'] != 'unknown':
                        feature_types[column['colName']] = (D3M_COLUMN_TYPE_CONSTANTS_TO_SEMANTIC_TYPES[column['colType']],
                                                            'refersTo' in column)
    except:
        logger.exception('Error reading the type of attributes')

    logger.info('Features with annotated types: [%s]', ', '.join(feature_types.keys()))

    return feature_types


def select_unkown_feature_types(csv_path, annotated_features, target_names, index_name):
    all_features = pd.read_csv(csv_path).columns
    unkown_feature_types = []

    for feature_name in all_features:
        if feature_name not in target_names and feature_name != index_name and feature_name not in annotated_features:
            unkown_feature_types.append(feature_name)

    logger.info('Features with unknown types: [%s]', ', '.join(unkown_feature_types))

    return unkown_feature_types


def indentify_feature_types(csv_path, unkown_feature_types, target_names, index_name):
    metadata = datamart_profiler.process_dataset(csv_path)
    new_types = {'https://metadata.datadrivendiscovery.org/types/Attribute': []}

    for index, item in enumerate(metadata['columns']):
        column_name = item['name']
        if column_name == index_name:
            new_types['https://metadata.datadrivendiscovery.org/types/PrimaryKey'] = [(index, column_name)]
        elif column_name in target_names:
            new_types['https://metadata.datadrivendiscovery.org/types/TrueTarget'] = [(index, column_name)]
        elif column_name in unkown_feature_types:
            semantic_types = item['semantic_types'] if len(item['semantic_types']) > 0 else [item['structural_type']]
            for semantic_type in semantic_types:
                if semantic_type == 'http://schema.org/Enumeration':  # Changing to D3M format
                    semantic_type = 'https://metadata.datadrivendiscovery.org/types/CategoricalData'
                if semantic_type == 'http://schema.org/identifier':
                    semantic_type = 'http://schema.org/Integer'
                if semantic_type not in new_types:
                    new_types[semantic_type] = []
                new_types[semantic_type].append((index, column_name))

            new_types['https://metadata.datadrivendiscovery.org/types/Attribute'].append((index, column_name))

    logger.info('New feature types:\n%s',
                '\n'.join(['%s = [%s]' % (k, ', '.join([i for _, i in v])) for k, v in new_types.items()]))

    return {k: [i for i, _ in v] for k, v in new_types.items()}


def profile_data(csv_path, index_name, target_names, dataset_doc):
    annotated_feature_types = read_annotated_feature_types(dataset_doc)
    unkown_feature_types = select_unkown_feature_types(csv_path, annotated_feature_types.keys(), target_names,
                                                       index_name)
    inferred_types = {}
    if len(unkown_feature_types) > 0:
        inferred_types = indentify_feature_types(csv_path, unkown_feature_types, target_names, index_name)
    #  Filter out  types of features which are foreign keys of other tables
    filtered_annotated_types = {k: v[0] for k, v in annotated_feature_types.items() if not v[1]}
    all_types = list(filtered_annotated_types.values()) + list(inferred_types.keys())

    return inferred_types, all_types
