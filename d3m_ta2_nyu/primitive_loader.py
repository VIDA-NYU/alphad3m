import os
import logging
import json
from d3m import index

logger = logging.getLogger(__name__)


PRIMITIVES_BY_NAME_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_by_name.json')
PRIMITIVES_BY_TYPE_PATH = os.path.join(os.path.dirname(__file__), '../resource/primitives_by_type.json')

INSTALLED_PRIMITIVES = sorted(index.search(), key=lambda x: x.endswith('SKlearn'), reverse=True)

BLACK_LIST = {
    'd3m.primitives.classification.random_classifier.Test',
    'd3m.primitives.classification.global_causal_discovery.ClassifierRPI',
    'd3m.primitives.classification.tree_augmented_naive_bayes.BayesianInfRPI',
    'd3m.primitives.classification.simple_cnaps.UBC',
    'd3m.primitives.classification.logistic_regression.UBC',
    'd3m.primitives.classification.multilayer_perceptron.UBC',
    'd3m.primitives.classification.canonical_correlation_forests.UBC',
    'd3m.primitives.regression.multilayer_perceptron.UBC',
    'd3m.primitives.regression.canonical_correlation_forests.UBC',
    'd3m.primitives.regression.linear_regression.UBC',
    'd3m.primitives.classification.inceptionV3_image_feature.Gator',
    'd3m.primitives.classification.search.Find_projections',
    'd3m.primitives.classification.search_hybrid.Find_projections',
    'd3m.primitives.regression.search_hybrid_numeric.Find_projections',
    'd3m.primitives.regression.search_numeric.Find_projections',
    'd3m.primitives.data_cleaning.binarizer.SKlearn',
    'd3m.primitives.feature_selection.rffeatures.Rffeatures',
    'd3m.primitives.feature_selection.mutual_info_classif.DistilMIRanking',
    'd3m.primitives.dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne',
    'd3m.primitives.data_cleaning.string_imputer.SKlearn',
    'd3m.primitives.data_cleaning.tabular_extractor.Common',
    'd3m.primitives.data_cleaning.missing_indicator.SKlearn',
    'd3m.primitives.data_transformation.gaussian_random_projection.SKlearn',
    'd3m.primitives.data_transformation.sparse_random_projection.SKlearn',
    'd3m.primitives.feature_extraction.boc.UBC',
    'd3m.primitives.feature_extraction.bow.UBC',
    'd3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec',
    'd3m.primitives.classification.mlp.BBNMLPClassifier',
    # Repeated primitives:
    'd3m.primitives.data_transformation.unary_encoder.DSBOX',
    'd3m.primitives.data_transformation.one_hot_encoder.TPOT',
    'd3m.primitives.data_transformation.one_hot_encoder.MakerCommon',
    'd3m.primitives.data_transformation.one_hot_encoder.PandasCommon',
    'd3m.primitives.feature_extraction.tfidf_vectorizer.BBNTfidfTransformer'
}


def get_primitive_class(name):
    return index.get_primitive(name)


def get_primitive_family(name):
    return get_primitive_class(name).metadata.to_json_structure()['primitive_family']


def get_primitive_algorithms(name):
    return get_primitive_class(name).metadata.to_json_structure()['algorithm_types']


def get_primitive_info(name):
    primitive_dict = get_primitive_class(name).metadata.to_json_structure()

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
    for primitive_name in INSTALLED_PRIMITIVES:
        if primitive_name not in BLACK_LIST:
            try:
                family = get_primitive_family(primitive_name)
                algorithm_types = get_primitive_algorithms(primitive_name)
            except:
                logger.error('Loading metadata about primitive %s', primitive_name)
                continue
            #  Use the algorithm types as families because they are more descriptive
            if family in {'DATA_TRANSFORMATION', 'DATA_PREPROCESSING', 'DATA_CLEANING'}:
                family = algorithm_types[0]

            # Changing the primitive families using some predefined rules
            if primitive_name in {'d3m.primitives.feature_construction.corex_text.DSBOX',
                                  'd3m.primitives.data_transformation.encoder.DistilTextEncoder',
                                  'd3m.primitives.feature_extraction.tfidf_vectorizer.SKlearn',
                                  'd3m.primitives.feature_extraction.count_vectorizer.SKlearn'}:
                family = 'TEXT_ENCODER'

            elif primitive_name in {'d3m.primitives.data_cleaning.quantile_transformer.SKlearn',
                                    'd3m.primitives.data_cleaning.normalizer.SKlearn',
                                    'd3m.primitives.normalization.iqr_scaler.DSBOX'}:
                family = 'FEATURE_SCALING'

            elif primitive_name in {'d3m.primitives.feature_extraction.feature_agglomeration.SKlearn',
                                    'd3m.primitives.feature_selection.mutual_info_classif.DistilMIRanking'}:
                family = 'FEATURE_SELECTION'

            elif primitive_name in {'d3m.primitives.feature_extraction.pca.SKlearn',
                                    'd3m.primitives.feature_extraction.truncated_svd.SKlearn',
                                    'd3m.primitives.feature_extraction.pca_features.RandomizedPolyPCA',
                                    'd3m.primitives.data_transformation.gaussian_random_projection.SKlearn',
                                    'd3m.primitives.data_transformation.sparse_random_projection.SKlearn'}:
                family = 'DIMENSIONALITY_REDUCTION'  # Or should it be FEATURE_SELECTION ?

            elif primitive_name in {'d3m.primitives.feature_extraction.boc.UBC',
                                    'd3m.primitives.feature_extraction.bow.UBC',
                                    'd3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec',
                                    'd3m.primitives.feature_extraction.tfidf_vectorizer.BBNTfidfTransformer'}:
                family = 'NATURAL_LANGUAGE_PROCESSING'

            elif primitive_name in {'d3m.primitives.classification.bert_classifier.DistilBertPairClassification',
                                    'd3m.primitives.classification.text_classifier.DistilTextClassifier'}:
                family = 'TEXT_CLASSIFIER'

            elif primitive_name in {'d3m.primitives.data_transformation.data_cleaning.DistilEnrichDates',
                                    'd3m.primitives.data_cleaning.cleaning_featurizer.DSBOX'}:
                family = 'DATETIME_ENCODER'

            elif primitive_name in {'d3m.primitives.vertex_nomination.seeded_graph_matching.DistilVertexNomination',
                                    'd3m.primitives.classification.gaussian_classification.JHU'}:
                family = 'VERTEX_CLASSIFICATION'

            elif primitive_name in {'d3m.primitives.feature_extraction.yolo.DSBOX'}:
                family = 'OBJECT_DETECTION'

            if family == 'ENCODE_ONE_HOT':
                family = 'CATEGORICAL_ENCODER'

            if family not in primitives:
                primitives[family] = []
            primitives[family].append(primitive_name)

    # Duplicate TEXT_ENCODER primitives for NATURAL_LANGUAGE_PROCESSING family
    primitives['NATURAL_LANGUAGE_PROCESSING'] = primitives['TEXT_ENCODER'] + primitives['NATURAL_LANGUAGE_PROCESSING']

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
