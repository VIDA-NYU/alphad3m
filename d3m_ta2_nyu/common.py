"""Some utilities specific to the D3M project.
"""

import logging
import math
import sklearn.metrics


logger = logging.getLogger(__name__)

SCORES_FROM_SCHEMA = {
    'accuracy': 'ACCURACY',
    'f1': 'F1',
    'precision': 'PRECISION',
    'recall': 'RECALL',
    'f1Micro': 'F1_MICRO',
    'f1Macro': 'F1_MACRO',
    'rocAuc': 'ROC_AUC',
    'rocAucMicro': 'ROC_AUC_MICRO',
    'rocAucMacro': 'ROC_AUC_MACRO',
    'meanSquaredError': 'MEAN_SQUARED_ERROR',
    'rootMeanSquaredError': 'ROOT_MEAN_SQUARED_ERROR',
    'meanAbsoluteError': 'MEAN_ABSOLUTE_ERROR',
    'rSquared': 'R_SQUARED',
    'normalizedMutualInformation': 'NORMALIZED_MUTUAL_INFORMATION',
    'jaccardSimilarityScore': 'JACCARD_SIMILARITY_SCORE',
    'objectDetectionAP': 'OBJECT_DETECTION_AVERAGE_PRECISION',
    'averageMeanReciprocalRank': 'AVERAGE_MEAN_RECIPROCAL_RANK'
    # 'precisionAtTopK': 'PRECISION_AT_TOP_K',
}

SCORES_TO_SCHEMA = {v: k for k, v in SCORES_FROM_SCHEMA.items()}

# 1 if lower values of that metric indicate a better classifier, -1 otherwise
SCORES_RANKING_ORDER = dict(
    ACCURACY=-1,
    F1=-1,
    PRECISION=-1,
    OBJECT_DETECTION_AVERAGE_PRECISION=-1,
    RECALL=-1,
    F1_MICRO=-1,
    F1_MACRO=-1,
    ROC_AUC=-1,
    ROC_AUC_MICRO=-1,
    ROC_AUC_MACRO=-1,
    MEAN_SQUARED_ERROR=1,
    ROOT_MEAN_SQUARED_ERROR=1,
    MEAN_ABSOLUTE_ERROR=1,
    R_SQUARED=-1,
    NORMALIZED_MUTUAL_INFORMATION=-1,
    JACCARD_SIMILARITY_SCORE=-1,
    EXECUTION_TIME=1,
)

TASKS_FROM_SCHEMA = {
    'classification': 'CLASSIFICATION',
    'regression': 'REGRESSION',
    'clustering': 'CLUSTERING',
    'linkPrediction': 'LINK_PREDICTION',
    'vertexNomination': 'VERTEX_NOMINATION',
    'communityDetection': 'COMMUNITY_DETECTION',
    'graphClustering': 'GRAPH_CLUSTERING',
    'graphMatching': 'GRAPH_MATCHING',
    'timeSeriesForecasting': 'TIME_SERIES_FORECASTING',
    'collaborativeFiltering': 'COLLABORATIVE_FILTERING',
    'objectDetection': 'OBJECT_DETECTION',
    'vertexClassification': 'VERTEX_CLASSIFICATION',
    'semiSupervisedClassification': 'SEMISUPERVISED_CLASSIFICATION',
    'semiSupervisedRegression': 'SEMISUPERVISED_REGRESSION'
}


def normalize_score(metric, score, order):
    """Normalize the score to a value between 0 and 1.

    :param order: Either ``"asc"`` (higher the better) or ``"desc"`` (lower the
        better).
    """
    order_mult = dict(asc=1.0, desc=-1.0)[order]
    order_mult *= SCORES_RANKING_ORDER[metric]
    try:
        return 1.0 / (1.0 + math.exp(order_mult * score))
    except ArithmeticError:  # OverflowError can happen with weird scores
        return dict(asc=0.0, desc=1.0)[order]


def format_metrics(problem):
    metrics = []

    for metric in problem['problem']['performance_metrics']:
        metric_name = metric['metric'].name
        formatted_metric = {'metric': metric_name}

        if len(metric) > 1:  # Metric has parameters
            formatted_metric['params'] = {}
            for param in metric.keys():
                if param != 'metric':
                    formatted_metric['params'][param] = metric[param]

        metrics.append(formatted_metric)

    return metrics
