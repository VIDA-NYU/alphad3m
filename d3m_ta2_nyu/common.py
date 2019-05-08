"""Some utilities specific to the D3M project.
"""

import logging
import math
import sklearn.metrics


logger = logging.getLogger(__name__)


def _root_mean_squared_error_avg(y_true, y_pred):
    l2_sum = 0
    count = 0
    for t, p in zip(y_true, y_pred):
        l2_sum += math.sqrt(sklearn.metrics.mean_squared_error(t, p))
        count += 1
    return l2_sum / count


SCORES_TO_SKLEARN = dict(
    ACCURACY=sklearn.metrics.accuracy_score,
    PRECISION=lambda y_true, y_pred:
              sklearn.metrics.precision_score(y_true, y_pred, average='micro'),
    RECALL=lambda y_true, y_pred:
              sklearn.metrics.recall_score(y_true, y_pred, average='micro'),
    F1=lambda y_true, y_pred:
        sklearn.metrics.f1_score(y_true, y_pred,
                                 average='binary', pos_label=1),
    F1_MICRO=lambda y_true, y_pred:
        sklearn.metrics.f1_score(y_true, y_pred, average='micro'),
    F1_MACRO=lambda y_true, y_pred:
        sklearn.metrics.f1_score(y_true, y_pred, average='macro'),
    ROC_AUC=sklearn.metrics.roc_auc_score,
    ROC_AUC_MICRO=lambda y_true, y_pred:
        sklearn.metrics.roc_auc_score(y_true, y_pred, average='micro'),
    ROC_AUC_MACRO=lambda y_true, y_pred:
        sklearn.metrics.roc_auc_score(y_true, y_pred, average='macro'),
    MEAN_SQUARED_ERROR=sklearn.metrics.mean_squared_error,
    ROOT_MEAN_SQUARED_ERROR=lambda y_true, y_pred:
        math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred)),
    ROOT_MEAN_SQUARED_ERROR_AVG=_root_mean_squared_error_avg,
    MEAN_ABSOLUTE_ERROR=sklearn.metrics.mean_absolute_error,
    R_SQUARED=sklearn.metrics.r2_score,
    NORMALIZED_MUTUAL_INFORMATION=sklearn.metrics.normalized_mutual_info_score,
    # FIXME: JACCARD_SIMILARITY_SCORE
    EXECUTION_TIME=None,
)

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
    'rootMeanSquaredErrorAvg': 'ROOT_MEAN_SQUARED_ERROR_AVG',
    'meanAbsoluteError': 'MEAN_ABSOLUTE_ERROR',
    'rSquared': 'R_SQUARED',
    'normalizedMutualInformation': 'NORMALIZED_MUTUAL_INFORMATION',
    'jaccardSimilarityScore': 'JACCARD_SIMILARITY_SCORE',
    'objectDetectionAP': 'OBJECT_DETECTION_AVERAGE_PRECISION'
    # 'precisionAtTopK': 'PRECISION_AT_TOP_K',
}

SCORES_TO_SCHEMA = {v: k for k, v in SCORES_FROM_SCHEMA.items()}

# 1 if lower values of that metric indicate a better classifier, -1 otherwise
SCORES_RANKING_ORDER = dict(
    ACCURACY=-1,
    F1=-1,
    PRECISION=-1,
    RECALL=-1,
    F1_MICRO=-1,
    F1_MACRO=-1,
    ROC_AUC=-1,
    ROC_AUC_MICRO=-1,
    ROC_AUC_MACRO=-1,
    MEAN_SQUARED_ERROR=1,
    ROOT_MEAN_SQUARED_ERROR=1,
    ROOT_MEAN_SQUARED_ERROR_AVG=1,
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
    'objectDetection': 'OBJECT_DETECTION'
}

TASKS_TO_SCHEMA = {v: k for k, v in TASKS_FROM_SCHEMA.items()}

SUBTASKS_FROM_SCHEMA = {
    'none': 'NONE',
    'binary': 'BINARY',
    'multiClass': 'MULTICLASS',
    'multiLabel': 'MULTILABEL',
    'univariate': 'UNIVARIATE',
    'multivariate': 'MULTIVARIATE',
    'overlaping': 'OVERLAPPING',
    'nonOverlapping': 'NONOVERLAPPING',
}

SUBTASKS_TO_SCHEMA = {v: k for k, v in SUBTASKS_FROM_SCHEMA.items()}


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
