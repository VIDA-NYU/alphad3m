"""Some utilities specific to the D3M project.
"""

import logging
import math

logger = logging.getLogger(__name__)

# 1 if lower values of that metric indicate a better classifier, -1 otherwise
SCORES_RANKING_ORDER = dict(
    ACCURACY=-1,
    PRECISION=-1,
    RECALL=-1,
    F1=-1,
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
    PRECISION_AT_TOP_K=1,
    OBJECT_DETECTION_AVERAGE_PRECISION=-1,
    HAMMING_LOSS=1,
    EXECUTION_TIME=1,
)


def normalize_score(metric, score, order):
    """Normalize the score to a value between 0 and 1.

    :param order: Either ``"asc"`` (higher the better) or ``"desc"`` (lower the
        better).
    """
    order_mult = dict(asc=1.0, desc=-1.0)[order]
    order_mult *= SCORES_RANKING_ORDER.get(metric, 1)
    try:
        return 1.0 / (1.0 + math.exp(order_mult * score))
    except ArithmeticError:  # OverflowError can happen with weird scores
        return dict(asc=0.0, desc=1.0)[order]
