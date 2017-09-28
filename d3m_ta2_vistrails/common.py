import json
import logging
import math
import os.path
import pandas
import sklearn.metrics


logger = logging.getLogger(__name__)


def _read_file(data_schema, filename, data_type,
               single_column=False):
    if not os.path.exists(filename):
        logger.info("file %s not present; skipping", filename)
        return None

    out = {'index': None}
    data_frame = pandas.read_csv(filename)
    data_column_names = []
    data_index = []
    for feature in data_schema[data_type]:
        if 'index' in feature['varRole']:
            data_index = list(data_frame[feature['varName']])
            out['index'] = data_index
        elif 'attribute' in feature['varRole']:
            data_column_names.append(feature['varName'])
        elif 'target' in feature['varRole']:
            data_column_names.append(feature['varName'])

    data_columns = data_frame.keys()
    for key in data_columns:
        if key not in data_column_names:
            data_frame = data_frame.drop(key, axis=1)

    data_frame = pandas.DataFrame(data=data_frame, index=data_index)
    out['columns'] = data_column_names
    list_ = data_frame.as_matrix()
    if single_column:
        assert list_.shape[1] == 1
        list_ = list_.reshape((-1,))
    out['list'] = list_
    out['frame'] = data_frame
    return out


def read_dataset(data_path):
    output = {}

    with open(os.path.join(data_path, 'dataSchema.json')) as fp:
        data_schema = json.load(fp)

        train_data_file = os.path.join(data_path, 'trainData.csv')
        output['trainData'] = _read_file(
            data_schema['trainData'],
            train_data_file,
            'trainData')

        train_target_file = os.path.join(data_path, 'trainTargets.csv')
        output['trainTargets'] = _read_file(
            data_schema['trainData'],
            train_target_file,
            'trainTargets',
            single_column=True)

        test_data_file = os.path.join(data_path, 'testData.csv')
        output['testData'] = _read_file(
            data_schema['trainData'],
            test_data_file,
            'trainData')

    return output


SCORES_TO_SKLEARN = dict(
    ACCURACY=sklearn.metrics.accuracy_score,
    F1=sklearn.metrics.f1_score,
    F1_MICRO=lambda y_true, y_pred:
        sklearn.metrics.f1_score(y_true, y_pred, average='micro'),
    F1_MACRO=lambda y_true, y_pred:
        sklearn.metrics.f1_score(y_true, y_pred, average='macro'),
    ROC_AUC=sklearn.metrics.roc_auc_score,
    ROC_AUC_MICRO=lambda y_true, y_pred:
        sklearn.metrics.roc_auc_score(y_true, y_pred, average='micro'),
    ROC_AUC_MACRO=lambda y_true, y_pred:
        sklearn.metrics.roc_auc_score(y_true, y_pred, average='macro'),
    MEAN_SQUARED_ERROR=sklearn.metrics.mean_squared_error,  # FIXME: Not in gRPC?
    ROOT_MEAN_SQUARED_ERROR=lambda y_true, y_pred:
        math.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred)),
    # FIXME: ROOT_MEAN_SQUARED_ERROR_AVG // sum(mean_squared_error_list)/len(mean_squared_error_list)
    # FIXME: MEAN_ABSOLUTE_ERROR // sklearn.metrics.mean_absolute_error
    R_SQUARED=sklearn.metrics.r2_score,
    EXECUTION_TIME=None,
)

SCORES_FROM_SCHEMA = {
    'accuracy': 'ACCURACY',
    'f1': 'F1',
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
}
