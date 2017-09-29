import json
import logging
import math
import os.path
import pandas
from skimage import io
import numpy as np
from sklearn.decomposition import IncrementalPCA
import sklearn.metrics


logger = logging.getLogger(__name__)


safe_shell_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                       "abcdefghijklmnopqrstuvwxyz"
                       "0123456789"
                       "-+=/:.,%_")


def shell_escape(s):
    r"""Given bl"a, returns "bl\\"a".
    """
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    if not s or any(c not in safe_shell_chars for c in s):
        return '"%s"' % (s.replace('\\', '\\\\')
                          .replace('"', '\\"')
                          .replace('`', '\\`')
                          .replace('$', '\\$'))
    else:
        return s


def _read_file(data_schema, filename, data_type,
               single_column=False):
    if not os.path.exists(filename):
        logger.info("file %s not present; skipping", filename)
        return None

    out = {'index': None, 'image': False, 'categorical': False}    
    data_frame = pandas.read_csv(filename)
    data_column_names = []
    data_index = []
    file_data = {}    
    for feature in data_schema[data_type]:
        try:
            if 'index' in feature['varRole']:
                data_index = list(data_frame[feature['varName']])
                out['index'] = data_index
            elif 'attribute' in feature['varRole']:
                data_column_names.append(feature['varName'])
                out['categorical'] = out['categorical'] or (feature['varType'] == 'categorical')
            elif 'target' in feature['varRole']:
                data_column_names.append(feature['varName'])
        except KeyError:
            if 'file' in feature['varType']:
                file_data['column_name'] = feature['varName']
                file_data['fileType'] = feature['varFileType']
                file_data['fileFormat'] = feature['varFileFormat']

    data_frame_copy = data_frame.copy()
    data_columns = data_frame_copy.keys()
    for key in data_columns:
        if key not in data_column_names:
            data_frame_copy = data_frame_copy.drop(key, axis=1)

    raw_data_np = np.empty([len(data_frame_copy.values), 0])
    if file_data:
        data_path = os.path.dirname(os.path.abspath(filename))
        file_names = [os.path.join(data_path, 'raw_data', file_name) for file_name in data_frame[file_data['column_name']]]
        
        if file_data['fileType'] == 'image':
            out['image'] = True
            count = 0
            ipca = IncrementalPCA(n_components=2, batch_size=10)
            training_data = []
            raw_data_np = None
            for image in file_names:
                image_out = read_image_file(image)
                training_data.append(image_out['image_array'])
                sample = np.array(image_out['image_array']).reshape(1, -1)
                if count > 7:
                    if raw_data_np is None:
                        raw_data_np = ipca.fit_transform(np.array(training_data))
                    else:
                        raw_data_np = np.vstack((raw_data_np, ipca.transform(sample)))
                count = count + 1
            
    result_data = data_frame_copy.values
    
    if len(raw_data_np) > 0:
        if len(result_data) > 0:
            result_data = np.hstack((result_data, raw_data_np))
        else:
            result_data =  raw_data_np
            
    if (not np.shape(result_data)[1] == len(data_columns)) or out['image']:
        final_data_frame = pandas.DataFrame(data=result_data, index=data_index)
    else:
        final_data_frame = pandas.DataFrame(data=result_data, columns=data_column_names, index=data_index)
    out['columns'] = data_column_names
    list_ = final_data_frame.as_matrix()
    if single_column:
        assert list_.shape[1] == 1
        list_ = list_.reshape((-1,))
    out['list'] = list_
    out['frame'] = final_data_frame
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


def read_image_file(filename):
    out = {}
    image_ndarray = io.imread(filename)
    image_2darray = image_ndarray.reshape((len(image_ndarray), -1))
    out['image_array'] = image_2darray.reshape((1, -1))[0]
    return out


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

# 1 if lower values of that metric indicate a better classifier, -1 otherwise
SCORES_RANKING_ORDER = dict(
    ACCURACY=-1,
    F1=-1,
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
    EXECUTION_TIME=1,
)
