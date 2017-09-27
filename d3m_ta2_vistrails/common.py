import json
import logging
import os.path
import pandas


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
