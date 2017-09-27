from __future__ import division

from vistrails.core.modules.vistrails_module import Module

import pandas as pd
import json
import os.path


class ReadDataSchema(Module):
    """ Read the D3M data schema and parse the corresponding trainData, trainTargets and testData
    """
    _input_ports = [('data_path', '(org.vistrails.vistrails.basic:Directory)')]
    _output_ports = [
        ('trainData_index', '(org.vistrails.vistrails.basic:List)'),
        ('trainData_columns', '(org.vistrails.vistrails.basic:List)'),
        ('trainData_list',  '(org.vistrails.vistrails.basic:List)'), ('trainData_frame', '(org.vistrails.vistrails.basic:List)'),
        ('trainTargets_index', '(org.vistrails.vistrails.basic:List)'),
        ('trainTargets_columns', '(org.vistrails.vistrails.basic:List)'),
        ('trainTargets_list',  '(org.vistrails.vistrails.basic:List)'), ('trainTargets_frame', '(org.vistrails.vistrails.basic:List)'),
        ('testData_index', '(org.vistrails.vistrails.basic:List)'),
        ('testData_columns', '(org.vistrails.vistrails.basic:List)'),
        ('testData_list',  '(org.vistrails.vistrails.basic:List)'), ('testData_frame', '(org.vistrails.vistrails.basic:List)'),
    ]

    def _readFile(self, data_schema, filename, data_type, output_type,
                  single_column=False):
        try:
            data_frame = pd.read_csv(filename)
            data_column_names = []
            data_index = []
            for feature in data_schema[data_type]:
                if 'index' in feature['varRole']:
                    data_index = list(data_frame[feature['varName']])
                    self.set_output(output_type + '_index', data_index)
                elif 'attribute' in feature['varRole']:
                    data_column_names.append(feature['varName'])
                elif 'target' in feature['varRole']:
                    data_column_names.append(feature['varName'])

            data_columns = data_frame.keys()
            for key in data_columns:
                if key not in data_column_names:
                    data_frame = data_frame.drop(key, axis=1)

            data_frame = pd.DataFrame(data=data_frame, index=data_index)
            self.set_output(output_type + '_columns', data_column_names)
            list_ = data_frame.as_matrix()
            if single_column:
                assert list_.shape[1] == 1
                list_ = list_.reshape((-1,))
            self.set_output(output_type + '_list', list_)
            self.set_output(output_type + '_frame', data_frame)
        except IOError:
            print output_type + " not found"

    def compute(self):
        data_path = self.get_input('data_path').name
        with open(os.path.join(data_path, 'dataSchema.json')) as fp:
            data_schema = json.load(fp)

            train_data_file = os.path.join(data_path, 'trainData.csv')
            self._readFile(data_schema['trainData'], train_data_file,
                           'trainData', 'trainData')

            train_target_file = os.path.join(data_path, 'trainTargets.csv')
            self._readFile(data_schema['trainData'], train_target_file,
                           'trainTargets', 'trainTargets',
                           single_column=True)

            test_data_file = os.path.join(data_path, 'testData.csv')
            self._readFile(data_schema['trainData'], test_data_file,
                           'trainData', 'testData')


_modules = [ReadDataSchema]
