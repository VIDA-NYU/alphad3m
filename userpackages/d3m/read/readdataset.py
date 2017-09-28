from __future__ import division

from vistrails.core.modules.vistrails_module import Module

import pandas as pd
import json
import os.path

from d3m_ta2_vistrails.common import read_dataset


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

    def compute(self):
        data = read_dataset(self.get_input('data_path').name)
        for output_type in ['trainData', 'trainTargets', 'testData']:
            out = data.get(output_type)
            if out is None:
                continue
            self.set_output(output_type + '_index', out['index'])
            self.set_output(output_type + '_columns', out['columns'])
            self.set_output(output_type + '_list', out['list'])
            self.set_output(output_type + '_frame', out['frame'])


_modules = [ReadDataSchema]
