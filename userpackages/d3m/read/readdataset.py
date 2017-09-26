###############################################################################
##
## Copyright (C) 2014-2016, New York University.
## Copyright (C) 2013-2014, NYU-Poly.
## All rights reserved.
## Contact: contact@vistrails.org
##
## This file is part of VisTrails.
##
## "Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimer.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimer in the
##    documentation and/or other materials provided with the distribution.
##  - Neither the name of the New York University nor the names of its
##    contributors may be used to endorse or promote products derived from
##    this software without specific prior written permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
## THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
## PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
## CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
## EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
## PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
## OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
## WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
## OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
##
###############################################################################

from __future__ import division

from vistrails.core.modules.basic_modules import ListType
from vistrails.core.modules.vistrails_module import Module, ModuleError

from vistrails.packages.tabledata.common import Table, TableObject, InternalModuleError

import pandas as pd
import json
import os.path
import numpy as np

class ReadDataSchema(Module):
    """ Read the D3M data schema and parse the corresponding trainData, trainTargets and testData
    """
    _input_ports = [('data_path', '(org.vistrails.vistrails.basic:String)')]
    _output_ports = [('trainData_index', '(org.vistrails.vistrails.basic:List)'),
                     ('trainData_columns', '(org.vistrails.vistrails.basic:List)'),
                     ('trainData_table', Table), ('trainData_frame', '(org.vistrails.vistrails.basic:List)'),
                     ('trainTargets_index', '(org.vistrails.vistrails.basic:List)'),
                     ('trainTargets_columns', '(org.vistrails.vistrails.basic:List)'),
                     ('trainTargets_table', Table), ('trainTargets_frame', '(org.vistrails.vistrails.basic:List)'),
                     ('testData_index', '(org.vistrails.vistrails.basic:List)'),
                     ('testData_columns', '(org.vistrails.vistrails.basic:List)'),
                     ('testData_table', Table), ('testData_frame', '(org.vistrails.vistrails.basic:List)')
    ]
    def _readFile(self, data_schema, filename, data_type='trainData',output_type='trainData'):
        try:
            data_frame = pd.read_csv(filename)
            data_column_names = []
            data_index = []
            for feature in data_schema[data_type]:
                if 'index' in feature['varRole']:
                    data_index = list(data_frame[feature['varName']])
                    self.set_output(output_type+'_index', data_index)
                elif 'attribute' in feature['varRole']:
                    data_column_names.append(feature['varName'])
                elif 'target' in feature['varRole']:
                    data_column_names.append(feature['varName'])

            data_columns = data_frame.keys()
            for key in data_columns:
                if key not in data_column_names:
                    data_frame = data_frame.drop(key, axis=1)
        
            data_frame = pd.DataFrame(data=data_frame, index=data_index)
            self.set_output(output_type+'_columns', data_column_names)
            self.set_output(output_type+'_table', data_frame.values)
            self.set_output(output_type+'_frame', data_frame)
        except IOError:
            print output_type + " not found"


    def compute(self):
        data_path = self.get_input('data_path')
        with open(os.path.join(data_path, 'dataSchema.json')) as fp:
            data_schema = json.load(fp)

            train_data_file =  os.path.join(data_path, 'trainData.csv')
            self._readFile(data_schema['trainData'], train_data_file)
            
            train_target_file =  os.path.join(data_path, 'trainTargets.csv')
            self._readFile(data_schema['trainData'], train_target_file, 'trainTargets', 'trainTargets')

            test_data_file =  os.path.join(data_path, 'testData.csv')
            self._readFile(data_schema['trainData'], test_data_file, 'trainData', 'testData')
            
_modules = [ReadDataSchema]

###############################################################################
