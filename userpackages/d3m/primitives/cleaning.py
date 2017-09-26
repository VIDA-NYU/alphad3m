from __future__ import division

from vistrails.core.modules.config import ModuleSettings
from vistrails.core.modules.vistrails_module import Module
from vistrails.core.packagemanager import get_package_manager

from dsbox.datapreprocessing.cleaner import Encoder as _Encoder
from dsbox.datapreprocessing.cleaner import Imputation as _Imputation

from sklearn.metrics import SCORERS

class Encoder(Module):
    """ DSBox Data Encoder

    A one-hot encoder, which
    
    1. is given rules or values to identify categorical columns/
       - categorical_features:
            '95in10': is category if 95% of the column fall into 10 values.
       - tex2int: if non-categorical text/string columns be mapped to integers
       - n_limit: max number of distinct values to one-hot encode,
         remaining values with fewer occurence are put in [colname]_other_ column.
    
    2. uses data given in fit() function to tune the encoder.
    
    3. transform(): input data to be encoded and output the result.
    
    Team: USC ISI
    Version: 0.1.2
    Source Code: https://github.com/usc-isi-i2/dsbox-cleaning.git

    Parameters
    -------
    n_limit: int (default=10)
       Max number of distinct values to one-hot encode
       is_hyperparameter=true
    
    text2int: boolean (default=false)
       Encode text strings as integers. For example, set to true if text strings encode categorical data, and set to false if text strings contain natural language text
       is_hyperparameter=true

    """
    _input_ports = [("data", "basic:List", {'shape': 'circle'}),
                   ('text2int','(org.vistrails.vistrails.basic:Boolean)', {'optional': True, 'defaults': "['False']"}),
                   ('n_limit','(org.vistrails.vistrails.basic:Integer)', {'optional': True, 'defaults': "['10']"})]
    _output_ports = [("model", "sklearn:Estimator", {'shape': 'diamond'})]

    def compute(self):
    	if "text2int" in self.inputPorts:
            _text2int = self.get_input("text2int")
        else:
        	_text2int = False
        if "n_limit" in self.inputPorts:
    	    _n_limit = self.get_input("n_limit")
    	else:
    		_n_limit = 10
    	
        enc = _Encoder(text2int = _text2int, n_limit = _n_limit)
        if "data" in self.inputPorts:
            enc.fit(self.get_input("data"))
    	self.set_output("model", enc)


class Imputation(Module):
    """Profile the data"""
    _input_ports = [("data", "basic:List", {'shape': 'square'}),
                    ("target", "basic:List", {'optional': True,'shape': 'square'}),
                    ("model", "sklearn:Estimator", {'shape': 'diamond'}),
                    ("metric", "basic:String", {"defaults": ["accuracy"]}),
                    ("greater_is_better", "basic:Boolean", {"defaults": [True]}),
                    ("strategy", "basic:String", {"defaults": ["iteratively_regre"]}),]

    _output_ports = [("model", "sklearn:Estimator", {'shape': 'diamond'})]

    def compute(self):

        scorer = SCORERS[self.get_input("metric")]
        imputation = _Imputation(model=self.get_input("model"),
                                scorer=scorer, strategy=self.get_input("strategy"),
                                greater_is_better=self.get_input("greater_is_better"))
        if "target" in self.inputPorts:
            imputation.fit(self.get_input("data"),self.get_input("target"))
        else:
            imputation.fit(self.get_input("data"))
        
    	self.set_output("model",imputation)

'''
class Discretizer(Module)
	"""Profile the data"""
	_input_ports = [("data", "basic:List", {'shape': 'circle'})]
    _output_ports = [("result", "basic:String", {'shape': 'circle'})]

    def compute(self):
        #enc = _Encoder()
        
        if "data" in self.inputPorts:
            result = ""
            #result = profile_data(data)
    	self.set_output("result", result)
'''


_modules = [Encoder, Imputation,]

from vistrails.core.modules.utils import make_modules_dict
_modules = make_modules_dict(_modules,
                             namespace = 'cleaning')
