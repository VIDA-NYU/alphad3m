from __future__ import division

from vistrails.core.modules.config import ModuleSettings
from vistrails.core.modules.vistrails_module import Module
from vistrails.core.packagemanager import get_package_manager

from dsbox.datapreprocessing.profiler import data_profile


class Profiler(Module):
    """Profile the data"""
    _input_ports = [("data", "basic:List", {'shape': 'circle'})]
    _output_ports = [("result", "basic:String", {'shape': 'circle'})]

    def compute(self):
        #enc = _Encoder()
        if "data" in self.inputPorts:
            result = data_profile(data)
    	self.set_output("result", result)


_modules = [Profiler,]
from vistrails.core.modules.utils import make_modules_dict
_modules = make_modules_dict(_modules,
                             namespace = 'profiling')