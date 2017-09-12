from __future__ import division

from vistrails.core.modules.config import ModuleSettings
from vistrails.core.modules.vistrails_module import Module
from vistrails.core.packagemanager import get_package_manager

from dsbox.datapreprocessing.profiler import Profiler as _Profiler


class Profiler(Module):
    """Profile the data"""
    _input_ports = [("data", "basic:String", {'shape': 'square'}),("dataFrame", "basic:List", {'shape': 'circle'})]
    _output_ports = [("result", "basic:String", {'shape': 'square'})]

    def compute(self):
        profiler = _Profiler()
        if "data" in self.inputPorts:
            result = profiler.profile_data(self.get_input("data"))
        elif "dataFrame" in self.inputPorts:
            result = profiler.profile_data(self.get_input("dataFrame"))
        else:
            result = ""
        self.set_output("result",result)


_modules = [Profiler,]
from vistrails.core.modules.utils import make_modules_dict
_modules = make_modules_dict(_modules,
                             namespace = 'profiling')