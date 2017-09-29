try:
    import cPickle as pickle
except ImportError:
    import pickle
import os.path
from vistrails.core.modules.vistrails_module import Module, ModuleError

from . import configuration


class Persist(Module):
    _input_ports = [('value', '(org.vistrails.vistrails.basic:Variant)')]
    _output_ports = [('value', '(org.vistrails.vistrails.basic:Variant)')]

    def compute(self):
        module_id = self.moduleInfo['moduleId']

        if self.has_input('value'):
            value = self.get_input('value')
            self.set_output('value', value)

            if configuration.has('file_store'):
                path = os.path.join(configuration.file_store,
                                    '%d.pkl' % module_id)
                with open(path, 'wb') as fp:
                    pickle.dump(value, fp)
        else:
            if not configuration.has('file_store'):
                raise ModuleError(self, "No input and no path to load from")

            path = os.path.join(configuration.file_store,
                                '%d.pkl' % module_id)
            with open(path, 'rb') as fp:
                value = pickle.load(fp)
            self.set_output('value', value)


class Internal(Module):
    _output_ports = [('value', '(org.vistrails.vistrails.basic:Variant)')]

    values = {}

    def compute(self):
        module_id = self.moduleInfo['moduleId']

        self.set_output('value', Internal.values[module_id])


_modules = [Persist, Internal]
