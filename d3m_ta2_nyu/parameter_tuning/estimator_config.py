import typing
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
     UniformFloatHyperparameter, UniformIntegerHyperparameter, \
     NormalFloatHyperparameter, FloatHyperparameter, IntegerHyperparameter
from d3m import index
from d3m.metadata.hyperparams import Bounded, Enumeration, Uniform, UniformInt, Normal


PRIMITIVES = index.search()

def primitive_config(cs, primitive_name):
    PrimitiveClass = index.get_primitive(primitive_name)
    HyperparameterClass = typing.get_type_hints(PrimitiveClass.__init__)['hyperparams']
    if HyperparameterClass:
        config = HyperparameterClass.configuration
        parameter_list = []
        for p in config:
            parameter_name = primitive_name + '|' + p
            if isinstance(config[p], Bounded):
                lower = config[p].lower
                upper = config[p].upper
                default = config[p].get_default()
                if type(default) == int:
                    cs_param = UniformIntegerHyperparameter(parameter_name, lower, upper, default_value=default)
                else:
                    cs_param = UniformFloatHyperparameter(parameter_name, lower, upper, default_value=default)
                parameter_list.append(cs_param)
            elif isinstance(config[p], Uniform):
                lower = config[p].lower
                upper = config[p].upper
                default = config[p].get_default()
                cs_param = UniformFloatHyperparameter(parameter_name, lower, upper, default_value=default)
                parameter_list.append(cs_param)
            elif isinstance(config[p], UniformInt):
                lower = config[p].lower
                upper = config[p].upper
                default = config[p].get_default()
                cs_param = UniformIntegerHyperparameter(parameter_name, lower, upper, default_value=default)
                parameter_list.append(cs_param)
            elif isinstance(config[p], Normal):
                lower = config[p].lower
                upper = config[p].upper
                default = config[p].get_default()
                cs_param = NormalFloatHyperparameter(parameter_name, lower, upper, default_value=default)
                parameter_list.append(cs_param)
            elif isinstance(config[p], Enumeration):
                values = config[p].values
                default = config[p].get_default()
                cs_param = CategoricalHyperparameter(parameter_name, values, default_value=default)
                parameter_list.append(cs_param)
        cs.add_hyperparameters(parameter_list)


def get_random_hyperparameters(estimator):
    pass

def get_default_hyperparameters(estimator):
    pass

def encode_hyperparameter(estimator,parameter,value):
    pass


def decode_hyperparameter(estimator,parameter,value):
    pass


def is_estimator(name):
    if name not in PRIMITIVES:
        return False
    klass = index.get_primitive(name)
    family = klass.metadata.to_json_structure()['primitive_family']
    return family == 'CLASSIFICATION' or family == 'REGRESSION'

