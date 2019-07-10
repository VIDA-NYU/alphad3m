import typing
import importlib
import numpy as np
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, FloatHyperparameter, IntegerHyperparameter, \
    OrdinalHyperparameter
from smac.facade.smac_facade import SMAC
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from d3m_ta2_nyu.parameter_tuning.primitive_config import get_primitive_config
from d3m import index

MAX_RUNS = 10


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def config_from_primitives(primitives):
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    for primitive in primitives:
        get_primitive_config(cs, primitive)

    return cs


def hyperparams_from_config(name, cfg):
    primitive_class = index.get_primitive(name)
    hyperparameter_class = typing.get_type_hints(primitive_class.__init__)['hyperparams']
    hyperparameter_config = hyperparameter_class.configuration

    kw_args = {}

    for key in hyperparameter_config:
        cfg_key = name + '|' + key
        if cfg_key in cfg:
            print('yes', key, cfg[cfg_key])
            kw_args[key] = cfg[cfg_key]
        else:
            print('no', key, hyperparameter_config[key].get_default(), type(hyperparameter_config[key]))
            kw_args[key] = hyperparameter_config[key].get_default()

    hy = hyperparameter_class(**kw_args)

    return hy


class HyperparameterTuning(object):
    def __init__(self, primitives):
        self.cs = config_from_primitives(primitives)
        # Avoiding too many iterations
        self.runcount = 1

        for param in self.cs.get_hyperparameters():
            if isinstance(param, IntegerHyperparameter):
                self.runcount *= (param.upper - param.lower)
            elif isinstance(param, CategoricalHyperparameter):
                self.runcount *= len(param.choices)
            elif isinstance(param, OrdinalHyperparameter):
                self.runcount *= len(param.sequence)
            elif isinstance(param, FloatHyperparameter):
                self.runcount = MAX_RUNS
                break

        self.runcount = min(self.runcount, MAX_RUNS)

    def tune(self, runner, wallclock):
        # Scenario object
        cutoff = wallclock / (self.runcount / 2)  # Allow long pipelines to try to execute half of the iterations limit
        scenario = Scenario({"run_obj": "quality",  # We optimize quality (alternatively runtime)
                             "runcount-limit": self.runcount,  # Maximum function evaluations
                             "wallclock-limit": wallclock,
                             "cutoff_time": cutoff,
                             "cs": self.cs,  # Configuration space
                             "deterministic": "true",
                             "output_dir": "/tmp/"
                             })
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=runner)

        return smac.optimize()
