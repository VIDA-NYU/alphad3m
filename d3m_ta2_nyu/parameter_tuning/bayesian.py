import os
import typing
import importlib
import numpy as np
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, FloatHyperparameter
from smac.facade.smac_facade import SMAC
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from d3m_ta2_nyu.parameter_tuning.estimator_config import primitive_config
from d3m import index


def get_class(name):
    package, classname = name.rsplit('.', 1)
    return getattr(importlib.import_module(package), classname)


def estimator_from_cfg(cfg, name):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    EstimatorClass = get_class(name)
    clf = EstimatorClass(**cfg)
    return clf


def cfg_from_primitives(primitives):
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    for primitive in primitives:
        primitive_config(cs,primitive)
    return cs


def hyperparams_from_cfg(name, cfg):
    PrimitiveClass = index.get_primitive(name)
    HyperparameterClass = typing.get_type_hints(PrimitiveClass.__init__)['hyperparams']
    hyperparameter_config = HyperparameterClass.configuration

    kw_args = {}

    for key in hyperparameter_config:
        cfg_key = name+'|'+key
        if cfg_key in cfg:
            kw_args[key] = cfg[cfg_key]
        else:
            kw_args[key] = hyperparameter_config[key].get_default()

    hy = HyperparameterClass(**kw_args)

    return hy


class HyperparameterTuning(object):
    def __init__(self, primitives):
        self.cs = cfg_from_primitives(primitives)
        # Avoiding too many iterations
        self.runcount = 1
        for param in self.cs.get_hyperparameters():
            if isinstance(param,FloatHyperparameter):
                self.runcount = 100
                break
            elif isinstance(param,CategoricalHyperparameter):
                self.runcount *= len(param.choices)
            else:
                self.runcount *= (param.upper - param.lower)
        self.runcount = min(self.runcount,100)

    def tune(self, runner, wallclock=1800):
        # Scenario object
        if 'TA2_DEBUG_BE_FAST' in os.environ:
            self.runcount = 10
        cutoff = wallclock / (self.runcount / 2) # Allow long pipelines to try to execute half of the iterations limit
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": self.runcount,  # maximum function evaluations
                             "wallclock-limit": wallclock,
                             "cutoff_time": cutoff,
                             "cs": self.cs,  # configuration space
                             "deterministic": "true",
                             "output_dir": "/tmp/"
                             })
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=runner)

        return smac.optimize()
