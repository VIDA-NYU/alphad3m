import numpy as np
import os
import typing
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_facade import SMAC
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from d3m_ta2_nyu.parameter_tuning.estimator_config import ESTIMATORS, primitive_config
from d3m_ta2_nyu.workflow.module_loader import get_class


def cfg_from_estimator(estimator):
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    ESTIMATORS[estimator](cs)
    return cs


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


def primitive_from_cfg(name, cfg):
    PrimitiveClass = get_class(name)
    HyperparameterClass = typing.get_type_hints(PrimitiveClass.__init__)['hyperparams']
    hyperparameter_config = HyperparameterClass.configuration

    kw_args = {}

    for key in hyperparameter_config:
        cfg_key = name+'|'+key
        if cfg_key in cfg:
            kw_args[key] = cfg[cfg_key]
        else:
            kw_args[key] = hyperparameter_config[key].default

    hy = HyperparameterClass(**kw_args)
    primitive = PrimitiveClass(hyperparams=hy)
    return primitive


class HyperparameterTuning(object):
    def __init__(self, primitives):
        if type(primitives) == list:
            self.cs = cfg_from_primitives(primitives)
        else:
            self.cs = cfg_from_estimator(primitives)

    def tune(self, runner):
        # Scenario object
        RUNCOUNT = 100
        if 'TA2_DEBUG_BE_FAST' in os.environ:
            RUNCOUNT = 10
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": RUNCOUNT,  # maximum function evaluations
                             "cs": self.cs,  # configuration space
                             "deterministic": "true",
                             "output_dir": "/tmp/"
                             })
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=runner)

        return smac.optimize()
