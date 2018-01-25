import numpy as np
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_facade import SMAC
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

from d3m_ta2_nyu.parameter_tuning.estimator_config import ESTIMATORS
from d3m_ta2_nyu.workflow.module_loader import get_class


def cfg_from_estimator(estimator):
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    ESTIMATORS[estimator](cs)
    return cs


def cfg_from_pipeline(pipeline):
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()
    ESTIMATORS[pipeline['estimator']](cs)
    return cs


def estimator_from_cfg(cfg, name):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    EstimatorClass = get_class(name)
    clf = EstimatorClass(**cfg)
    return clf


class HyperparameterTuning(object):
    def __init__(self, estimator):
        self.cs = cfg_from_estimator(estimator)

    def tune(self, runner):
        # Scenario object
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": 10,  # maximum function evaluations
                             "cs": self.cs,  # configuration space
                             "deterministic": "true",
                             "output_dir": "/tmp/"
                             })
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=runner)

        return smac.optimize()
