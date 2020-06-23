import logging
import numpy as np
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import IntegerHyperparameter, FloatHyperparameter, CategoricalHyperparameter, \
    OrdinalHyperparameter
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from d3m_ta2_nyu.parameter_tuning.primitive_config import load_primitive_configspace, load_hyperparameters


MAX_RUNS = 100
logger = logging.getLogger(__name__)


def build_configspace(primitives):
    # Build Configuration Space which defines all parameters and their ranges
    configspace = ConfigurationSpace()
    for primitive in primitives:
        load_primitive_configspace(configspace, primitive)

    return configspace


def get_new_hyperparameters(primitive_name, configspace):
    hyperparameters = load_hyperparameters(primitive_name)
    new_hyperparameters = {}

    for hyperparameter_name in hyperparameters:
        hyperparameter_config_name = primitive_name + '|' + hyperparameter_name
        if hyperparameter_config_name in configspace:
            new_hyperparameters[hyperparameter_name] = configspace[hyperparameter_config_name]
            logger.info('New value for %s=%s', hyperparameter_config_name, new_hyperparameters[hyperparameter_name])

    return new_hyperparameters


class HyperparameterTuning(object):
    def __init__(self, primitives):
        self.configspace = build_configspace(primitives)
        # Avoiding too many iterations
        self.runcount = 1

        for param in self.configspace.get_hyperparameters():
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

    def tune(self, runner, wallclock, output_dir):
        # Scenario object
        cutoff = wallclock / (self.runcount / 10)  # Allow long pipelines to try to execute one fourth of the iterations limit
        scenario = Scenario({"run_obj": "quality",  # We optimize quality (alternatively runtime)
                             "runcount-limit": self.runcount,  # Maximum function evaluations
                             "wallclock-limit": wallclock,
                             "cutoff_time": cutoff,
                             "cs": self.configspace,  # Configuration space
                             "deterministic": "true",
                             "output_dir": output_dir,
                             "abort_on_first_run_crash": False
                             })
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=runner)

        return smac.optimize()
