import logging
import numpy as np
import importlib

#Import classifiers and sklearn functions
from sklearn import datasets

from sklearn.model_selection import cross_val_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

from d3m_ta2_nyu.parameter_tuning.estimator_config import ESTIMATORS

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

def estimator_from_cfg(cfg):
	cfg = {k : cfg[k] for k in cfg if cfg[k]}
	estimator = pipeline['estimator']
	EstimatorClass = getattr(importlib.import_module(estimator[:estimator.rfind('.')]), estimator[estimator.rfind('.') + 1:])
	clf = EstimatorClass(**cfg)
	return clf


class HyperparameterTuning(object):
	def __init__(self,estimator):
		self.cs = cfg_from_estimator(estimator) 
		

	def tune(self,runner):
		# Scenario object
		scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
	                     "runcount-limit": 100,  # maximum function evaluations
	                     "cs": self.cs,               # configuration space
	                     "deterministic": "true"
	                     })
		smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
	        tae_runner=runner)

		return smac.optimize()






def main(pipeline, data,target):

	def execute(cfg):
		clf = estimator_from_cfg(cfg)
		scores = cross_val_score(clf, data, target, cv=5)
		return 1-np.mean(scores)  # Minimize!

	#logger = logging.getLogger("SVMExample")
	logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

	cs = cfg_from_pipeline(pipeline)
	
	# Scenario object
	scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
	                     "runcount-limit": 200,  # maximum function evaluations
	                     "cs": cs,               # configuration space
	                     "deterministic": "true"
	                     })


	# Optimize, using a SMAC-object
	print("Optimizing! Depending on your machine, this might take a few minutes.")
	smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
	        tae_runner=execute)

	# Example call of the function
	# It returns: Status, Cost, Runtime, Additional Infos
	def_value = execute(cs.get_default_configuration())
	print("Default Value: %.2f" % (def_value))

	incumbent = smac.optimize()

	inc_value = execute(incumbent)

	print("Optimized Value: %.2f" % (inc_value))


# We load the iris-dataset (a widely used benchmark)
iris = datasets.load_iris()
data = iris.data
target = iris.target

pipeline = {}
#only testing classifiers for now
for key in list(ESTIMATORS.keys())[:6]:
	pipeline['estimator'] = key
	main(pipeline,data,target)



