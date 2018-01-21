import logging
import numpy as np

#Import classifiers and sklearn functions
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier
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


# We load the iris-dataset (a widely used benchmark)
iris = datasets.load_iris()
data = iris.data
target = iris.target

# We define a classifier to tune the hyperparameters, currently only svm and Random Forest are accepted
classifier = 'RamdonForestClassifier'

def cfg_from_classifier(classifier):
	# Build Configuration Space which defines all parameters and their ranges
	cs = ConfigurationSpace()

	if classifier == 'svm':

		# We define a few possible types of SVM-kernels and add them as "kernel" to our cs
		kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
		cs.add_hyperparameter(kernel)

		# There are some hyperparameters shared by all kernels
		C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
		shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="true")
		cs.add_hyperparameters([C, shrinking])
		
		# Others are kernel-specific, so we can add conditions to limit the searchspace
		degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)     # Only used by kernel poly
		coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
		cs.add_hyperparameters([degree, coef0])
		use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
		use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
		cs.add_conditions([use_degree, use_coef0])

		# This also works for parameters that are a mix of categorical and values from a range of numbers
		# For example, gamma can be either "auto" or a fixed float
		gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
		gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
		cs.add_hyperparameters([gamma, gamma_value])
		# We only activate gamma_value if gamma is set to "value"
		cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
		# And again we can restrict the use of gamma in general to the choice of the kernel
		cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))
	else:
                #Random Forest hyperparameters
		max_depth = UniformIntegerHyperparameter("max_depth", 1, 10, default_value=4) 
		max_features = UniformFloatHyperparameter("max_features", 0.1, 0.9, default_value=0.1) 
		n_estimators = UniformIntegerHyperparameter("n_estimators", 1, 100, default_value=10)         
		criterion =CategoricalHyperparameter("criterion", ['gini', 'entropy'], default_value='entropy')
		cs.add_hyperparameters([max_depth, max_features, n_estimators, criterion])

	return cs

def main(classifier, data,target):

	def execute(cfg):
		cfg = {k : cfg[k] for k in cfg if cfg[k]}
	    
		if classifier == 'svm':
	    	# We translate boolean values:
			cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
			# And for gamma, we set it to a fixed value or to "auto" (if used)
			if "gamma" in cfg:
				cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
				cfg.pop("gamma_value", None)  # Remove "gamma_value"

			clf = svm.SVC(**cfg, random_state=42)
		else:
			clf = RandomForestClassifier(**cfg)

		scores = cross_val_score(clf, data, target, cv=5)
		return 1-np.mean(scores)  # Minimize!

	#logger = logging.getLogger("SVMExample")
	logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

	cs = cfg_from_classifier(classifier)
	
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


main(classifier,data,target)



