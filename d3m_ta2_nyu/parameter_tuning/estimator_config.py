import logging
import numpy as np


# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition




def svm_config(cs):

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

def random_forest_config(cs):
	#Random Forest hyperparameters
	max_depth = UniformIntegerHyperparameter("max_depth", 1, 10, default_value=4) 
	max_features = UniformFloatHyperparameter("max_features", 0.1, 0.9, default_value=0.1) 
	n_estimators = UniformIntegerHyperparameter("n_estimators", 1, 100, default_value=10)         
	criterion =CategoricalHyperparameter("criterion", ['gini', 'entropy'], default_value='entropy')
	cs.add_hyperparameters([max_depth, max_features, n_estimators, criterion])


def svc_config(cs):
	penalty = CategoricalHyperparameter("penalty", ['l1', 'l2'], default_value='l2')
	loss =CategoricalHyperparameter("loss", ['hinge', 'squared_hinge'], default_value='squared_hinge')
	C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
	tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-4)
	max_iter = UniformIntegerHyperparameter("max_iter", 1, 2000, default_value=1000)
	dual = UniformIntegerHyperparameter("dual", 0, 1, default_value=1)
	cs.add_hyperparameters([penalty,loss, C, tol, max_iter, dual])



def kneighbors_config(cs):
	n_neighbors = UniformIntegerHyperparameter("n_neighbors", 1, 10, default_value=5)
	weights = CategoricalHyperparameter("weights", ['uniform', 'distance'], default_value='uniform')
	algorithm = CategoricalHyperparameter("algorithm", ['auto','ball_tree', 'kd_tree', 'brute'], default_value='auto')
	leaf_size = UniformIntegerHyperparameter("leaf_size", 10, 50, default_value=30)
	p = UniformIntegerHyperparameter("p", 1, 2, default_value=2)
	cs.add_hyperparameters([n_neighbors, weights, algorithm, leaf_size, p])

def decision_tree_config(cs):
	criterion = CategoricalHyperparameter("criterion", ['gini', 'entropy'], default_value='gini')
	splitter = CategoricalHyperparameter("splitter", ['best', 'random'], default_value='best')
	max_depth = UniformIntegerHyperparameter("max_depth", 1, 10, default_value=None)
	min_samples_split = UniformFloatHyperparameter("min_samples_split", 0.1, 0.9, default_value=0.1)
	min_samples_leaf = UniformFloatHyperparameter("min_samples_leaf", 0.1, 0.9, default_value=0.1)
	max_features = UniformFloatHyperparameter("max_features", 0.1, 0.9, default_value=0.1) 
	presort = UniformIntegerHyperparameter("presort", 0, 1, default_value=0)
	cs.add_hyperparameters([criterion,splitter,max_depth,min_samples_split, min_samples_leaf,max_features,presort])



def multinomial_config(cs):
	alpha = UniformFloatHyperparameter("alpha", 0.0, 1.0, default_value=1.0)
	fit_prior = UniformIntegerHyperparameter("fit_prior", 0, 1, default_value=1)
	cs.add_hyperparameters([alpha,fit_prior])

def logistic_regression_config(cs):
	penalty = CategoricalHyperparameter("penalty", ['l1', 'l2'], default_value='l2')
	dual = UniformIntegerHyperparameter("dual", 0, 1, default_value=1)
	tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-4)
	C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
	max_iter = UniformIntegerHyperparameter("max_iter", 1, 200, default_value=100)
	cs.add_hyperparameters([penalty, C, tol, max_iter, dual])


def linear_regression_config(cs):
	fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
	normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=0)
	copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
	cs.add_hyperparameters([fit_intercept, normalize, copy_X])


def bayesian_ridge_config(cs):
	fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
	normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=0)
	copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
	n_iter = UniformIntegerHyperparameter("n_iter", 100, 500, default_value=300)
	tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-3)
	alpha_1 = UniformFloatHyperparameter("alpha_1", 1e-8, 1.0, default_value=1e-6)
	alpha_2 = UniformFloatHyperparameter("alpha_2", 1e-8, 1.0, default_value=1e-6)
	lambda_1 = UniformFloatHyperparameter("lambda_1", 1e-8, 1.0, default_value=1e-6)
	lambda_2 = UniformFloatHyperparameter("lambda_2", 1e-8, 1.0, default_value=1e-6)
	compute_score = UniformIntegerHyperparameter("compute_score", 0, 1, default_value=0)
	cs.add_hyperparameters([fit_intercept, normalize, copy_X, n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, compute_score])





def lasso_config(cs):
	fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
	normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=0)
	copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
	max_iter = UniformIntegerHyperparameter("max_iter", 100, 500, default_value=300)
	tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-3)
	cv = UniformIntegerHyperparameter("cv", 1, 10, default_value=3)
	positive = UniformIntegerHyperparameter("positive", 0, 1, default_value=1)
	selection = CategoricalHyperparameter("selection", ['cyclic', 'random'], default_value='cyclic')
	eps = UniformFloatHyperparameter("eps", 1e-8, 1.0, default_value=1e-3)
	cs.add_hyperparameters([fit_intercept, normalize, copy_X, max_iter, tol, cv, positive, selection, eps])


def ridge_config(cs):
	fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
	normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=0)
	copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
	max_iter = UniformIntegerHyperparameter("max_iter", 100, 500, default_value=300)
	tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-3)
	alpha = UniformFloatHyperparameter("alpha", 0.1, 3.0, default_value=1.0)
	solver = CategoricalHyperparameter("solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'], default_value='auto')
	cs.add_hyperparameters([fit_intercept, normalize, copy_X, max_iter, tol, alpha, solver])

def lars_config(cs):
	fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
	normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=1)
	copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
	positive = UniformIntegerHyperparameter("positive", 0, 1, default_value=1)
	fit_path = UniformIntegerHyperparameter("fit_path", 0, 1, default_value=1)
	eps = UniformFloatHyperparameter("eps", 1e-18, 1.0, default_value=2.2204460492503131e-16)
	cs.add_hyperparameters([fit_intercept, normalize, copy_X,positive,fit_path, eps])


ESTIMATORS = {
        'sklearn.svm.classes.LinearSVC': svc_config,
        'sklearn.neighbors.classification.KNeighborsClassifier': kneighbors_config,
        'sklearn.tree.tree.DecisionTreeClassifier': decision_tree_config,
        'sklearn.naive_bayes.MultinomialNB': multinomial_config,
        'sklearn.ensemble.forest.RandomForestClassifier': random_forest_config,
        'sklearn.linear_model.logistic.LogisticRegression': logistic_regression_config,
        'sklearn.linear_model.base.LinearRegression': linear_regression_config,
        'sklearn.linear_model.bayes.BayesianRidge': bayesian_ridge_config,
        'sklearn.linear_model.coordinate_descent.LassoCV': lasso_config,
        'sklearn.linear_model.ridge.Ridge': ridge_config,
        'sklearn.linear_model.least_angle.Lars': lars_config
}



