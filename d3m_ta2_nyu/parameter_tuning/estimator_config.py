import typing
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
     UniformFloatHyperparameter, UniformIntegerHyperparameter, \
     NormalFloatHyperparameter, FloatHyperparameter, IntegerHyperparameter
import ConfigSpace.conditions
from smac.configspace import ConfigurationSpace
from d3m_metadata.hyperparams import Bounded, Enumeration, Uniform, UniformInt, Normal
from d3m_ta2_nyu.workflow.module_loader import get_class
import random


def svm_config(cs):
    # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
    kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
    cs.add_hyperparameter(kernel)

    # There are some hyperparameters shared by all kernels
    C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
    shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="true")
    cs.add_hyperparameters([C, shrinking])

    # Others are kernel-specific, so we can add conditions to limit the searchspace
    degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)  # Only used by kernel poly
    coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
    cs.add_hyperparameters([degree, coef0])
    use_degree = ConfigSpace.conditions.InCondition(child=degree, parent=kernel, values=["poly"])
    use_coef0 = ConfigSpace.conditions.InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
    cs.add_conditions([use_degree, use_coef0])

    # This also works for parameters that are a mix of categorical and values from a range of numbers
    # For example, gamma can be either "auto" or a fixed float
    gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
    gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
    cs.add_hyperparameters([gamma, gamma_value])
    # We only activate gamma_value if gamma is set to "value"
    cs.add_condition(ConfigSpace.conditions.InCondition(child=gamma_value, parent=gamma, values=["value"]))
    # And again we can restrict the use of gamma in general to the choice of the kernel
    cs.add_condition(ConfigSpace.conditions.InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))


def random_forest_config(cs):
    # Random Forest hyperparameters
    max_depth = UniformIntegerHyperparameter("max_depth", 1, 10, default_value=4)
    max_features = UniformFloatHyperparameter("max_features", 0.1, 0.9, default_value=0.1)
    n_estimators = UniformIntegerHyperparameter("n_estimators", 1, 100, default_value=10)
    criterion = CategoricalHyperparameter("criterion", ['gini', 'entropy'], default_value='entropy')
    cs.add_hyperparameters([max_depth, max_features, n_estimators, criterion])


def svc_config(cs):
    penalty = CategoricalHyperparameter("penalty", ['l1', 'l2'], default_value='l2')
    loss = CategoricalHyperparameter("loss", ['hinge', 'squared_hinge'], default_value='squared_hinge')
    C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
    tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-4)
    max_iter = UniformIntegerHyperparameter("max_iter", 1, 2000, default_value=1000)
    dual = UniformIntegerHyperparameter("dual", 0, 1, default_value=1)
    cs.add_hyperparameters([penalty, loss, C, tol, max_iter, dual])


def linear_svc_config(cs):
    penalty = CategoricalHyperparameter("penalty", ['l1', 'l2'], default_value='l2')
    loss = CategoricalHyperparameter("loss", ['hinge', 'squared_hinge'], default_value='squared_hinge')
    C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
    tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-4)
    max_iter = UniformIntegerHyperparameter("max_iter", 1, 2000, default_value=1000)
    dual = UniformIntegerHyperparameter("dual", 0, 1, default_value=1)
    cs.add_hyperparameters([penalty, loss, C, tol, max_iter, dual])


def kneighbors_config(cs):
    n_neighbors = UniformIntegerHyperparameter("n_neighbors", 1, 10, default_value=5)
    weights = CategoricalHyperparameter("weights", ['uniform', 'distance'], default_value='uniform')
    algorithm = CategoricalHyperparameter("algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'], default_value='auto')
    leaf_size = UniformIntegerHyperparameter("leaf_size", 10, 50, default_value=30)
    p = UniformIntegerHyperparameter("p", 1, 2, default_value=2)
    cs.add_hyperparameters([n_neighbors, weights, algorithm, leaf_size, p])


def decision_tree_config(cs):
    criterion = CategoricalHyperparameter("criterion", ['gini', 'entropy'], default_value='gini')
    splitter = CategoricalHyperparameter("splitter", ['best', 'random'], default_value='best')
    # max_depth = UniformIntegerHyperparameter("max_depth", 1, 10, default_value=None)
    min_samples_split = UniformIntegerHyperparameter("min_samples_split", 2, 10, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter("min_samples_leaf", 1, 10, default_value=1)
    max_features = CategoricalHyperparameter("max_features", ['auto', 'sqrt', 'log2', None], default_value=None)
    presort = UniformIntegerHyperparameter("presort", 0, 1, default_value=0)
    cs.add_hyperparameters([criterion, splitter, min_samples_split, min_samples_leaf, max_features, presort])


def multinomial_config(cs):
    alpha = UniformFloatHyperparameter("alpha", 0.0, 1.0, default_value=1.0)
    fit_prior = UniformIntegerHyperparameter("fit_prior", 0, 1, default_value=1)
    cs.add_hyperparameters([alpha, fit_prior])


def mlp_config(cs):
    alpha = UniformFloatHyperparameter("alpha", 0.0, 1.0, default_value=1.0)
    activation = CategoricalHyperparameter("activation", ['identity', 'logistic', 'tanh', 'relu'], default_value='relu')
    solver = CategoricalHyperparameter('solver', ['lbfgs', 'sgd', 'adam'], default_value='adam')
    learning_rate = CategoricalHyperparameter('learning_rate', ['constant', 'invscaling', 'adaptive'],
                                              default_value='constant')
    momentum = UniformFloatHyperparameter("momentum", 0.0, 1.0, default_value=0.9)
    cs.add_hyperparameters([alpha, activation, solver, learning_rate, momentum])


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
    cs.add_hyperparameters(
        [fit_intercept, normalize, copy_X, n_iter, tol, alpha_1, alpha_2, lambda_1, lambda_2, compute_score])


def lasso_config(cs):
    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
    normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=0)
    copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
    max_iter = UniformIntegerHyperparameter("max_iter", 100, 5000, default_value=1000)
    tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-4)
    cv = UniformIntegerHyperparameter("cv", 1, 10, default_value=3)
    positive = UniformIntegerHyperparameter("positive", 0, 1, default_value=0)
    selection = CategoricalHyperparameter("selection", ['cyclic', 'random'], default_value='cyclic')
    cs.add_hyperparameters([fit_intercept, normalize, copy_X, max_iter, tol, cv, positive, selection])


def elastic_net_config(cs):
    alpha = UniformFloatHyperparameter("alpha", 0.0, 1.0, default_value=1.0)
    l1_ratio = UniformFloatHyperparameter("l1_ratio", 0.0, 1.0, default_value=0.5)
    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
    normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=0)
    copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
    max_iter = UniformIntegerHyperparameter("max_iter", 100, 5000, default_value=1000)
    tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-4)
    positive = UniformIntegerHyperparameter("positive", 0, 1, default_value=0)
    selection = CategoricalHyperparameter("selection", ['cyclic', 'random'], default_value='cyclic')
    cs.add_hyperparameters([alpha, l1_ratio, fit_intercept, normalize, copy_X, max_iter, tol, positive, selection])


def ridge_config(cs):
    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
    normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=0)
    copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
    max_iter = UniformIntegerHyperparameter("max_iter", 100, 500, default_value=300)
    tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-3)
    alpha = UniformFloatHyperparameter("alpha", 0.1, 3.0, default_value=1.0)
    solver = CategoricalHyperparameter("solver", ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                                       default_value='auto')
    cs.add_hyperparameters([fit_intercept, normalize, copy_X, max_iter, tol, alpha, solver])


def theil_sen_regressor_config(cs):
    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
    copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
    tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-3)
    max_iter = UniformIntegerHyperparameter("max_iter", 100, 500, default_value=300)
    cs.add_hyperparameters([fit_intercept, copy_X, max_iter, tol])


def huber_regressor_config(cs):
    epsilon = UniformFloatHyperparameter("epsilon", 1.0, 3.0, default_value=1.35)
    alpha = UniformFloatHyperparameter("alpha", 0.00001, 1.0, default_value=0.0001)
    fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=True)
    tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-5)
    max_iter = UniformIntegerHyperparameter("max_iter", 50, 500, default_value=100)
    warm_start = CategoricalHyperparameter('warm_start', [True, False], default_value=False)
    cs.add_hyperparameters([epsilon, alpha, fit_intercept, warm_start, max_iter, tol])


def ard_regression(cs):
    copy_X = CategoricalHyperparameter("copy_X", [True, False], default_value=True)
    fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=True)
    normalize = CategoricalHyperparameter("normalize", [True, False], default_value=False)
    tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-3)
    n_iter = UniformIntegerHyperparameter("max_iter", 100, 500, default_value=300)
    cs.add_hyperparameters([normalize, fit_intercept, copy_X, n_iter, tol])


def lars_config(cs):
    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
    normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=1)
    copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
    positive = UniformIntegerHyperparameter("positive", 0, 1, default_value=1)
    fit_path = UniformIntegerHyperparameter("fit_path", 0, 1, default_value=1)
    n_nonzero_coefs = UniformIntegerHyperparameter('n_nonzero_coefs', 100, 1000, default_value=500)
    cs.add_hyperparameters([fit_intercept, normalize, copy_X, positive, fit_path, n_nonzero_coefs])


def lasso_lars_config(cs):
    fit_intercept = UniformIntegerHyperparameter("fit_intercept", 0, 1, default_value=1)
    normalize = UniformIntegerHyperparameter("normalize", 0, 1, default_value=1)
    copy_X = UniformIntegerHyperparameter("copy_X", 0, 1, default_value=1)
    positive = UniformIntegerHyperparameter("positive", 0, 1, default_value=1)
    fit_path = UniformIntegerHyperparameter("fit_path", 0, 1, default_value=1)
    max_iter = UniformIntegerHyperparameter("max_iter", 100, 1000, default_value=500)
    cs.add_hyperparameters([fit_intercept, normalize, copy_X, positive, fit_path, max_iter])



def gp_config(cs):
    warm_start = CategoricalHyperparameter('warm_start', [True, False], default_value=False)
    multi_class = CategoricalHyperparameter('multi_class', ['one_vs_rest', 'one_vs_one'], default_value='one_vs_rest')
    max_iter_predict = UniformIntegerHyperparameter('max_iter_predict', 50, 200, default_value=100)
    cs.add_hyperparameters([warm_start, multi_class, max_iter_predict])


def rbf_config(cs):
    length_scale = UniformFloatHyperparameter('length_scale', 1e-05, 1e5, default_value=1.0)
    cs.add_hyperparameter(length_scale)


def ada_boost_config(cs):
    n_estimators = UniformIntegerHyperparameter('n_estimators', 10, 200, default_value=50)
    learning_rate = UniformFloatHyperparameter('learning_rate', 1e-5, 10., default_value=1.)
    algorithm = CategoricalHyperparameter('algorithm', ['SAMME', 'SAMME.R'], default_value='SAMME.R')
    cs.add_hyperparameters([n_estimators, learning_rate, algorithm])


def qda_config(cs):
    tol = UniformFloatHyperparameter("tol", 1e-8, 1.0, default_value=1e-4)
    reg_param = UniformFloatHyperparameter("reg_param", 0.0, 0.9, default_value=0.0)
    store_covariance = CategoricalHyperparameter('store_covariance', [True, False, None], default_value=None)
    cs.add_hyperparameters([tol, reg_param, store_covariance])


def sgd_config(cs):
    penalty = CategoricalHyperparameter("penalty", ['l1', 'l2', 'none', 'elasticnet'], default_value='l2')
    loss = CategoricalHyperparameter("loss", ['hinge', 'squared_hinge', 'log', 'modified_huber', 'perceptron'],
                                     default_value='hinge')
    alpha = UniformFloatHyperparameter("alpha", 0.00001, 1.0, default_value=0.0001)
    fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=True)
    warm_start = CategoricalHyperparameter('warm_start', [True, False], default_value=False)
    learning_rate = CategoricalHyperparameter('learning_rate', ['optimal', 'constant', 'invscaling'],
                                              default_value='optimal')
    eta0 = UniformFloatHyperparameter('eta0', 0.0, 1.0, default_value=0.0)
    cs.add_hyperparameters([penalty, loss, alpha, fit_intercept, warm_start, learning_rate, eta0])


def sgd_regressor_config(cs):
    penalty = CategoricalHyperparameter("penalty", ['l1', 'l2', 'none', 'elasticnet'], default_value='l2')
    loss = CategoricalHyperparameter("loss",
                                     ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                                     default_value='squared_loss')
    alpha = UniformFloatHyperparameter("alpha", 0.00001, 1.0, default_value=0.0001)
    fit_intercept = CategoricalHyperparameter("fit_intercept", [True, False], default_value=True)
    warm_start = CategoricalHyperparameter('warm_start', [True, False], default_value=False)
    learning_rate = CategoricalHyperparameter('learning_rate', ['optimal', 'constant', 'invscaling'],
                                              default_value='optimal')
    eta0 = UniformFloatHyperparameter('eta0', 0.0, 1.0, default_value=0.01)
    cs.add_hyperparameters([penalty, loss, alpha, fit_intercept, warm_start, learning_rate, eta0])


def passive_agressive_config(cs):
    C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
    cs.add_hyperparameter(C)


def ransac_regressor_config(cs):
    stop_probability = UniformFloatHyperparameter("stop_probability", 0.0, 1.0, default_value=0.9)
    max_trials = UniformIntegerHyperparameter('max_trials', 50, 200, default_value=100)
    cs.add_hyperparameters([stop_probability, max_trials])


def primitive_config(cs, primitive_name):
    PrimitiveClass = get_class(primitive_name)
    HyperparameterClass = typing.get_type_hints(PrimitiveClass.__init__)['hyperparams']
    if HyperparameterClass:
        config = HyperparameterClass.configuration
        parameter_list = []
        for p in config:
            parameter_name = primitive_name + '|' + p
            if isinstance(config[p], Bounded):
                lower = config[p].lower
                upper = config[p].upper
                default = config[p].default
                if type(default) == int:
                    cs_param = IntegerHyperparameter(parameter_name, lower, upper, default_value=default)
                else:
                    cs_param = FloatHyperparameter(parameter_name, lower, upper, default_value=default)
                parameter_list.append(cs_param)
            elif isinstance(config[p], Uniform):
                lower = config[p].lower
                upper = config[p].upper
                default = config[p].default
                cs_param = UniformFloatHyperparameter(parameter_name, lower, upper, default_value=default)
                parameter_list.append(cs_param)
            elif isinstance(config[p], UniformInt):
                lower = config[p].lower
                upper = config[p].upper
                default = config[p].default
                cs_param = UniformIntegerHyperparameter(parameter_name, lower, upper, default_value=default)
                parameter_list.append(cs_param)
            elif isinstance(config[p], Normal):
                lower = config[p].lower
                upper = config[p].upper
                default = config[p].default
                cs_param = NormalFloatHyperparameter(parameter_name, lower, upper, default_value=default)
                parameter_list.append(cs_param)
            elif isinstance(config[p], Enumeration):
                values = config[p].values
                default = config[p].default
                cs_param = CategoricalHyperparameter(parameter_name, values, default_value=default)
                parameter_list.append(cs_param)
        cs.add_hyperparameters(parameter_list)


ESTIMATORS = {
    'sklearn.svm.classes.LinearSVC': linear_svc_config,
    'sklearn.neighbors.classification.KNeighborsClassifier': kneighbors_config,
    'sklearn.tree.tree.DecisionTreeClassifier': decision_tree_config,
    'sklearn.naive_bayes.MultinomialNB': multinomial_config,
    'sklearn.ensemble.forest.RandomForestClassifier': random_forest_config,
    'sklearn.linear_model.logistic.LogisticRegression': logistic_regression_config,
    'sklearn.linear_model.base.LinearRegression': linear_regression_config,
    'sklearn.linear_model.bayes.BayesianRidge': bayesian_ridge_config,
    'sklearn.linear_model.ridge.Ridge': ridge_config,
    'sklearn.linear_model.least_angle.Lars': lars_config,
    'sklearn.neural_network.MLPClassifier': mlp_config,
    'sklearn.svm.SVC': svc_config,
    'sklearn.gaussian_process.GaussianProcessClassifier': gp_config,
    'sklearn.gaussian_process.kernels.RBF': rbf_config,
    'sklearn.tree.DecisionTreeClassifier': decision_tree_config,
    'sklearn.ensemble.AdaBoostClassifier': ada_boost_config,
    'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis': qda_config,
    'sklearn.linear_model.SGDClassifier': sgd_config,
    'sklearn.linear_model.ARDRegression': ard_regression,
    'sklearn.linear_model.BayesianRidge': bayesian_ridge_config,
    'sklearn.linear_model.ElasticNet': elastic_net_config,
    'sklearn.linear_model.HuberRegressor': huber_regressor_config,
    'sklearn.linear_model.Lars': lars_config,
    'sklearn.linear_model.Lasso': lasso_config,
    'sklearn.linear_model.LassoLars': lasso_lars_config,
    'sklearn.linear_model.PassiveAggressiveRegressor': passive_agressive_config,
    'sklearn.linear_model.RANSACRegressor': ransac_regressor_config,
    'sklearn.linear_model.Ridge': ridge_config,
    'sklearn.linear_model.SGDRegressor': sgd_regressor_config,
    'sklearn.linear_model.TheilSenRegressor': theil_sen_regressor_config

}





def get_random_hyperparameters(estimator):
    param_dict = {}
    if estimator in ESTIMATORS:
        cs = ConfigurationSpace()
        ESTIMATORS[estimator](cs)
        for param in cs.get_hyperparameters():
            if isinstance(param, CategoricalHyperparameter):
                param_dict[param.name] = random.choice(param.choices)
            elif isinstance(param, UniformFloatHyperparameter):
                param_dict[param.name] = random.uniform(param.lower,param.upper)
            else:
                param_dict[param.name] = random.randint(param.lower, param.upper)
    return param_dict


def get_default_hyperparameters(estimator):
    param_dict = {}
    if estimator in ESTIMATORS:
        cs = ConfigurationSpace()
        ESTIMATORS[estimator](cs)
        for param in cs.get_hyperparameters():
            param_dict[param.name] = param.default_value
    return param_dict



def encode_hyperparameter(estimator,parameter,value):
    if estimator in ESTIMATORS:
        cs = ConfigurationSpace()
        ESTIMATORS[estimator](cs)
        for param in cs.get_hyperparameters():
            if param.name == parameter:
                if value in param.choices:
                    return param.choices.index(value)
                else:
                    return -1
    return -1


def decode_hyperparameter(estimator,parameter,value):
    if estimator in ESTIMATORS:
        cs = ConfigurationSpace()
        ESTIMATORS[estimator](cs)
        for param in cs.get_hyperparameters():
            if param.name == parameter:
                if value in range(len(param.choices)):
                    return param.choices[value]
                else:
                    return -1
    return -1
