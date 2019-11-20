import typing
import logging
import json
from d3m import index
from d3m.metadata.hyperparams import Bounded, Enumeration, UniformInt, UniformBool, Uniform, Normal, Union, \
    Constant as ConstantD3M
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.forbidden import ForbiddenEqualsClause, ForbiddenAndConjunction
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
     UniformIntegerHyperparameter, UnParametrizedHyperparameter, Constant, NormalFloatHyperparameter


logger = logging.getLogger(__name__)
PRIMITIVES = index.search()
HYPERPARAMS_FROM_METALEARNING_PATH = os.path.join(os.path.dirname(__file__), '../resource/hyperparams.json')

@staticmethod
def get_hyperparams_from_metalearnig():
    with open(HYPERPARAMS_FROM_METALEARNING_PATH) as fin:
        search_space = json.load(fin)
        return  search_space

def get_primitive_config(cs, primitive_name):
    primitive_class = index.get_primitive(primitive_name)
    hyperparameter_class = typing.get_type_hints(primitive_class.__init__)['hyperparams']
    default_config = get_default_configspace(primitive_name)
    default_hyperparameters = set(default_config.get_hyperparameter_names())

    if hyperparameter_class:
        config = hyperparameter_class.configuration
        hyperparameters = []
        for hp_name in config:
            new_hp_name = primitive_name + '|' + hp_name
            if new_hp_name in default_hyperparameters and not isinstance(config[hp_name], Union):
                new_hp = default_config.get_hyperparameter(new_hp_name)
            else:
                new_hp = cast_hyperparameters(config[hp_name], new_hp_name)

            if new_hp is not None:
                hyperparameters.append(new_hp)

        cs.add_hyperparameters(hyperparameters)

        for condition in default_config.get_conditions():
            try:
                cs.add_condition(condition)
            except Exception as e:
                logger.warning('Not possible to add condition', e)

        for forbidden in default_config.get_forbiddens():
            try:
                cs.add_forbidden_clause(forbidden)
            except Exception as e:
                logger.warning('Not possible to add forbidden clause', e)


def cast_hyperparameters(hyperparameter, name):
    # From D3M hyperparameters to ConfigSpace hyperparameters
    # TODO: Include 'Union', 'Choice' and  'Set' (D3M hyperparameters)
    new_hyperparameter = None

    if isinstance(hyperparameter, Bounded):
        lower = hyperparameter.lower
        upper = hyperparameter.upper
        default = hyperparameter.get_default()
        if lower is None:
            lower = default
        if upper is None:
            upper = default * 2 if default > 0 else 10
        if hyperparameter.structural_type == int:
            new_hyperparameter = UniformIntegerHyperparameter(name, lower, upper, default_value=default)
        else:
            new_hyperparameter = UniformFloatHyperparameter(name, lower, upper, default_value=default)
    elif isinstance(hyperparameter, UniformBool):
        default = hyperparameter.get_default()
        new_hyperparameter = CategoricalHyperparameter(name, [True, False], default_value=default)
    elif isinstance(hyperparameter, UniformInt):
        lower = hyperparameter.lower
        upper = hyperparameter.upper
        default = hyperparameter.get_default()
        new_hyperparameter = UniformIntegerHyperparameter(name, lower, upper, default_value=default)
    elif isinstance(hyperparameter, Uniform):
        lower = hyperparameter.lower
        upper = hyperparameter.upper
        default = hyperparameter.get_default()
        new_hyperparameter = UniformFloatHyperparameter(name, lower, upper, default_value=default)
    elif isinstance(hyperparameter, Normal):
        default = hyperparameter.get_default()
        new_hyperparameter = NormalFloatHyperparameter(name, default_value=default)
    elif isinstance(hyperparameter, Enumeration):
        values = hyperparameter.values
        default = hyperparameter.get_default()
        new_hyperparameter = CategoricalHyperparameter(name, values, default_value=default)
    elif isinstance(hyperparameter, ConstantD3M):
        new_hyperparameter = Constant(name, hyperparameter.get_default())

    return new_hyperparameter


def is_tunable(name):
    if name not in PRIMITIVES:
        return False
    if name in {'d3m.primitives.feature_extraction.yolo.DSBOX'}:
        # This primitive is not in the OBJECT_DETECTION family, so compares it by its name
        return True

    klass = index.get_primitive(name)
    family = klass.metadata.to_json_structure()['primitive_family']

    return family in {'CLASSIFICATION', 'REGRESSION', 'TIME_SERIES_CLASSIFICATION', 'TIME_SERIES_FORECASTING',
                      'SEMISUPERVISED_CLASSIFICATION', 'COMMUNITY_DETECTION', 'VERTEX_CLASSIFICATION', 'GRAPH_MATCHING',
                      'LINK_PREDICTION'}


def get_default_configspace(primitive):
    default_config = ConfigurationSpace()

    if primitive in get_hyperparams_from_metalearnig:
        default_config.add_configuration_space(primitive, get_configspace_from_metalearning(primitive), '|')
    elif primitive in PRIMITIVES_DEFAULT_HYPERPARAMETERS:
        default_config.add_configuration_space(primitive, PRIMITIVES_DEFAULT_HYPERPARAMETERS[primitive](), '|')

    return default_config

def get_configspace_from_metalearning(metalearning_entry):
    cs = ConfigurationSpace()

    categorical_hyperparams = [
        CategoricalHyperparameter(
            name=hyperparam,
            choices=metalearning_entry[hyperparam]['choices'],
            default_value=metalearning_entry[hyperparam]['choices'])
    for hyperparam in metalearning_entry]

    cs.add_hyperparameters(categorical_hyperparams)

    return cs

# CLASSIFICATION
def adaboost():
    cs = ConfigurationSpace()

    n_estimators = UniformIntegerHyperparameter(
        name="n_estimators", lower=50, upper=500, default_value=50, log=False)
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.01, upper=2, default_value=0.1, log=True)
    algorithm = CategoricalHyperparameter(
        name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")

    cs.add_hyperparameters([n_estimators, learning_rate, algorithm])

    return cs


def bernoulli_nb():
    cs = ConfigurationSpace()

    # the smoothing parameter is a non-negative float
    # I will limit it to 1000 and put it on a logarithmic scale. (SF)
    # Please adjust that, if you know a proper range, this is just a guess.
    alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100,
                                       default_value=1, log=True)

    fit_prior = CategoricalHyperparameter(name="fit_prior",
                                          choices=[True, False],
                                          default_value=True)

    cs.add_hyperparameters([alpha, fit_prior])

    return cs


def decision_tree():
    cs = ConfigurationSpace()

    criterion = CategoricalHyperparameter(
        "criterion", ["gini", "entropy"], default_value="gini")
    min_samples_split = UniformIntegerHyperparameter(
        "min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
    max_features = UnParametrizedHyperparameter('max_features', 1.0)
    max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
    min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

    cs.add_hyperparameters([criterion, max_features,
                            min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_leaf_nodes,
                            min_impurity_decrease])

    return cs


def extra_trees():
    cs = ConfigurationSpace()

    n_estimators = Constant("n_estimators", 100)
    criterion = CategoricalHyperparameter(
        "criterion", ["gini", "entropy"], default_value="gini")

    # The maximum number of features used in the forest is calculated as m^max_features, where
    # m is the total number of features, and max_features is the hyperparameter specified below.
    # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
    # corresponds with Geurts' heuristic.
    max_features = UniformFloatHyperparameter(
        "max_features", 0., 1., default_value=0.5)

    max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")

    min_samples_split = UniformIntegerHyperparameter(
        "min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = UnParametrizedHyperparameter('min_weight_fraction_leaf', 0.)
    max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
    min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)

    bootstrap = CategoricalHyperparameter(
        "bootstrap", [True, False], default_value=False)
    cs.add_hyperparameters([n_estimators, criterion, max_features,
                            max_depth, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_leaf_nodes,
                            min_impurity_decrease, bootstrap])

    return cs


def gradient_boosting():
    cs = ConfigurationSpace()
    loss = Constant("loss", "deviance")
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
    n_estimators = UniformIntegerHyperparameter(
        "n_estimators", 50, 500, default_value=100)
    max_depth = UniformIntegerHyperparameter(
        name="max_depth", lower=1, upper=10, default_value=3)
    criterion = CategoricalHyperparameter(
        'criterion', ['friedman_mse', 'mse', 'mae'],
        default_value='mse')
    min_samples_split = UniformIntegerHyperparameter(
        name="min_samples_split", lower=2, upper=20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        name="min_samples_leaf", lower=1, upper=20, default_value=1)
    min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
    subsample = UniformFloatHyperparameter(
        name="subsample", lower=0.01, upper=1.0, default_value=1.0)
    max_features = UniformFloatHyperparameter(
        "max_features", 0.1, 1.0, default_value=1)
    max_leaf_nodes = UnParametrizedHyperparameter(
        name="max_leaf_nodes", value="None")
    min_impurity_decrease = UnParametrizedHyperparameter(
        name='min_impurity_decrease', value=0.0)
    cs.add_hyperparameters([loss, learning_rate, n_estimators, max_depth,
                            criterion, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, subsample,
                            max_features, max_leaf_nodes,
                            min_impurity_decrease])

    return cs


def k_nearest_neighbors():
    cs = ConfigurationSpace()

    n_neighbors = UniformIntegerHyperparameter(
        name="n_neighbors", lower=1, upper=100, log=True, default_value=1)
    weights = CategoricalHyperparameter(
        name="weights", choices=["uniform", "distance"], default_value="uniform")
    p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)
    cs.add_hyperparameters([n_neighbors, weights, p])

    return cs


def lda():
    cs = ConfigurationSpace()
    shrinkage = CategoricalHyperparameter(
        "shrinkage", ["None", "auto", "manual"], default_value="None")

    n_components = UniformIntegerHyperparameter('n_components', 1, 250, default_value=10)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
    cs.add_hyperparameters([shrinkage, n_components, tol])

    return cs


def linear_svc():
    cs = ConfigurationSpace()

    penalty = CategoricalHyperparameter(
        "penalty", ["l1", "l2"], default_value="l2")
    loss = CategoricalHyperparameter(
        "loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
    dual = Constant("dual", False)
    # This is set ad-hoc
    tol = UniformFloatHyperparameter(
        "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
    C = UniformFloatHyperparameter(
        "C", 0.03125, 32768, log=True, default_value=1.0)
    multi_class = Constant("multi_class", "ovr")
    # These are set ad-hoc
    fit_intercept = Constant("fit_intercept", True)
    intercept_scaling = Constant("intercept_scaling", 1)
    cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class,
                            fit_intercept, intercept_scaling])

    penalty_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(penalty, "l1"),
        ForbiddenEqualsClause(loss, "hinge")
    )
    constant_penalty_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(dual, False),
        ForbiddenEqualsClause(penalty, "l2"),
        ForbiddenEqualsClause(loss, "hinge")
    )
    penalty_and_dual = ForbiddenAndConjunction(
        ForbiddenEqualsClause(dual, False),
        ForbiddenEqualsClause(penalty, "l1")
    )
    cs.add_forbidden_clause(penalty_and_loss)
    cs.add_forbidden_clause(constant_penalty_and_loss)
    cs.add_forbidden_clause(penalty_and_dual)

    return cs


def svc():
    C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                                   default_value=1.0)
    # No linear kernel here, because we have liblinear
    kernel = CategoricalHyperparameter(name="kernel",
                                       choices=["rbf", "poly", "sigmoid"],
                                       default_value="rbf")
    degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
    gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                       log=True, default_value=0.1)
    # TODO this is totally ad-hoc
    coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
    # probability is no hyperparameter, but an argument to the SVM algo
    shrinking = CategoricalHyperparameter("shrinking", [True, False],
                                          default_value=True)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                     log=True)
    # cache size is not a hyperparameter, but an argument to the program!
    max_iter = UnParametrizedHyperparameter("max_iter", -1)

    cs = ConfigurationSpace()
    cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                            tol, max_iter])

    degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
    coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)

    return cs


def multinomial_nb():
    cs = ConfigurationSpace()

    # the smoothing parameter is a non-negative float
    # I will limit it to 100 and put it on a logarithmic scale. (SF)
    # Please adjust that, if you know a proper range, this is just a guess.
    alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100,
                                       default_value=1, log=True)

    fit_prior = CategoricalHyperparameter(name="fit_prior",
                                          choices=[True, False],
                                          default_value=True)

    cs.add_hyperparameters([alpha, fit_prior])

    return cs


def passive_aggressive():
    C = UniformFloatHyperparameter("C", 1e-5, 10, 1.0, log=True)
    fit_intercept = UnParametrizedHyperparameter("fit_intercept", True)
    loss = CategoricalHyperparameter(
        "loss", ["hinge", "squared_hinge"], default_value="hinge"
    )

    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4,
                                     log=True)
    # Note: Average could also be an Integer if > 1
    average = CategoricalHyperparameter('average', ['False', 'True'],
                                        default_value='False')

    cs = ConfigurationSpace()
    cs.add_hyperparameters([loss, fit_intercept, tol, C, average])

    return cs


def qda():
    reg_param = UniformFloatHyperparameter('reg_param', 0.0, 1.0,
                                           default_value=0.0)
    cs = ConfigurationSpace()
    cs.add_hyperparameter(reg_param)

    return cs


def random_forest():
    cs = ConfigurationSpace()
    n_estimators = Constant("n_estimators", 100)
    criterion = CategoricalHyperparameter(
        "criterion", ["gini", "entropy"], default_value="gini")

    # The maximum number of features used in the forest is calculated as m^max_features, where
    # m is the total number of features, and max_features is the hyperparameter specified below.
    # The default is 0.5, which yields sqrt(m) features as max_features in the estimator. This
    # corresponds with Geurts' heuristic.
    max_features = UniformFloatHyperparameter(
        "max_features", 0., 1., default_value=0.5)

    max_depth = UnParametrizedHyperparameter("max_depth", "None")
    min_samples_split = UniformIntegerHyperparameter(
        "min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
    max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
    min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
    bootstrap = CategoricalHyperparameter(
        "bootstrap", [True, False], default_value=True)
    cs.add_hyperparameters([n_estimators, criterion, max_features,
                            max_depth, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_leaf_nodes,
                            bootstrap, min_impurity_decrease])

    return cs


def sgd():
    cs = ConfigurationSpace()

    loss = CategoricalHyperparameter("loss",
                                     ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                                     default_value="log")
    penalty = CategoricalHyperparameter(
        "penalty", ["l1", "l2", "elasticnet"], default_value="l2")
    alpha = UniformFloatHyperparameter(
        "alpha", 1e-7, 1e-1, log=True, default_value=0.0001)
    l1_ratio = UniformFloatHyperparameter(
        "l1_ratio", 1e-9, 1, log=True, default_value=0.15)
    fit_intercept = UnParametrizedHyperparameter("fit_intercept", True)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, log=True,
                                     default_value=1e-4)
    epsilon = UniformFloatHyperparameter(
        "epsilon", 1e-5, 1e-1, default_value=1e-4, log=True)
    learning_rate = CategoricalHyperparameter(
        "learning_rate", ["optimal", "invscaling", "constant"],
        default_value="invscaling")
    eta0 = UniformFloatHyperparameter(
        "eta0", 1e-7, 1e-1, default_value=0.01, log=True)
    power_t = UniformFloatHyperparameter("power_t", 1e-5, 1,
                                         default_value=0.5)
    average = CategoricalHyperparameter(
        "average", [False, True], default_value=False)
    cs.add_hyperparameters([loss, penalty, alpha, l1_ratio, fit_intercept,
                            tol, epsilon, learning_rate, eta0, power_t,
                            average])

    # TODO add passive/aggressive here, although not properly documented?
    elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
    epsilon_condition = EqualsCondition(epsilon, loss, "modified_huber")

    power_t_condition = EqualsCondition(power_t, learning_rate,
                                        "invscaling")

    # eta0 is only relevant if learning_rate!='optimal' according to code
    # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
    # linear_model/sgd_fast.pyx#L603
    eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling",
                                                        "constant"])
    cs.add_conditions([elasticnet, epsilon_condition, power_t_condition,
                       eta0_in_inv_con])

    return cs


# REGRESSION

def adaboost_regression():
    cs = ConfigurationSpace()

    # base_estimator = Constant(name="base_estimator", value="None")
    n_estimators = UniformIntegerHyperparameter(
        name="n_estimators", lower=50, upper=500, default_value=50,
        log=False)
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.01, upper=2, default_value=0.1,
        log=True)
    loss = CategoricalHyperparameter(
        name="loss", choices=["linear", "square", "exponential"],
        default_value="linear")

    cs.add_hyperparameters([n_estimators, learning_rate, loss])

    return cs


def ard_regression():
    cs = ConfigurationSpace()
    n_iter = UnParametrizedHyperparameter("n_iter", value=300)
    tol = UniformFloatHyperparameter("tol", 10 ** -5, 10 ** -1,
                                     default_value=10 ** -3, log=True)
    alpha_1 = UniformFloatHyperparameter(name="alpha_1", lower=10 ** -10,
                                         upper=10 ** -3, default_value=10 ** -6)
    alpha_2 = UniformFloatHyperparameter(name="alpha_2", log=True,
                                         lower=10 ** -10, upper=10 ** -3,
                                         default_value=10 ** -6)
    lambda_1 = UniformFloatHyperparameter(name="lambda_1", log=True,
                                          lower=10 ** -10, upper=10 ** -3,
                                          default_value=10 ** -6)
    lambda_2 = UniformFloatHyperparameter(name="lambda_2", log=True,
                                          lower=10 ** -10, upper=10 ** -3,
                                          default_value=10 ** -6)
    threshold_lambda = UniformFloatHyperparameter(name="threshold_lambda",
                                                  log=True,
                                                  lower=10 ** 3,
                                                  upper=10 ** 5,
                                                  default_value=10 ** 4)
    fit_intercept = UnParametrizedHyperparameter("fit_intercept", True)

    cs.add_hyperparameters([n_iter, tol, alpha_1, alpha_2, lambda_1,
                            lambda_2, threshold_lambda, fit_intercept])

    return cs


def decision_tree_regression():
    cs = ConfigurationSpace()

    criterion = CategoricalHyperparameter('criterion',
                                          ['mse', 'friedman_mse', 'mae'])
    max_features = Constant('max_features', 1.0)
    min_samples_split = UniformIntegerHyperparameter(
        "min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.0)
    max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
    min_impurity_decrease = UnParametrizedHyperparameter(
        'min_impurity_decrease', 0.0)

    cs.add_hyperparameters([criterion, max_features,
                            min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_leaf_nodes,
                            min_impurity_decrease])

    return cs


def extra_trees_regression():
    cs = ConfigurationSpace()

    n_estimators = Constant("n_estimators", 100)
    criterion = CategoricalHyperparameter("criterion",
                                          ['mse', 'friedman_mse', 'mae'])
    max_features = UniformFloatHyperparameter(
        "max_features", 0.1, 1.0, default_value=1)

    max_depth = UnParametrizedHyperparameter(name="max_depth", value="None")
    max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")

    min_samples_split = UniformIntegerHyperparameter(
        "min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 20, default_value=1)
    min_impurity_decrease = UnParametrizedHyperparameter(
        'min_impurity_decrease', 0.0
    )

    bootstrap = CategoricalHyperparameter(
        "bootstrap", [True, False], default_value=False)

    cs.add_hyperparameters([n_estimators, criterion, max_features,
                            max_depth, max_leaf_nodes, min_samples_split,
                            min_samples_leaf, min_impurity_decrease,
                            bootstrap])

    return cs


def gaussian_process_regression():
    alpha = UniformFloatHyperparameter(
        name="alpha", lower=1e-14, upper=1.0, default_value=1e-8, log=True)

    cs = ConfigurationSpace()
    cs.add_hyperparameters([alpha])

    return cs


def gradient_boosting_regression():
    cs = ConfigurationSpace()
    loss = CategoricalHyperparameter(
        "loss", ["ls", "lad", "huber", "quantile"], default_value="ls")
    learning_rate = UniformFloatHyperparameter(
        name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
    n_estimators = UniformIntegerHyperparameter(
        "n_estimators", 50, 500, default_value=100)
    max_depth = UniformIntegerHyperparameter(
        name="max_depth", lower=1, upper=10, default_value=3)
    min_samples_split = UniformIntegerHyperparameter(
        name="min_samples_split", lower=2, upper=20, default_value=2, log=False)
    min_samples_leaf = UniformIntegerHyperparameter(
        name="min_samples_leaf", lower=1, upper=20, default_value=1, log=False)
    min_weight_fraction_leaf = UnParametrizedHyperparameter(
        "min_weight_fraction_leaf", 0.)
    subsample = UniformFloatHyperparameter(
        name="subsample", lower=0.01, upper=1.0, default_value=1.0, log=False)
    max_features = UniformFloatHyperparameter(
        "max_features", 0.1, 1.0, default_value=1)
    max_leaf_nodes = UnParametrizedHyperparameter(
        name="max_leaf_nodes", value="None")
    min_impurity_decrease = UnParametrizedHyperparameter(
        name='min_impurity_decrease', value=0.0)
    alpha = UniformFloatHyperparameter(
        "alpha", lower=0.75, upper=0.99, default_value=0.9)

    cs.add_hyperparameters([loss, learning_rate, n_estimators, max_depth,
                            min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, subsample, max_features,
                            max_leaf_nodes, min_impurity_decrease, alpha])

    cs.add_condition(InCondition(alpha, loss, ['huber', 'quantile']))

    return cs


def k_nearest_neighbors_regression():
    cs = ConfigurationSpace()

    n_neighbors = UniformIntegerHyperparameter(
        name="n_neighbors", lower=1, upper=100, log=True, default_value=1)
    weights = CategoricalHyperparameter(
        name="weights", choices=["uniform", "distance"], default_value="uniform")
    p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)

    cs.add_hyperparameters([n_neighbors, weights, p])

    return cs


def linear_svr_regression():
    cs = ConfigurationSpace()
    C = UniformFloatHyperparameter(
        "C", 0.03125, 32768, log=True, default_value=1.0)
    loss = CategoricalHyperparameter(
        "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"],
        default_value="squared_epsilon_insensitive")
    # Random Guess
    epsilon = UniformFloatHyperparameter(
        name="epsilon", lower=0.001, upper=1, default_value=0.1, log=True)
    dual = Constant("dual", False)
    # These are set ad-hoc
    tol = UniformFloatHyperparameter(
        "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
    fit_intercept = Constant("fit_intercept", True)
    intercept_scaling = Constant("intercept_scaling", 1)

    cs.add_hyperparameters([C, loss, epsilon, dual, tol, fit_intercept,
                            intercept_scaling])

    dual_and_loss = ForbiddenAndConjunction(
        ForbiddenEqualsClause(dual, False),
        ForbiddenEqualsClause(loss, "epsilon_insensitive")
    )
    cs.add_forbidden_clause(dual_and_loss)

    return cs


def svr_regression():
    C = UniformFloatHyperparameter(
        name="C", lower=0.03125, upper=32768, log=True, default_value=1.0)
    # Random Guess
    epsilon = UniformFloatHyperparameter(name="epsilon", lower=0.001,
                                         upper=1, default_value=0.1,
                                         log=True)

    kernel = CategoricalHyperparameter(
        name="kernel", choices=['linear', 'poly', 'rbf', 'sigmoid'],
        default_value="rbf")
    degree = UniformIntegerHyperparameter(
        name="degree", lower=2, upper=5, default_value=3)

    gamma = UniformFloatHyperparameter(
        name="gamma", lower=3.0517578125e-05, upper=8, log=True, default_value=0.1)

    # TODO this is totally ad-hoc
    coef0 = UniformFloatHyperparameter(
        name="coef0", lower=-1, upper=1, default_value=0)
    # probability is no hyperparameter, but an argument to the SVM algo
    shrinking = CategoricalHyperparameter(
        name="shrinking", choices=[True, False], default_value=True)
    tol = UniformFloatHyperparameter(
        name="tol", lower=1e-5, upper=1e-1, default_value=1e-3, log=True)
    max_iter = UnParametrizedHyperparameter("max_iter", -1)

    cs = ConfigurationSpace()
    cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                           tol, max_iter, epsilon])

    degree_depends_on_kernel = InCondition(child=degree, parent=kernel,
                                           values=('poly', 'rbf', 'sigmoid'))
    gamma_depends_on_kernel = InCondition(child=gamma, parent=kernel,
                                          values=('poly', 'rbf'))
    coef0_depends_on_kernel = InCondition(child=coef0, parent=kernel,
                                          values=('poly', 'sigmoid'))
    cs.add_conditions([degree_depends_on_kernel, gamma_depends_on_kernel,
                       coef0_depends_on_kernel])

    return cs


def random_forest_regression():
    cs = ConfigurationSpace()
    n_estimators = Constant("n_estimators", 100)
    criterion = CategoricalHyperparameter("criterion",
                                          ['mse', 'friedman_mse', 'mae'])
    max_features = UniformFloatHyperparameter(
        "max_features", 0.1, 1.0, default_value=1.0)
    max_depth = UnParametrizedHyperparameter("max_depth", "None")
    min_samples_split = UniformIntegerHyperparameter(
        "min_samples_split", 2, 20, default_value=2)
    min_samples_leaf = UniformIntegerHyperparameter(
        "min_samples_leaf", 1, 20, default_value=1)
    min_weight_fraction_leaf = \
        UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
    max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", "None")
    min_impurity_decrease = UnParametrizedHyperparameter(
        'min_impurity_decrease', 0.0)
    bootstrap = CategoricalHyperparameter(
        "bootstrap", [True, False], default_value=True)

    cs.add_hyperparameters([n_estimators, criterion, max_features,
                            max_depth, min_samples_split, min_samples_leaf,
                            min_weight_fraction_leaf, max_leaf_nodes,
                            min_impurity_decrease, bootstrap])

    return cs


def ridge_regression():
    cs = ConfigurationSpace()
    alpha = UniformFloatHyperparameter(
        "alpha", 10 ** -5, 10., log=True, default_value=1.)
    fit_intercept = UnParametrizedHyperparameter("fit_intercept", True)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1,
                                     default_value=1e-3, log=True)
    cs.add_hyperparameters([alpha, fit_intercept, tol])

    return cs


def sgd_regression():
    cs = ConfigurationSpace()

    loss = CategoricalHyperparameter("loss",
                                     ["squared_loss", "huber", "epsilon_insensitive",
                                      "squared_epsilon_insensitive"],
                                     default_value="squared_loss")
    penalty = CategoricalHyperparameter(
        "penalty", ["l1", "l2", "elasticnet"], default_value="l2")
    alpha = UniformFloatHyperparameter(
        "alpha", 1e-7, 1e-1, log=True, default_value=0.0001)
    l1_ratio = UniformFloatHyperparameter(
        "l1_ratio", 1e-9, 1., log=True, default_value=0.15)
    fit_intercept = UnParametrizedHyperparameter(
        "fit_intercept", True)
    tol = UniformFloatHyperparameter(
        "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
    epsilon = UniformFloatHyperparameter(
        "epsilon", 1e-5, 1e-1, default_value=0.1, log=True)
    learning_rate = CategoricalHyperparameter(
        "learning_rate", ["optimal", "invscaling", "constant"],
        default_value="invscaling")
    eta0 = UniformFloatHyperparameter(
        "eta0", 1e-7, 1e-1, default_value=0.01, log=True)
    power_t = UniformFloatHyperparameter(
        "power_t", 1e-5, 1, default_value=0.25)
    average = CategoricalHyperparameter(
        "average", [False, True], default_value=False)

    cs.add_hyperparameters([loss, penalty, alpha, l1_ratio, fit_intercept,
                            tol, epsilon, learning_rate, eta0,
                            power_t, average])

    # TODO add passive/aggressive here, although not properly documented?
    elasticnet = EqualsCondition(l1_ratio, penalty, "elasticnet")
    epsilon_condition = InCondition(epsilon, loss,
                                    ["huber", "epsilon_insensitive", "squared_epsilon_insensitive"])

    # eta0 is only relevant if learning_rate!='optimal' according to code
    # https://github.com/scikit-learn/scikit-learn/blob/0.19.X/sklearn/
    # linear_model/sgd_fast.pyx#L603
    eta0_in_inv_con = InCondition(eta0, learning_rate, ["invscaling",
                                                        "constant"])
    power_t_condition = EqualsCondition(power_t, learning_rate,
                                        "invscaling")

    cs.add_conditions([elasticnet, epsilon_condition, power_t_condition,
                       eta0_in_inv_con])

    return cs



PRIMITIVES_DEFAULT_HYPERPARAMETERS = {
    'd3m.primitives.classification.ada_boost.SKlearn': adaboost,
    'd3m.primitives.classification.bernoulli_naive_bayes.SKlearn': bernoulli_nb,
    'd3m.primitives.classification.decision_tree.SKlearn': decision_tree,
    'd3m.primitives.classification.extra_trees.SKlearn': extra_trees,
    'd3m.primitives.classification.gradient_boosting.SKlearn': gradient_boosting,
    'd3m.primitives.classification.k_neighbors.SKlearn': k_nearest_neighbors,
    'd3m.primitives.classification.linear_discriminant_analysis.SKlearn': lda,
    'd3m.primitives.classification.linear_svc.SKlearn': linear_svc,
    'd3m.primitives.classification.svc.SKlearn': svc,
    'd3m.primitives.classification.multinomial_naive_bayes.SKlearn': multinomial_nb,
    'd3m.primitives.classification.passive_aggressive.SKlearn': passive_aggressive,
    'd3m.primitives.classification.quadratic_discriminant_analysis.SKlearn': qda,
    'd3m.primitives.classification.random_forest.SKlearn': random_forest,
    'd3m.primitives.classification.sgd.SKlearn': sgd,
    'd3m.primitives.regression.ada_boost.SKlearn': adaboost_regression,
    'd3m.primitives.regression.ard.SKlearn': ard_regression,
    'd3m.primitives.regression.decision_tree.SKlearn': decision_tree_regression,
    'd3m.primitives.regression.extra_trees.SKlearn': extra_trees_regression,
    'd3m.primitives.regression.gaussian_process.SKlearn': gaussian_process_regression,
    'd3m.primitives.regression.gradient_boosting.SKlearn': gradient_boosting_regression,
    'd3m.primitives.regression.k_neighbors.SKlearn': k_nearest_neighbors_regression,
    'd3m.primitives.regression.linear_svr.SKlearn': linear_svr_regression,
    'd3m.primitives.regression.svr.SKlearn': svr_regression,
    'd3m.primitives.regression.random_forest.SKlearn': random_forest_regression,
    'd3m.primitives.regression.ridge.SKlearn': ridge_regression,
    'd3m.primitives.regression.sgd.SKlearn': sgd_regression
}
