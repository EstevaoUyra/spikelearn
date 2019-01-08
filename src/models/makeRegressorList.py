from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import clone
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, ARDRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from functools import partial

def makeRegressorList():
    regressors =   [{'name':  'knn',
                    'hyperparameter_space': [(1,50)],                       # N neighbors,
                    'hyperparameter_names': ['n_neighbors'],
                    'n_calls':{'rand':10 ,'opt':20 },
                    'func': KNeighborsRegressor        # predict_proba
                    },

                    {'name':  'linSVM',
                    'hyperparameter_space': [(1e-4, 1e4, "log-uniform"), (0, 1)],                     # C, A(class_weight)
                    'hyperparameter_names': ['C', 'epsilon'],
                    'n_calls':{'rand':10 ,'opt':50 },
                    'func': partial(LinearSVR, max_iter=1e4)                   # decision_function  class_weight
                    },

                    {'name':  'rbfSVM',
                    'hyperparameter_space': [(1e-4, 1e6, "log-uniform"),       # C
                                            (1e-4, 10, "log-uniform"), # gamma
                                            (0, 1)],    #epsilon
                    'hyperparameter_names': ['C','gamma', 'epsilon'],
                    'n_calls':{'rand':50 ,'opt':120 },  # decision_function
                    'func': SVR
                    },
                    {'name':  'ElasticNet',
                    'hyperparameter_space': [(1e-4, 1e4, "log-uniform"),
                                            (0, 1)],
                    'hyperparameter_names': ['alpha','l1_ratio'],
                    'n_calls':{'rand':10 ,'opt':20 },  # decision_function
                    'func': ElasticNet
                    },

                    {'name':  'Decision_tree',
                    'hyperparameter_space': [(2, 8),                     # max_depth
                                            [.7,'sqrt','log2']],          # max_features
                    'hyperparameter_names': ['max_depth', 'max_features'],
                    'n_calls':{'rand':10 ,'opt':30 },
                    'func': DecisionTreeRegressor       #class_weight predict_proba
                    },

                    {'name':  'Random_forest',
                    'hyperparameter_space': [(2, 8),                    # max_depth
                                            (2,80),                     # n_estimators
                                            [.7,'sqrt','log2']],          # max_features
                    'hyperparameter_names': ['max_depth','n_estimators', 'max_features'],
                    'n_calls':{'rand':20 ,'opt':80 },
                    'func': RandomForestRegressor      # class_weight, predict_proba, oob_decision_function_
                    },

                    {'name':  'Neural_network',
                    'hyperparameter_space': [(1e-6,1e-1, "log-uniform"),   # alpha
                                            (2,40)],                        # hidden_layer_sizes,
                    'hyperparameter_names': ['alpha', 'hidden_layer_sizes'],
                    'n_calls':{'rand':20 ,'opt':80 },
                    'func': MLPRegressor               # predict_proba
                    },

                    {'name':  'XGboost',
                    'hyperparameter_space': [(4, 10),
                                             (0.001, .3, 'log-uniform'),
                                             (10, 150),
                                             (1e-4,10.,"log-uniform"),
                                             (1, 30),
                                             (1e-4, 10., "log-uniform"),
                                             (1e-4, 10., "log-uniform"),
                                             (.1, 1., "log-uniform"),
                                             (.1, 1., "log-uniform")],
                    'hyperparameter_names': ['max_depth', 'learning_rate', 'n_estimators', 'gamma', 
                                             'min_child_weight', 'reg_alpha', 'reg_lambda', 'colsample_bytree', 'subsample'],
                    'n_calls':{'rand':(60) ,'opt':(100) },
                    'func': XGBRegressor
                    },
                    {'name':  'LightGBM',
                    'hyperparameter_space': [(4, 10),
                                             (6, 10),
                                             (0.001, .3, 'log-uniform'),
                                             (10, 150),
                                             ['gbdt','dart'],
                                             (1, 30),
                                             (1e-4, 10., "log-uniform"),
                                             (1e-4, 10., "log-uniform"),
                                             (.1, 1., "log-uniform"),
                                             (.1, 1., "log-uniform")],
                    'hyperparameter_names': ['max_depth','num_leaves', 'learning_rate', 'n_estimators', 'boosting_type', 
                                             'min_child_weight', 'reg_alpha', 'reg_lambda', 'colsample_bytree', 'subsample'],
                    'n_calls':{'rand':(60) ,'opt':(100) },
                    'func': LGBMRegressor
                    },
                    {'name':  'CatBoost',
                    'hyperparameter_space': [(1e-4, 1e4, 'log-uniform'), (1e-4, 1e4, 'log-uniform')],
                    'hyperparameter_names': ['l2_leaf_reg', 'bagging_temperature'],
                    'n_calls':{'rand':(10) ,'opt':(30) },
                    'func': partial(CatBoostRegressor, verbose=False)   # priors, predict_proba
                    },
                    {'name':  'BayesianRidge',
                    'hyperparameter_space': (),
                    'hyperparameter_names': (),
                    'n_calls':{'rand':() ,'opt':() },
                    'func': BayesianRidge   # priors, predict_proba
                    },
                    {'name':  'ARD',
                    'hyperparameter_space': (),
                    'hyperparameter_names': (),
                    'n_calls':{'rand':() ,'opt':() },
                    'func': ARDRegression   # priors, predict_proba
                    },
                    ]
    return regressors
