from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# Customized classifiers
elasticSGD = lambda **params: SGDClassifier(loss='log', penalty='elasticnet',
                                    tol=1e-5, **params)

def makeClassifierList():
    classifiers =   [{'name':  'knn',
                    'hyperparameter_space': [(1,50)],                       # N neighbors,
                    'hyperparameter_names': ['n_neighbors'],
                    'n_calls':{'rand':10 ,'opt':20 },
                    'func': KNeighborsClassifier        # predict_proba
                    },

                    {'name':  'linSVM',
                    'hyperparameter_space': [(1e-4, 1e5)],                     # C, A(class_weight)
                    'hyperparameter_names': ['C'],
                    'n_calls':{'rand':10 ,'opt':50 },
                    'func': LinearSVC                   # decision_function  class_weight
                    },

                    {'name':  'rbfSVM',
                    'hyperparameter_space': [(1e-4, 1e6, "log-uniform"),       # C
                                            (1e-4, 10, "log-uniform")],    # gamma
                    'hyperparameter_names': ['C','gamma'],
                    'n_calls':{'rand':50 ,'opt':120 },  # decision_function
                    'func': SVC
                    },

                    {'name':  'Decision_tree',
                    'hyperparameter_space': [(2, 8),                     # max_depth
                                            [.7,'sqrt','log2']],          # max_features
                    'hyperparameter_names': ['max_depth', 'max_features'],
                    'n_calls':{'rand':10 ,'opt':30 },
                    'func': DecisionTreeClassifier        #class_weight predict_proba
                    },

                    {'name':  'Random_forest',
                    'hyperparameter_space': [(2, 8),                    # max_depth
                                            (2,80),                     # n_estimators
                                            [.7,'sqrt','log2']],          # max_features
                    'hyperparameter_names': ['max_depth','n_estimators', 'max_features'],
                    'n_calls':{'rand':20 ,'opt':80 },
                    'func': RandomForestClassifier      # class_weight, predict_proba, oob_decision_function_
                    },

                    {'name':  'Neural_network',
                    'hyperparameter_space': [(1e-12,1e-1, "log-uniform"),   # alpha
                                            (2,40)],                        # hidden_layer_sizes,
                    'hyperparameter_names': ['alpha', 'hidden_layer_sizes'],
                    'n_calls':{'rand':20 ,'opt':80 },
                    'func': MLPClassifier               # predict_proba
                    },

                    {'name':  'Naive_bayes',
                    'hyperparameter_space': (),
                    'hyperparameter_names': (),
                    'n_calls':{'rand':() ,'opt':() },
                    'func': GaussianNB                  # predict_proba, priors
                    },

                    {'name':  'QDA',
                    'hyperparameter_space': (),
                    'hyperparameter_names': (),
                    'n_calls':{'rand':() ,'opt':() },
                    'func': QuadraticDiscriminantAnalysis #priors, predict_proba
                    },

                    {'name':  'LDA',
                    'hyperparameter_space': (),
                    'hyperparameter_names': (),
                    'n_calls':{'rand':() ,'opt':() },
                    'func': LinearDiscriminantAnalysis   # priors, predict_proba
                    },

                    {'name':  'Logistic_Regression',
                    'hyperparameter_space': (['l1','l2'],
                                             (1e-4, 1e4, "log-uniform")),
                    'hyperparameter_names': ['penalty', 'C'],
                    'n_calls':{'rand':(10) ,'opt':(20) },
                    'func': LogisticRegression          # class_weight, predict_proba
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
                    'hyperparameter_names': ['max_depth', 'learning_rate', 'n_estimators', 'gamma', 'min_child_weight', 'reg_alpha', 'reg_lambda', 'colsample_bytree', 'subsample'],
                    'n_calls':{'rand':(60) ,'opt':(100) },
                    'func': XGBClassifier
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
                    'hyperparameter_names': ['max_depth','num_leaves', 'learning_rate', 'n_estimators', 'boosting_type', 'min_child_weight', 'reg_alpha', 'reg_lambda', 'colsample_bytree', 'subsample'],
                    'n_calls':{'rand':(60) ,'opt':(100) },
                    'func': LGBMClassifier
                    }
                    ]
    return classifiers
