"""
This script runs optimization on SVM RBF hyperparameters C and gamma.
When loaded as a module, it provides the optimize function
"""
#import sys
#sys.path.append('/home/tevo/Documents/UFABC/Spikes')
import time
import sys
sys.path.append('/home/tevo/Documents/UFABC/Spikes')
sys.path.append('/home/tevo/Documents/UFABC/SingleUnit Spike Learning/src/models')
import os
os.chdir('/home/tevo/Documents/UFABC/Spikes')

import numpy as np
import pandas as pd
from skopt import gp_minimize, dump
from spikeHelper.loadSpike import Rat

from sklearn.base import clone
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.model_selection import ShuffleSplit
from makeClassifierList import makeClassifierList


kappa = make_scorer(cohen_kappa_score, weights = 'quadratic')

def shufflin_trial_score(clf, X,y, trial, train_size=30, n_splits = 10,scoring=kappa):
    sh = ShuffleSplit(n_splits=n_splits, train_size=train_size,test_size=.3)
    score = []
    print(clf.get_params())
    for train_trials, test_trials in sh.split(np.unique(trial)):
        train_idx, test_idx = n_to_idx(np.unique(trial)[train_trials], trial), n_to_idx(np.unique(trial)[test_trials], trial)
        clf_local = clone(clf)
        clf_local.fit(X[train_idx,:],y[train_idx])
        score.append(scoring(clf_local, X[test_idx,:], y[test_idx]))
        assert not any ([train_trial in trial[test_idx] for train_trial in trial[train_idx]])

    print(score,n_splits)
    return np.array(score)

def n_to_idx(n,redundant_reference):
    return np.nonzero([ti in n for ti in redundant_reference])[0]

def get_n_random_trials_indices(trial, n):
    """
    Enter redundant trial data, and receive back the indices of n trials chosen randomly
    """
    if n < 1:
        n = int(n*len(trial))

    uniqueTrials = np.unique(trial)
    chosenTrials = np.random.permutation(uniqueTrials)[:n]

    return np.nonzero([ti in chosenTrials for ti in trial])[0]


def splitFirstNtrials(X, y, trial, n_before_split):
    uniqueTrials = np.unique(trial)

    return (X[trial< uniqueTrials[n_before_split],:], y[trial< uniqueTrials[n_before_split]], trial[trial< uniqueTrials[n_before_split]],
           X[trial>= uniqueTrials[n_before_split],:], y[trial>= uniqueTrials[n_before_split]], trial[trial>= uniqueTrials[n_before_split]])


def optimize(clf, ratWrapper, train_size, n_calls=200, n_random_starts=100, space=(), parameter_names=()):
    """
    Runs bayesian optimization on a
    """

    def objective_(params):
        parameters = dict(zip(parameter_names,params))
        score = -1*np.mean(shufflin_trial_score(clf(**parameters), ratWrapper.X, ratWrapper.y, ratWrapper.trial, train_size=train_size ))
        return score

    return gp_minimize(objective_, space,n_calls=n_calls,n_random_starts=n_random_starts, n_jobs=3,verbose=3)

if __name__ == '__main__':

    classifiers =   makeClassifierList()

    # Make directory to save results
    saveDir = 'xgboost_tuning'
    os.makedirs(saveDir)
    for classifier in classifiers[-1:]:
        os.mkdir('%s/%s'%(saveDir,classifier['name']))

    best_parameters_from_all_optimizations = pd.DataFrame(columns = ['rat', 'classifier', 'parameters','time'])
    results_from_each_classifier = pd.DataFrame(columns=['rat', 'classifier', 'kappa', 'corr','time'])
    number_of_trials_for_each_rat = []

    for rat_number in [7,8,9,10]:
        print(rat_number)
        ratWrapper = Rat(rat_number)
        ratWrapper.selecTrials({'minDuration':1000})
        ratWrapper.selecTimes(200,700, z_transform=True)

        number_of_trials_for_each_rat.append(ratWrapper.trialsToUse.sum())

        for classifier in classifiers[-1:]:
            clf = classifier['func']
            print(classifier['name'])

            if len(classifier['hyperparameter_names']) > 0: # there are parameters to optimize

                init_time = time.time()
                res = optimize(clf, ratWrapper=ratWrapper, train_size=200,
                     n_calls=classifier['n_calls']['opt'], n_random_starts=classifier['n_calls']['rand'],
                     space=classifier['hyperparameter_space'], parameter_names=classifier['hyperparameter_names'])
                end_time = time.time()


                # Save hyperparameter optimization results
                single_results = pd.DataFrame(
                                        {'rat'  :  rat_number,
                                        'classifier': classifier['name'],
                                        'parameters':[{parameter_name:res.x[i] for i, parameter_name in enumerate(classifier['hyperparameter_names'])}],
                                        'time': end_time-init_time
                                        },
                                        index = [rat_number]
                                            )

                dump(res, '%s/%s/results_minimization_rat%d.pickle'%(saveDir,classifier['name'],rat_number),store_objective=False)
                single_results.to_csv('%s/%s/best_parameters_rat%d.csv'%(saveDir,classifier['name'],rat_number))
                if classifier in ['rbfSVM']:#['knn', 'Neural network','linSVM', 'Decision tree']:
                    plot_objective(res,levels=20)
                    plt.savefig('%s/hyperparameter_surface_%d.png'%(saveDir,rat_number), bbox_inches='tight')

                best_parameters_from_all_optimizations = best_parameters_from_all_optimizations.append(single_results)
    best_parameters_from_all_optimizations.to_csv(saveDir+'/best_parameters_from_all_optimizations.csv')
