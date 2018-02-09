"""
This script runs optimization on SVM RBF hyperparameters C and gamma.
When loaded as a module, it provides the optimize function
"""
#import sys
#sys.path.append('/home/tevo/Documents/UFABC/Spikes')
import time
import argparse
import os

import numpy as np
import pandas as pd
from skopt import gp_minimize, dump
import sys
sys.path.append('.')

from sklearn.base import clone
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit
from makeClassifierList import makeClassifierList

from spikelearn.data.selection import select, to_feature_array
from spikelearn.data import io

print('teste', 10)

kappa = make_scorer(cohen_kappa_score, weights = 'quadratic')

def normalized_distance(mean, observations, max_qdist):
    norm_dist = np.sum((observations-mean)**2)/max_qdist
    return 1 - norm_dist/len(observations)

def multiclass_pessimistic_score(y_true, y_pred, composite='min',
                        one_class_score_func=normalized_distance, labels=None):
    if labels is None:
        labels = np.unique(y_true)

    each_score = np.array([one_class_score_func(y, y_pred[y_true==y], max(y-min(labels), max(labels)-y)**2 ) for y in labels])

    if composite == 'geometric':
        return np.prod(each_score)**(1/len(each_score))
    elif composite == 'harmonic':
        eps = 1e-6
        return len(each_score)/(np.sum(1/(each_score+eps)))
    elif composite == 'min':
        return min(each_score)

score = make_scorer(multiclass_pessimistic_score)

RAT_LABELS = ['DRRD %d'%i for i in [7,8,9,10]]



def shufflin_trial_score(clf, X,y, trial, train_size=30, n_splits = 10,scoring=score, verbosity=0):
    sh = GroupShuffleSplit(n_splits=n_splits, train_size=train_size,test_size=.2)
    if verbosity >= 3:
        print(clf.get_params())

    score=[]
    for train_idx, test_idx in sh.split(X, y, trial):
        clf_local = clone(clf)
        clf_local.fit(X[train_idx,:],y[train_idx])
        score.append(scoring(clf_local, X[test_idx,:], y[test_idx]))
        assert not any ([train_trial in trial[test_idx] for train_trial in trial[train_idx]])

    if verbosity>=2:
        print('Scored {} in the {} splits'.format(score,n_splits))
    return np.array(score)

def splitFirstNtrials(X, y, trial, n_before_split):
    uniqueTrials = np.unique(trial)

    return (X[trial< uniqueTrials[n_before_split],:], y[trial< uniqueTrials[n_before_split]], trial[trial< uniqueTrials[n_before_split]],
           X[trial>= uniqueTrials[n_before_split],:], y[trial>= uniqueTrials[n_before_split]], trial[trial>= uniqueTrials[n_before_split]])


def optimize(clf, X, y, trial, train_size, n_calls=200, n_random_starts=100, space=(), parameter_names=(), verbosity=3):
    """
    Runs bayesian optimization on a
    """

    def objective_(params):
        parameters = dict(zip(parameter_names,params))
        score = -1*np.mean(shufflin_trial_score(clf(**parameters), X, y, trial, train_size=train_size ))
        return score

    return gp_minimize(objective_, space,n_calls=n_calls,n_random_starts=n_random_starts, n_jobs=3,verbose=verbosity)

if __name__ == '__main__':

    classifiers =   makeClassifierList()
    clfnames = [clf['name'] for clf in classifiers]
    parser = argparse.ArgumentParser(description='Get classifiers and directory\n Available classifiers are {}'.format(clfnames))

    parser.add_argument('--classifiers', '-clf', nargs='+', default='all')
    parser.add_argument('--output', '-o', default='hyperparameter_opt')
    parser.add_argument('--overwrite', '-ow', action='store_true')
    parser.add_argument('--verbose', '-v', action='count')
    kw = parser.parse_args()
    verbosity = kw.verbose

    # Select and report classifiers
    required_classifiers = sys.argv
    if verbosity >= 1:
        print('Required classifiers:', kw.classifiers)
    if kw.classifiers != 'all':
        classifiers = [clfdict for clfdict in classifiers if clfdict['name'] in required_classifiers]
    if verbosity >= 1:
        print('Enabled classifiers:', [clf['name'] for clf in classifiers])

    # Make directory to save results
    saveDir = kw.output
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    elif not kw.overwrite:
        raise OSError("Output directory {} already exists".format(saveDir))
    else:
        if verbosity >= 1:
            print('Overwritting files in directory {}'.format(saveDir))

    for classifier in classifiers:
        clfdir = '%s/%s'%(saveDir,classifier['name'])
        if not os.path.exists(clfdir):
            os.mkdir(clfdir)
        elif not kw.overwrite:
            raise OSError("Output directory {} already exists".format(clfdir))
        else:
            if verbosity >= 1:
                print('Overwritting files in directory {}'.format(clfdir))

    best_params_of_all = pd.DataFrame()
    number_of_trials_for_each_rat = []

    for rat_label in RAT_LABELS:
        if verbosity >= 1:
            print('Working in rat', rat_label)
        data = io.load(rat_label, 'wide_smoothed')
        # Z-score?
        data = select(data, _min_duration=1.5)
        X, y, trial = to_feature_array(data)

        number_of_trials_for_each_rat.append(np.unique(trial).shape[0])

        for classifier in classifiers:
            clf = classifier['func']
            if verbosity >= 1:
                print(classifier['name'])

            if len(classifier['hyperparameter_names']) > 0: # there are parameters to optimize

                init_time = time.time()
                res = optimize(clf, X, y, trial, train_size=.8,
                     n_calls=classifier['n_calls']['opt'],
                      n_random_starts=classifier['n_calls']['rand'],
                     space=classifier['hyperparameter_space'], parameter_names=classifier['hyperparameter_names'])
                end_time = time.time()


                # Save hyperparameter optimization results
                params = {parameter_name:res.x[i] for i, parameter_name in
                            enumerate( classifier['hyperparameter_names']) }
                single_results = pd.DataFrame(params, index = pd.Index([rat_label], name='label'))
                io.save(single_results, rat_label, 'hyperopt/%s'%classifier['name'], 'results')

                dump(res, '%s/%s/results_minimization_%s.pickle'%(saveDir,classifier['name'],rat_label),store_objective=False)

                single_results['clf'] = classifier['name']
                single_results = single_results.reset_index().set_index(['clf','label'])
                best_params_of_all = best_params_of_all.append(single_results)
        best_params_of_all.to_csv(saveDir+'/best_params_of_all.csv')
