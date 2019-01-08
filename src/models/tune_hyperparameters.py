"""
This script runs optimization on SVM RBF hyperparameters C and gamma.
When loaded as a module, it provides the optimize function
"""
import time
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from skopt import gp_minimize, dump
import sys
import pickle
sys.path.append('.')

from sklearn.base import clone
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.model_selection import GroupShuffleSplit

from makeClassifierList import makeClassifierList

from spikelearn.data import select, to_feature_array, frankenstein, get_each_other
from spikelearn.data import io, SHORTCUTS
from spikelearn.models import shuffle_val_predict

def optimize(clf, df, train_size, n_calls=200, n_random_starts=100, space=(), parameter_names=(), verbosity=3):
    """
    Runs bayesian optimization 
    """

    def objective_(params):
        parameters = dict(zip(parameter_names,params))
        res = shuffle_val_predict(clf(**parameters), [df], 
                                  train_size=train_size, n_splits=5, feature_scaling='robust')
        score = -1 * res.score.pearson_max.mean()
        return score

    return gp_minimize(objective_, space, n_calls=n_calls, n_random_starts=n_random_starts, 
                       n_jobs=30,verbose=verbosity)

# if __name__ == '__main__':

classifiers =   makeClassifierList()
clfnames = [clf['name'] for clf in classifiers]
parser = argparse.ArgumentParser(description='Get classifiers and directory\n Available classifiers are {}'.format(clfnames))

parser.add_argument('--classifiers', '-clf', nargs='+', default='all')
parser.add_argument('--output', '-o', default='hyperparameter_opt')
parser.add_argument('--overwrite', '-ow', action='store_true')
parser.add_argument('--verbose', '-v', action='count')
kw = parser.parse_args()
verbosity = kw.verbose if kw.verbose is not None else 0

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

# Merging rats
DR = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['DRRD']]
EZ = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['EZ']]

sp_pfc = frankenstein(DR, _min_duration=1.5, is_selected=True, is_tired=False).reset_index().groupby('trial')\
                                                    .filter(lambda df: df.trial.values[0]%2==1).set_index(['trial','time'])
ez_pfc = frankenstein(EZ, _min_duration=1.5, _min_quality=0, area='PFC').reset_index().groupby('trial')\
                                                    .filter(lambda df: df.trial.values[0]%2==1).set_index(['trial','time'])
ez_str = frankenstein(EZ, _min_duration=1.5, _min_quality=0, area='STR').reset_index().groupby('trial')\
                                                    .filter(lambda df: df.trial.values[0]%2==1).set_index(['trial','time'])
merged_rats = [sp_pfc, ez_pfc, ez_str]

for rat_label, df in zip(['sp_pfc', 'ez_pfc', 'ez_str'],
                         [sp_pfc,   ez_pfc,   ez_str]):
    if verbosity >= 1:
        print('Working in rat', rat_label)

    for classifier in classifiers:
        clf = classifier['func']
        if verbosity >= 1:
            print(classifier['name'])

        if len(classifier['hyperparameter_names']) > 0: # there are parameters to optimize

            init_time = time.time()
            res = optimize( clf, df, train_size=.8,
                              n_calls=classifier['n_calls']['opt'],
                              n_random_starts=classifier['n_calls']['rand'],
                              space=classifier['hyperparameter_space'], 
                              parameter_names=classifier['hyperparameter_names'])
            end_time = time.time()

            # Save hyperparameter optimization results
            params = dict(zip(classifier['hyperparameter_names'], res.x))
            func_vals = res.func_vals
            pickle.dump({'best_params':params, 'func_vals':func_vals, 'time':end_time-init_time}, 
                        open('{}/{}/{}_dict_res.pickle'.format(saveDir, classifier['name'], rat_label), 'wb'))
            
            single_results = pd.DataFrame(params, index = pd.Index([rat_label], name='label'))
#             io.save(single_results, rat_label, 'hyperopt/%s'%classifier['name'], 'results')

            dump(res, '%s/%s/results_minimization_%s.pickle'%(saveDir,classifier['name'],rat_label),store_objective=False)

            single_results['clf'] = classifier['name']
            single_results = single_results.reset_index().set_index(['clf','label'])
            best_params_of_all = best_params_of_all.append(single_results)
    pickle.dump(best_params_of_all, open(saveDir+'/best_params_of_all.csv','wb'))
