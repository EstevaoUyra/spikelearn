"""
For each rat

     time unit     value  shuffle     rat        dataset  when  num_trials
0     200    0 -0.038855        0  DRRD 7  wide_smoothed  init          20
1     300    0 -0.084481        0  DRRD 7  wide_smoothed  init          20
2     400    0  0.060537        0  DRRD 7  wide_smoothed  init          20
3     500    0  0.021759        0  DRRD 7  wide_smoothed  init          20
4     600    0 -0.336057        0  DRRD 7  wide_smoothed  init          20
...
173   500   17 -0.057553       10  DRRD 10 narrow_smoothed end          70
174   600   17 -0.240128       10  DRRD 10 narrow_smoothed end          70
175   700   17 -0.194336       10  DRRD 10 narrow_smoothed end          70
176   800   17  0.087844       10  DRRD 10 narrow_smoothed end          70
"""

# TODO: Threshold a partir dos p-valores
## Permutation test: 1000 bootstraps
## Non-parametric weight distribution
## Para fazer medida de performance, NÃO permutar teste.

# Usar gallistel
# D prime

# Saliency maps
## Keras - testar se existe p/ regressao log.
## permutation tests - like weights

import pandas as pd
import numpy as np
import sys
sys.path.append('.')


from spikelearn.models import shuffle_val_predict
from spikelearn.data import io
from spikelearn.data.selection import select, to_feature_array

from sklearn.linear_model import LogisticRegression
from itertools import product, chain


DRRD_RATS = ['DRRD 7','DRRD 8','DRRD 9','DRRD 10']
DATASETS = ['medium_smoothed']#['wide_smoothed', 'medium_smoothed', 'narrow_smoothed']
WHENS = ['init', 'end']
NUM_TRIALS = np.arange(10,100,5)
LOGCS = np.linspace(-1.5, 4, 20)


ANALYSIS_NTRIALS = product(DRRD_RATS, DATASETS, WHENS, NUM_TRIALS, [0])
ANALYSIS_REGUL = product(DRRD_RATS, DATASETS, WHENS, [50], LOGCS)
ANALYSIS = chain(ANALYSIS_NTRIALS, ANALYSIS_REGUL)

results = pd.DataFrame()
preds = pd.DataFrame()
acts = pd.DataFrame()
for rat, dataset, when, num_trials, logC in ANALYSIS:
    clf = LogisticRegression(C = 10**logC, penalty='l1')
    data = io.load(rat, dataset)
    units = data.groupby('is_selected').get_group(True).reset_index().unit.unique()
    data = select(data.reset_index(),
                        maxlen = num_trials*units.shape[0],
                        takefrom = when,
                        is_selected = True,
                        _min_duration = 1.5,
                        is_tired = False ).set_index(['trial','unit'])

    X, y, trial = to_feature_array(data)
    local_preds, local_results = shuffle_val_predict(clf, X, y, trial, cv='sh', get_weights = True, n_splits = 10)

    # Calculate mean and std activity
    activity = pd.DataFrame(X, index=pd.Index(y, name='time')).reset_index()
    mean = activity.groupby('time').mean()
    std = activity.groupby('time').std()/np.sqrt(np.unique(trial).shape[0])

    mean = mean.reset_index().melt(id_vars='time', var_name='unit', value_name='mean').set_index(['unit', 'time'])
    std = std.reset_index().melt(id_vars='time', var_name='unit', value_name='std').set_index(['unit', 'time'])

    # Save mean activity
    local_act = mean.join(std)
    local_act['rat'] = rat
    local_act['dataset'] = dataset
    local_act['when'] = when
    local_act['num_trials'] = num_trials
    local_act['logC'] = logC

    # Save weights
    local_results['rat'] = rat
    local_results['dataset'] = dataset
    local_results['when'] = when
    local_results['num_trials'] = num_trials
    local_results['logC'] = logC

    # Save predictions
    local_preds['rat'] = rat
    local_preds['dataset'] = dataset
    local_preds['when'] = when
    local_preds['num_trials'] = num_trials
    local_preds['logC'] = logC

    results = results.append(local_results)
    preds = preds.append(local_preds)
    acts = acts.append(local_act)

import os
#os.makedirs('data/results/changepoint')
results.to_csv('data/results/across_trial/feature_importance.csv')
preds.to_csv('data/results/across_trial/feature_importance_predictions.csv')
acts.to_csv('data/results/across_trial/feature_importance_activity.csv')
