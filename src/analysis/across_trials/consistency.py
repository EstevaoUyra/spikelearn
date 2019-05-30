import pickle
import sys
sys.path.append('.')
from itertools import product
import numpy as np
from spikelearn import remove_baseline
from spikelearn.data import io, SHORTCUTS, to_feature_array, select
from spikelearn.models.mahalanobis import MahalanobisClassifier

DSETS = ['no_smoothing', 'no_smoothing_norm', 'wide_smoothed']
allsims = {}

def bad_trials(label, threshold=20, only_indexes=True):
    data = io.load(label, 'no_smoothing')
    if 'quality' in data.columns:
        bad = select(data, _min_quality=0).full.apply(lambda x: np.max(x)>threshold).unstack().any(axis=1)
    else:
        bad = select(data, is_selected=True).full.apply(lambda x: np.max(x)>threshold).unstack().any(axis=1)
        
    if only_indexes:
        return bad[bad.values].index.values
    else:
        return bad

remove_baseline_flag = True
    
for rat, dset in product(SHORTCUTS['groups']['eletro'], DSETS):
    print(rat, dset)
    data = select(io.load(rat, dset), _min_duration=1.5, _max_duration=4.5, is_tired=False)
    data = data[~data.reset_index().trial.isin(bad_trials(rat)).values]
    if rat in SHORTCUTS['groups']['DRRD']:
        data = select(data, is_selected=True)
    else:
        data = select(data, _min_quality=0)
    data = to_feature_array(data, Xyt=False, subset='full')
    if remove_baseline_flag:
        data = remove_baseline(data, io.load(rat, 'baseline'), .5)
    X, y, trial = data.values, data.reset_index().time, data.reset_index().trial
    uniquetrials = np.unique(trial)
    for i, tr_ in enumerate(uniquetrials[30:]):
        mah = MahalanobisClassifier(shared_cov=True)
        mah.fit(X[(trial > uniquetrials[i]) & (trial < tr_)], y[(trial > uniquetrials[i]) & (trial < tr_)])
        allsims[(rat, dset, tr_)] = mah.transform(X[trial == tr_], y[trial == tr_])
        
if remove_baseline_flag:
    pickle.dump(allsims, open('data/results/consistency_of_activity_no_baseline.pickle', 'wb'))
else:
    pickle.dump(allsims, open('data/results/consistency_of_activity.pickle', 'wb'))