from itertools import product, cycle, chain
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from spikelearn.measures.similarity import unit_similarity_evolution
from spikelearn.models.shuffle_decoding import shuffle_val_predict
from spikelearn.data import io, to_feature_array, select, SHORTCUTS

from sklearn.linear_model import LogisticRegression

import os

# Parameters
TMIN = 1.5;
n_trial_per_splits = 60
folder = 'data/results/across_trials/cross_predictions/'
DSETS = ['medium_smoothed', 'medium_smoothed_norm',
         'narrow_smoothed', 'narrow_smoothed_norm',
         'wide_smoothed']
NSPLITS = 5

# Prepare output folders
#[os.makedirs(folder+dset) for dset in DSETS]

for label, dset in product(SHORTCUTS['groups']['DRRD'], DSETS):
    subset = 'full' if 'norm' in dset else 'cropped'

    data_ = select( io.load(label, dset),
                    _min_duration=TMIN, is_selected=True ).reset_index()
    print(label, dset)

    trials = data_.trial.unique()
    slice_bounds = trials[::n_trial_per_splits]
    n_slices = len(slice_bounds) - 1
    dfs = [to_feature_array( select( data_,
                                    _min_trial = slice_bounds[i],
                                    _max_trial = slice_bounds[i+1]
                                    ).set_index(['trial','unit']),
                            False, subset) for i in range(n_slices)]
    names = np.arange(n_slices)
    C1, C2 = np.linspace(-1.5, 5, 20), np.linspace(-5, 5, 20)

    res_pred = pd.DataFrame()
    res_weight = pd.DataFrame()

    for logC, regl in chain( zip(C1, cycle(['l1'])), zip(C2, cycle(['l2']))):
        clf = LogisticRegression( C=10**logC )
        one_p, one_w = shuffle_val_predict( clf, [df.reset_index() for df in dfs], names,
                                 X=dfs[0].columns, y='time', group='trial',
                                 cv='sh', n_splits = NSPLITS)
        one_w['logC'] = logC
        one_w['regl'] = regl
        one_p['logC'] = logC
        one_p['regl'] = regl

        res_pred = res_pred.append(one_p)
        res_weight = res_weight.append(one_w)

    filename = '{}_cross_pred.csv'.format(label)
    res_pred.to_csv('{}{}/{}'.format(folder, dset, filename))
    filename = '{}_cross_weight.csv'.format(label)
    res_weight.to_csv('{}{}/{}'.format(folder, dset, filename))
