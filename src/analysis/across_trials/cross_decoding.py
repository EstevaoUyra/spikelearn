from itertools import product, cycle, chain
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from spikelearn.measures.similarity import unit_similarity_evolution
from spikelearn.models.shuffle_decoding import shuffle_val_predict
from spikelearn.data import io, to_feature_array, select, SHORTCUTS, remove_baseline

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

import os

# Parameters
TMIN = 1.5;
n_trial_per_splits = 100
folder = 'data/results/across_trials/cross_decoding/'
DSETS = ['narrow_smoothed', 'narrow_smoothed_norm']#['medium_smoothed', 'medium_smoothed_norm',
        # 'narrow_smoothed', 'narrow_smoothed_norm',
        # 'wide_smoothed']
NSPLITS = 100
subset = 'cropped'
C1, C2 = np.linspace(-1.5, 5, 3), np.linspace(-5, 5, 3)
BASELINE_SIZE = .5


# Prepare output folders
#[os.makedirs(folder+dset) for dset in DSETS]

for label, dset in product(SHORTCUTS['groups']['DRRD'], DSETS):

    data_ = select( io.load(label, dset),
                    _min_duration=TMIN, is_selected=True ).reset_index()
    print(label, dset)

    # Divide dataset into parts
    ## Define how the separation will happen
    trials = data_.trial.unique()
    slice_bounds = trials[::n_trial_per_splits]
    n_slices = len(slice_bounds) - 1
    ## Enforce separation
    dfs = [to_feature_array( select( data_,
                                    _min_trial = slice_bounds[i],
                                    _max_trial = slice_bounds[i+1]
                                    ).set_index(['trial','unit']),
                            False, subset) for i in range(n_slices)]

    #baseline = io.load(label, 'epoched_spikes')
    #baseline = baseline.baseline.unstack('unit').applymap(len)/BASELINE_SIZE
    #dfs = [remove_baseline(df) for df in dfs]



    res_pred = pd.DataFrame()
    res_weight = pd.DataFrame()
    res_stats = pd.DataFrame()
    for logC, regl in chain( zip(C1, cycle(['l1'])), zip(C2, cycle(['l2']))):
        clf = LogisticRegression( C=10**logC )
        id_kwargs = {'logC':logC, 'penalty':regl}
        res = shuffle_val_predict( clf, dfs, n_splits = NSPLITS
                                    id_kwargs=id_kwargs)

        res_pred = res_pred.append(one_p)
        res_weight = res_weight.append(one_w)
        res_stats = res_stats.append(one_s)


    filename = '{}_cross_pred.csv'.format(label)
    res_pred.to_csv('{}{}/{}'.format(folder, dset, filename))
    filename = '{}_cross_weight.csv'.format(label)
    res_weight.to_csv('{}{}/{}'.format(folder, dset, filename))
    filename = '{}_cross_stats.csv'.format(label)
    res_stats.to_csv('{}{}/{}'.format(folder, dset, filename))
