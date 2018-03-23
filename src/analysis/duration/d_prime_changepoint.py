import pandas as pd
import numpy as np
import scipy.stats as st

import sys
sys.path.append('.')
import os

from spikelearn.data import io, to_feature_array, select, SHORTCUTS

# Directory
savedir = 'data/results/duration/d_prime'
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Parameters
DSETS = ['narrow_smoothed', 'narrow_smoothed_norm', 'medium_smoothed',
        'medium_smoothed_norm', 'narrow_smoothed', 'narrow_smoothed_norm',
        'wide_smoothed']

for label, dset in SHORTCUTS['groups']['DRRD']:

    # Load necessary data
    data = io.load(label, 'narrow_smoothed')
    data = select(data, is_tired = False) # Remove unwanted trials
    data = select(data, is_selected = True) # Remove unwanted unit



    # Get baseline
    baseline_idx = np.where( r.full_times.iloc[0] < 0 )[0]
    data['baseline'] = data.full.apply( lambda x: x[baseline_idx] )
    for i in baseline_idx:
        data['b%d'%i] = data['baseline'].values[i]
    b_idx = ['b%d'%i for i in baseline_idx]



    # Calculate D' for each


    # Calculate D'
    Dprime = pd.DataFrame(index=pd.Index(data.unit.unique(),name='unit') )
    for variable in ['is_short', 'is_init']:
        data_means = data.groupby([variable, 'unit']).mean().baseline
        data_var = data.groupby([variable, 'unit']).var().baseline

        diff_means = data_means.loc[False] - data_means.loc[True]
        pool_std = np.sqrt( data_var.loc[False] +
                            data_var.loc[True] )

        Dprime[variable + ' D\''] = diff_means.abs()/pool_std


    filename = '{}_Dprime.csv'.format(label)
    Dprime.to_csv('{}/{}'.format(savedir, filename))
