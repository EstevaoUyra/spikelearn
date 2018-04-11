import pandas as pd
import numpy as np
import scipy.stats as st
from spikelearn.measures.dprime import dprime
from itertools import product

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

# Parameters
T_short_MAX, T_long_MIN = 0.8, 1.2

for label in SHORTCUTS['groups']['DRRD']:

    # Load necessary data
    data = io.load(label, 'narrow_smoothed')
    data = select(data, is_tired = False) # Remove unwanted trials
    data = select(data, is_selected = True) # Remove unwanted unit
    cp = io.load(label, 'changepoint')
    data['before_cp'] = (data.reset_index().trial < cp.gallistel[0]).values

    # Separate trials long and short
    data['is_short'] = (data.duration < T_short_MAX)
    data['is_long'] =  (data.duration > T_long_MIN)
    data = data[data.is_short | data.is_long].reset_index()


    # Get baseline
    baseline_idx = np.where( data.full_times.iloc[0] < 0 )[0]
    data['baseline'] = data.full.apply( lambda x: x[baseline_idx] )
    for i in baseline_idx:
        data['b%d'%i] = np.vstack(data['baseline'].values)[:,i]
    b_idx = ['b%d'%i for i in baseline_idx]

    # Calculate D' for each
    units = data.reset_index().unit.unique()
    res={True : pd.DataFrame(columns = b_idx,
                             index= pd.Index(units, name='unit')),
         False : pd.DataFrame(columns = b_idx,
                               index= pd.Index(units, name='unit')) }
    for time, unit, cp in product(b_idx, units, [True, False]):
        selected = data.reset_index().groupby(['unit','before_cp']).get_group((unit, cp))
        res[cp].loc[unit, time] = dprime(selected, time, 'is_short')

    res[True]['before_cp'] = True
    res[False]['before_cp'] = False
    res = pd.concat(( res[True], res[False] ))

    #act_res = data.reset_index().groupby(['unit','before_cp']).mean()[b_]


    filename = '{}_Dprime_cp_init.csv'.format(label)
    res.to_csv('{}/{}'.format(savedir, filename))
    #filename = '{}_mean_activity_cp.csv'.format(label)
    #act_res.to_csv('{}/{}'.format(savedir, filename))
