import pandas as pd
import numpy as np
import scipy.stats as st
from spikelearn.measures.dprime import cohen_d
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
T_short_MAX, T_long_MIN = 1, 1
wsize = 60

for label in SHORTCUTS['groups']['DRRD']:

    print(label)
    # Load necessary data
    data = io.load(label, 'narrow_smoothed')
    data = select(data, is_tired = False) # Remove unwanted trials
    data = select(data, is_selected = True) # Remove unwanted unit
    cp = io.load(label, 'changepoint')
    data['before_cp'] = (data.reset_index().trial < cp.gallistel[0]).values

    # Separate trials long and short
    data['is_short'] = (data.duration < T_short_MAX)
    data['is_long'] =  (data.duration >= T_long_MIN)
    data = data[data.is_short | data.is_long].reset_index()

    # Get baseline
    baseline_idx = np.where( data.full_times.iloc[0] < 0 )[0]
    data['baseline'] = data.full.apply( lambda x: x[baseline_idx] )
    b_id = data.full_times.iloc[0][baseline_idx]
    for i, id_ in zip(baseline_idx, b_id):
        data[id_] = np.vstack(data['baseline'].values)[:,i]

    # Calculate D' for each
    units = data.reset_index().unit.unique()
    data = data.reset_index()
    res = data.melt(id_vars = ['trial', 'unit', 'before_cp', 'is_short'],
                      var_name = 'time', value_vars = b_id, value_name = 'fr')
    res = res.set_index(['unit','before_cp', 'time'])

    res['cp_d'] = np.nan
    res = res.sort_index()
    for time, unit, cp in product(b_id, units, [True, False]):
        selected = data.groupby(['unit','before_cp']).get_group((unit, cp))

        res.loc[(unit, cp, time), 'cp_d'] = cohen_d(time, 'is_short', selected)

    #wres = res.set_index('trial',append=True)[['is_short','fr']]
    #wres['both'] = list(zip(wres.is_short.values, wres.fr.values))
    #wres = wres.both.unstack(['unit','time'])zzz
    #cohen = lambda arr: cohen_d(np.vstack(arr)[:,1], np.vstack(arr)[:,0])
#    ww = wres.rolling(30).apply(cohen)


    res = res.reset_index('before_cp').set_index('trial',append=True).sort_index()
    res['window_d'] = np.nan
    # Rolling D
    for trialmin in data.trial.unique()[:-wsize-1] :
        toi = np.arange(trialmin, trialmin+wsize)
        if trialmin%10==0: print(trialmin)
        for time, unit in product(b_id, units):
            selected = data[data.trial.isin(toi)].groupby('unit').get_group(unit)
            res.loc[(unit, time, trialmin), 'window_d'] = cohen_d(time, 'is_short', selected)




    filename = '{}_Dprime_cp_init.csv'.format(label)
    res.to_csv('{}/{}'.format(savedir, filename))
    #filename = '{}_mean_activity_cp.csv'.format(label)
    #act_res.to_csv('{}/{}'.format(savedir, filename))
