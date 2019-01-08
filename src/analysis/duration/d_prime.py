import pandas as pd
import numpy as np
import scipy.stats as st

import sys
sys.path.append('.')
import os

from spikelearn.data import io, to_feature_array, select, SHORTCUTS

# Directory
savedir = 'data/results/duration/d_prime2'
if not os.path.exists(savedir):
    os.makedirs(savedir)

# Parameters
T_short_MAX, T_long_MIN = 1.5, 1.5

for label in SHORTCUTS['groups']['DRRD']:

    # Load necessary data
    data = io.load(label, 'epoched_spikes')
    data = select(data, is_tired = False) # Remove unwanted trials
    data = select(data, is_selected = True) # Remove unwanted unit

    # Separate trials long and short
    data['is_short'] = (data.duration < T_short_MAX)
    data['is_long'] =  (data.duration > T_long_MIN)
    data = data[data.is_short | data.is_long].reset_index()

    # Separate along session moment
    data['is_init'] = data.trial < data.trial.quantile(.5)

    # Number of spikes during baseline
    data['baseline'] = data.baseline.apply(len)

    # Calculate D'
    Dprime = pd.DataFrame(index=pd.Index(data.unit.unique(),name='unit') )
    for variable in ['is_short', 'is_init']:
        data_means = data.groupby([variable, 'unit']).mean().baseline
        data_var = data.groupby([variable, 'unit']).var().baseline

        diff_means = data_means.loc[False] - data_means.loc[True]
        pool_std = np.sqrt( data_var.loc[False] +
                            data_var.loc[True] )

        Dprime[variable + ' D\''] = diff_means.abs()/pool_std

    # Calculate detrended Dprime
    ## For duration
    sess, dur = 'is_init D\'', 'is_short D\''
    det, det_ses = 'detrended Duration D\'', 'detrended Trial Index D\''

    Dprime['regress'] = np.polyval( np.polyfit(Dprime[sess],
                                               Dprime[dur], 1 ),
                                    Dprime[sess] )
    Dprime[det] = Dprime[dur].values - Dprime['regress']

    ## For trial index
    reg_ses = np.polyval( np.polyfit(Dprime[dur],
                                 Dprime[sess], 1 ),
                          Dprime[dur] )
    Dprime[det_ses] = Dprime[sess].values - reg_ses


    # Best D' activity
    activity = io.load(label, 'narrow_smoothed_viz')
    activity = to_feature_array(activity, False, subset='full')
    for Dp, name in [(det, 'duration'), (det_ses, 'index')]:
        act = activity.unstack('trial').transpose().reset_index('trial').loc[Dprime[Dp].idxmax()]
        act = act[act.trial.isin(data.trial)]
        act['is_short'] = data.drop_duplicates('trial').set_index('trial').loc[act.trial].is_short.values
        act['is_init'] = act.trial < act.trial.quantile(.5)
        act.set_index('trial').to_csv('{}/{}_{}'.format(savedir, label, name, Dprime[Dp].idxmax()))
    filename = '{}_Dprime.csv'.format(label)
    Dprime.to_csv('{}/{}'.format(savedir, filename))
