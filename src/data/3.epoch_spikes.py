"""
From the DataFrame of shape (n_units, 3),
generates a new one with (n_units x n_trials, ),
epoching the spike trains of each trial, and saving each to
dataset name epoched_spikes
"""
import numpy as np
import pandas as pd

import sys
sys.path.append('.')
from spikelearn.data import io, SHORTCUTS

# Load into DataFrames each data
for rat in SHORTCUTS['groups']['ALL']:
    spikes = io.load(rat, 'spikes')
    trials = np.unique(np.hstack(spikes.trial.apply(np.unique))).astype(int)
    epoched = np.array([[spikes.trial_time[iunit][spikes.trial[iunit]==itrial] for itrial in trials] for iunit in spikes.index ] )


    epoched = pd.DataFrame(epoched, columns= pd.Index(trials,name='trial'),
                            index=pd.Index(spikes.index, name='unit')).reset_index().melt('unit', value_name='time').set_index('trial')

    # Make identifiers
    behav = io.load(rat, 'behav_stats')
    epoched = epoched.join(behav)

    # Importing manual selection
    if io.dataset_exist(rat,'selected_neurons'):
        selection_neurons = io.load(rat, 'selected_neurons')
        epoched['is_selected'] = epoched.unit.apply(lambda x: x in selection_neurons)
        epoched['comments'] = ''
        #selected_neurons = selection_neurons.groupby('selected').get_group(' yes')
        #epoched['is_selected'] = epoched.unit.apply(lambda x: x in selected_neurons.index)
        #epoched['comments'] = epoched.unit.apply(lambda x: (selection_neurons.loc[x,' comments'] if x in selection_neurons.index else ''))

    # Make redundant yet useful data
    epoched['with_baseline'] = epoched['time']
    epoched['baseline'] = epoched.apply(lambda x: [x.time[x.time<=0]] if len(x.time)>0 else [[]], axis=1).apply(lambda x: x[0])
    epoched['time'] = epoched.apply(lambda x: [x.time[x.time>0]] if len(x.time)>0 else [[]], axis=1).apply(lambda x: x[0])
    epoched['time_from_offset'] = epoched.apply(lambda x: [x.time - x.duration] if len(x.time)>0 else [[]], axis=1).apply(lambda x: x[0])
    epoched['normalized_time'] = epoched.apply(lambda x: [x.time[x.time>0]/x.duration] if len(x.time)>0 else [[]], axis=1).apply(lambda x: x[0])

    io.save(epoched.reset_index().set_index(['trial', 'unit']), rat, 'epoched_spikes', 'processed')
