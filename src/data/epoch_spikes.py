import numpy as np
import pandas as pd

import os
#os.chdir('../../')
import sys
sys.path.append('.')
from spikelearn.dataprocessing import io, SHORTCUTS


# Load into DataFrames each data
for rat in SHORTCUTS:
    spikes = io.load(rat, 'spikes')
    trials = np.unique(np.hstack(spikes.trial.apply(np.unique))).astype(int)
    epoched = np.array([[spikes.trial_time[iunit][spikes.trial[iunit]==itrial] for itrial in trials] for iunit in spikes.index ] )


    epoched = pd.DataFrame(epoched, columns= pd.Index(trials,name='trial'),
                            index=pd.Index(spikes.index, name='unit')).reset_index().melt('unit', value_name='time')

    behavior = io.load(rat, 'behavior')
    for trial_feat in behavior.columns:
        epoched[trial_feat] = epoched.apply(lambda x: behavior[trial_feat][x.trial], axis=1)

    selected_neurons = io.load(rat, 'selected_neurons') if io.dataset_exist(rat,'selected_neurons') else np.array([-1])

    epoched['with_baseline'] = epoched['time']
    epoched['baseline'] = epoched.apply(lambda x: [x.time[x.time<=0]] if len(x.time)>0 else [[]], axis=1).apply(lambda x: x[0])
    epoched['time'] = epoched.apply(lambda x: [x.time[x.time>0]] if len(x.time)>0 else [[]], axis=1).apply(lambda x: x[0])
    epoched['time_from_offset'] = epoched.apply(lambda x: [x.time - x.duration] if len(x.time)>0 else [[]], axis=1).apply(lambda x: x[0])
    epoched['normalized_time'] = epoched.apply(lambda x: [x.time[x.time>0]/x.duration] if len(x.time)>0 else [[]], axis=1).apply(lambda x: x[0])

    epoched['is_selected'] = epoched.unit.apply(lambda x: x in selected_neurons)

    io.save(epoched.set_index('trial'), rat, 'epoched_spikes', 'processed')
