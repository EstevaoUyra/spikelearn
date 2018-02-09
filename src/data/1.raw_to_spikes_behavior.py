"""
Transforms the .mat file outputted from the spikesorting into two DataFrames,
saving each into pickle files, and adding them to shortcuts,
with dataset names 'spikes' and 'behavior'
"""

import sys
sys.path.append('.')
from scipy.io import loadmat
from spikelearn.data import io, SHORTCUTS

import pandas as pd
import numpy as np

def spikes_behavior_from_mat(filename):
    """
    Loads a mat-file into two DataFrames

    Parameters
    ----------

    Returns
    -------
    spikes : DataFrame (n_units, 3)
        Contains three ndarray fields, indexed by the unit (neuron).
        Each ndarray has the form (n_spikes_i,) being different for each row.
        'times' holds the absolute times of each spike from the session begin.
        'trial' holds the trial number of each corresponding spike from times.
        'trial_time' has the relative time of each spike from trial onset.

    behavior : DataFrame (n_trials, 3)
        Contains five number fields of trial attributes.
        'onset' is the time of trial beginning
        'offset' is the end of the trial
        'duration' is equal to offset - onset
    """
    #TODO decide if it would make sense to keep sortIdx and sortLabel
    data = loadmat(filename)

    spikes = data['dados'][0,0][1]
    behavior = data['dados'][0,0][0]

    spikes = pd.DataFrame([[ spikes[0,i][0][:,0], spikes[0,i][0][:,1]] for i in range(spikes.shape[1]) if spikes[0,i][0].shape[1]==2], columns=['times','trial'])

    behavior = pd.DataFrame(np.transpose(behavior[0,0][0]),   columns=['one','onset','offset','zero', 'duration', 'sortIdx', 'sortLabel']).drop(['one', 'zero', 'sortIdx', 'sortLabel'], axis=1)
    behavior['trial'] = np.arange(1, behavior.shape[0]+1)
    behavior=behavior.set_index('trial')

    # Calculate relative spike time
    spikes['trial_time'] = pd.DataFrame(np.transpose([spikes.times[i] - behavior.iloc[spikes.trial[i]-1].onset.as_matrix() for i in range(spikes.shape[0])]))

    return spikes, behavior

# Load into DataFrames each data
for rat in SHORTCUTS['groups']['ALL']:
    filepath = io.load(rat, 'spikesorted', getpath=True)
    spikes, behavior = spikes_behavior_from_mat(filepath)

    identifiers = dict(session=rat.split()[0], rat_number=rat.split()[1] )
    io.save(spikes, rat, 'spikes', 'interim', **identifiers)
    io.save(behavior, rat, 'behavior', 'interim', **identifiers)
