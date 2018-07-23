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
import h5py


def spikes_behavior_from_ez(filename):
    def behav_to_df(f, is_h5=True):
        if is_h5:
            behav = pd.DataFrame({'duration':f['behavior']['DRRD'][:].reshape(-1),
            'offset':f['behavior']['NPEnd'][:].reshape(-1),
            'onset':f['behavior']['NPStart'][:].reshape(-1)}, index=pd.Index(np.arange(f['behavior']['DRRD'].shape[1])+1, name='trial'))
        else:
            behav = pd.DataFrame({'duration':f['behavior']['DRRD'][0,0].reshape(-1),
        'offset':f['behavior']['NPEnd'][0,0].reshape(-1),
        'onset':f['behavior']['NPStart'][0,0].reshape(-1)}, index=pd.Index(np.arange(f['behavior']['DRRD'][0,0].shape[0])+1, name='trial'))
        
        assert not any(behav.duration - (behav.offset-behav.onset)> 1e-10), 'There are inconsistencies in duration'
        return behav
        
    def spikes_inside(times, onset, offset, baseline = .5):
        return times[np.logical_and(times>(onset-baseline), times<offset)]

    def relevant_spikes(times, behavior):
        spk, trials = [], []
        for trial, row in behavior.iterrows():
            aux_spk = list(spikes_inside(times, row.onset, row.offset))
            spk += aux_spk
            trials += [trial for i in range(len(aux_spk))]

        return np.array(spk), np.array(trials)
    try:
        behavior = behav_to_df(h5py.File('%s/Behavior.mat'%filename, 'r'))
    except:
        behavior = behav_to_df(loadmat('%s/Behavior.mat'%filename), is_h5=False)
    mat = loadmat('%s/spikes/openephys/openephys.spikes.cellinfo.mat'%filename)['spikes'][0,0]

    quality = pd.read_csv('%s/spikes/openephys/cluster_quality.tsv'%filename, '\t')

    infos = pd.DataFrame(mat[4].squeeze(), columns=['waveforms']).join(quality)
    infos['area'] = np.hstack(mat[6].squeeze())

    spikes = pd.DataFrame(mat[1].squeeze(),
                            columns=['times']).applymap(np.hstack).join(infos)
    spikes['trial'] = spikes.times.apply(lambda x: relevant_spikes(x,
                                                                behavior)[1])
    spikes['times'] = spikes.times.apply(lambda x: relevant_spikes(x,
                                                                behavior)[0])

    # Calculate relative spike time
    spikes['trial_time'] = pd.DataFrame(np.transpose([spikes.times[i] - behavior.iloc[spikes.trial[i]-1].onset.as_matrix() for i in range(spikes.shape[0])]))
    
    return spikes, behavior

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
    if rat in SHORTCUTS['groups']['GB']:
        spikes, behavior = spikes_behavior_from_mat(filepath)
    elif rat in SHORTCUTS['groups']['EZ']:
        print(filepath)
        spikes, behavior = spikes_behavior_from_ez(filepath)
    else:
        raise NotImplementedError('This dataset is not included as a special case')

    identifiers = dict(session=rat.split()[0], rat_number=rat.split()[1] )
    io.save(spikes, rat, 'spikes', 'interim', **identifiers)
    io.save(behavior, rat, 'behavior', 'interim', **identifiers)
