import os
#os.chdir('../../')
import sys
sys.path.append('.')

from spikelearn.dataprocessing import io, SHORTCUTS

# Load into DataFrames each data
for rat in SHORTCUTS:
    filepath = io.load(rat, 'spikesorted', getpath=True)
    spikes, behavior = io.spikes_behavior_from_mat(filepath)

    identifiers = dict(session=rat.split()[0], rat_number=rat.split()[1] )
    io.save(spikes, rat, 'spikes', 'interim', **identifiers)
    io.save(behavior, rat, 'behavior', 'interim', **identifiers)
