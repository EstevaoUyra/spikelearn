"""
"""
import numpy as np
import pandas as pd

import sys
sys.path.append('.')
from spikelearn.data import io, SHORTCUTS

# Load into DataFrames each data
for rat in SHORTCUTS:
    behav = io.load(rat, 'behavior')
    behav['intertrial_interval'] = np.hstack((0,behav.onset.values[1:]-behav.offset.values[:-1]))

    tiredness = io.load(rat, 'tiredness').values[0,0] if io.dataset_exist(rat, 'tiredness') else float('inf')
    behav['is_tired'] = behav.index > tiredness
    io.save(behav, rat, 'behav_stats', 'interim')
