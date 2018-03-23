"""Takes off the edges and predicts, until there is nothing left."""

from itertools import product, cycle, chain
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from spikelearn.measures.similarity import unit_similarity_evolution
from spikelearn.models.shuffle_decoding import shuffle_val_predict
from spikelearn.data import io, to_feature_array, select, SHORTCUTS, remove_baseline

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

import os
import pickle

# Parameters
TMIN = 1.5;
folder = 'data/results/across_trials/edge_crop_decoding/'
if not os.path.exists(folder):
    os.makedirs(folder)

DSETS = ['narrow_smoothed', 'narrow_smoothed_norm']#['medium_smoothed', 'medium_smoothed_norm',
        # 'narrow_smoothed', 'narrow_smoothed_norm',
        # 'wide_smoothed']
NSPLITS = 30
subset = 'cropped'

clf = LogisticRegression()
for label, dset in product( SHORTCUTS['groups']['DRRD'], DSETS ):

    data = select( io.load(label, dset), _min_duration=1.5,
                    is_selected=True, is_tired=False )
    data = to_feature_array(data)

    times = data.reset_index().time.unique()

    res = []
    for crop in range(len(times - 1)//2):
        if crop >0:
            to_use_times = times[crop:-crop]
        else:
            to_use_times = times
        df = select(data.reset_index(),
                time_in_=to_use_times).set_index(['trial', 'time'])
        res.append(shuffle_val_predict(clf, df, n_splits = NSPLITS, get_weights=False) )

    pickle.dump(res, open('{}{}_{}'.format(folder, label, dset),'wb') )
