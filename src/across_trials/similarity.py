from itertools import product, cycle, chain
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from spikelearn.measures.similarity import unit_similarity_evolution
from spikelearn.models.shuffle_decoding import shuffle_val_predict
from spikelearn.data import io, to_feature_array, select, SHORTCUTS

from sklearn.linear_model import LogisticRegression

import os
# Parameters
DSETS = ['medium_smoothed', 'medium_smoothed_norm',
         'narrow_smoothed', 'narrow_smoothed_norm',
         'wide_smoothed']
TMIN = 1.5
WSIZE = 50; PRED_WSIZE = 50
folder = 'data/results/across_trials/similarity/'

# Prepare output folders
#[os.makedirs(folder+dset) for dset in DSETS]

# Run
for label, dset in product(SHORTCUTS['groups']['DRRD'], DSETS):
    subset = 'full' if 'norm' in dset else 'cropped'
    viz = lambda dset: dset+'_viz' if 'norm' not in dset or 'narrow' in dset else dset
    data_ = select( io.load(label, viz(dset)),
                   _min_duration=TMIN, is_selected=True )
    data = to_feature_array(data_, False, subset)

    # Pearson similarity
    res_sim = pd.DataFrame()
    for unit in data.columns:
        sim_mat = unit_similarity_evolution(data[unit], WSIZE)
        sim_mat['unit'] = unit
        sim_mat['dset'] = dset
        res_sim = res_sim.append(sim_mat)


    # ML prediction comparison
    res_pred = pd.DataFrame()
    data_ = select(io.load(label, dset), _min_duration=TMIN, is_selected=True)
    data = to_feature_array( select( data_, is_tired = False),
                            False, subset).unstack(-1).reset_index()

    trials = data.trial.unique()
    data['init'] = ((data['trial'] < trials[PRED_WSIZE]).astype(int) -
                        (data['trial'] > trials[-PRED_WSIZE]).astype(int))
    train = data[ data.init != 0 ]
    X, y = train.drop(['trial', 'init'], axis=1), ((train.init + 1)/2)

    ## Fit and predict
    C1, C2 = np.linspace(-1.5, 5, 20), np.linspace(-5, 5, 20)
    for logC, regl in chain( zip(C1, cycle(['l1'])), zip(C2, cycle(['l2']))):
        clf = LogisticRegression( C=10**logC )
        preds = clf.fit(X, y).predict_proba(data.drop(['trial', 'init'], axis=1))
        one_pred = pd.DataFrame( {'predicted':preds[:,1], 'true':data.init},
                                 index = data.trial )
        one_pred['logC'] = logC; one_pred['penalty'] = regl
        one_pred['dset'] = dset;
        res_pred = res_pred.append(one_pred)


    filename = '{}_w{}_t{}_unit_sim_evolution.csv'.format(label, WSIZE, TMIN)
    res_sim.to_csv('{}{}/{}'.format(folder, dset, filename))
    filename = '{}_w{}_t{}_pred_init_end.csv'.format(label, WSIZE, TMIN)
    res_pred.to_csv('{}{}/{}'.format(folder, dset, filename))
