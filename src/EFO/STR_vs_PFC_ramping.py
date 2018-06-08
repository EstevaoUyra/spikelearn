import pandas as pd
from itertools import product

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
import os
import sys
sys.path.append('.')

from spikelearn.models import shuffle_val_predict
from spikelearn.data import io, SHORTCUTS, to_feature_array, select, remove_baseline
from spikelearn.measures import ramping_p
import numpy as np

# Parameters
tmin = 1.5;
tmax = 10;
MIN_QUALITY = 0
# DSETS = ['wide_smoothed', 'medium_smoothed', 'medium_smoothed_norm', 'huge_smoothed']
DSETS = ['wide_smoothed', 'huge_smoothed', 'medium_smoothed']
CLFs = [(LogisticRegression(), 'LogisticRegression')]
BLINE = [False]

basedir = 'data/results/double_recording/ramping'

n_splits = 50
SUBSETS = ['cropped', 'full']
RAMPING = [True, False]
LABELS = list(SHORTCUTS['groups']['EZ'].keys())+['all']

def select_ramping_neurons(label, tmin=1.5, min_quality = MIN_QUALITY,
                            P_VALUE_RAMP=.05, return_ramping=True):
    """
    Returns the indices of ramping neurons.

    If not return ramping, instead returns non-ramping neurons
    """
    data = io.load(label, 'no_smoothing')
    data = select(data, _min_duration=tmin, _mineq_quality = min_quality)
    fr = to_feature_array(data, subset='cropped')

    rp = lambda df: ramping_p(df.value, df.time)
    p_ramp = fr.reset_index().drop('trial', axis=1).melt(id_vars=['time']).groupby('unit').apply(rp)
    is_ramp = p_ramp < P_VALUE_RAMP
    return is_ramp[is_ramp==return_ramping].index.values



conditions = product(DSETS, CLFs, BLINE, SUBSETS, LABELS)
for dset, (clf, clfname), bline, subset, label in conditions:
    savedir = '{}/{}/{}/{}'.format(basedir, clfname, dset,
                                            subset)
    print(savedir, '\n', label)
    if not os.path.exists(savedir):
        os.makedirs(savedir)


    alldata = {}
    for ramping in [True, False]:

        if label=='all':
            print([lab for lab in SHORTCUTS['groups']['EZ']])
            dsets = [select(io.load(lab, dset).reset_index(),
                            _mineq_quality=MIN_QUALITY,
                            _in_unit= select_ramping_neurons(lab,
                                                    return_ramping=ramping),
                            _min_duration=tmin, _max_duration=tmax).set_index(['trial','unit']) for lab in SHORTCUTS['groups']['EZ']]
            print([ds.reset_index().trial.nunique() for ds in dsets])
            dsets = [ds for ds in dsets if ds.reset_index().trial.nunique()>0]
            n_trials = np.min([ds.reset_index().trial.nunique() for ds in dsets])
            n_bins = dsets[0][subset].apply(len).min()
            alldata[ramping]=[]
            print(n_trials, n_bins)
            for area in [None, 'PFC', 'STR']:
                if area is not None:
                    area_dsets = [select(ds, area=area) for ds in dsets]
                else:
                    area_dsets = dsets
                df = pd.concat([to_feature_array(ds, subset=subset).reset_index('trial', drop=True).iloc[:n_trials*n_bins] for ds in area_dsets],axis=1)
                df['trial'] = np.hstack([n_bins*[i] for i in range(n_trials)])
                df = df.reset_index().set_index(['trial','time'])
                alldata[ramping].append(df)

        else:
            selected_neurons = select_ramping_neurons(label, return_ramping=ramping)
            data = io.load(label, dset)
            data = select(data, _mineq_quality=MIN_QUALITY,
                            _min_duration=tmin, _max_duration=tmax)
            data = select( data.reset_index(),
                           _in_unit=selected_neurons ).set_index(['trial','unit'])
            dataPFC = select(data, area='PFC')
            dataSTR = select(data, area='STR')
            print(data.shape, dataPFC.shape, dataSTR.shape)

            data = to_feature_array(data, subset=subset)
            dataPFC = to_feature_array(dataPFC, subset=subset)
            dataSTR = to_feature_array(dataSTR, subset=subset)
            alldata[ramping] = (data, dataPFC, dataSTR)


    # Compare using the same number of neurons
    dfs = [alldata[True][1], alldata[True][2],
            alldata[False][1], alldata[False][2]]
    names = ['Ramping PFC', 'Ramping STR', 'No ramps PFC', 'No ramps STR']
    try:
        res = shuffle_val_predict(clf, dfs, names,
                                    n_splits = n_splits, feature_scaling='standard',
                                    balance_feature_number=True)
        res.save('{}/{}.pickle'.format(savedir, label))
    except: pass
