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


# Parameters
tmin = 1.5;
tmax = 10;
MINQUALITY = 0
# DSETS = ['wide_smoothed', 'medium_smoothed', 'medium_smoothed_norm', 'huge_smoothed']
DSETS = ['wide_smoothed', 'huge_smoothed', 'medium_smoothed']
CLFs = [(LogisticRegression(), 'LogisticRegression'),
            (GaussianNB(),'NaiveBayes') ]
BLINE = [True, False]

basedir = 'data/results/double_recording'

ntrials_init_vs_after = 70
n_splits = 50
ntrials_total = 400
SUBSETS = ['cropped', 'full']

conditions = product(DSETS, CLFs, BLINE, SHORTCUTS['groups']['EZ'], SUBSETS)

for dset, (clf, clfname), bline, label, subset in conditions:
    savedir = '{}/{}/{}/{}/{}/{}'.format(basedir, clfname, dset,
                                            subset, label, bline)
    print(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    data = io.load(label, dset)
    data = select(data, _mineq_quality=MINQUALITY,
                    _min_duration=tmin, _max_duration=tmax)
    dataPFC = select(data, area='PFC')
    dataSTR = select(data, area='STR')
    print(data.shape, dataPFC.shape, dataSTR.shape)
    baseline = io.load(label, 'baseline')
    if bline:
        data = remove_baseline(to_feature_array(data, subset=subset), baseline, .5)
        dataPFC = remove_baseline(to_feature_array(dataPFC, subset=subset), baseline, .5)
        dataSTR = remove_baseline(to_feature_array(dataSTR, subset=subset), baseline, .5)
    else:
        data = to_feature_array(data, subset=subset)
        dataPFC = to_feature_array(dataPFC, subset=subset)
        dataSTR = to_feature_array(dataSTR, subset=subset)

    # In each compare first 50 with next 50.
    trials = data.reset_index().trial.unique()
    sep1 = trials[ntrials_init_vs_after]
    sep2 = trials[2*ntrials_init_vs_after+1]
    for df, area in zip([data, dataPFC, dataSTR], ['both', 'PFC', 'STR']):
        try:
            ini_df = select(df.reset_index(), _maxeq_trial=sep1)
            ini_df = ini_df.set_index(['trial','time'])
            end_df = select(df.reset_index(), _min_trial=sep1, _maxeq_trial=sep2)
            end_df = end_df.set_index(['trial','time'])
            res = shuffle_val_predict(clf, [ini_df, end_df],
                                        ['first%s'%ntrials_init_vs_after,
                                         'next%s'%ntrials_init_vs_after],
                                        n_splits=n_splits, feature_scaling='standard',
                                        cross_prediction=True, balance_feature_number=False)
            res.save('{}/{}_init_vs_after.pickle'.format(savedir,area))
        except:
            print('could not calculate {} for {}'.format(area, label))

    # Compare striatum with pfc
    try:
        res = shuffle_val_predict(clf, [dataPFC, dataSTR], ['PFC', 'STR'],
                                    n_splits = n_splits, feature_scaling='standard',
                                    balance_feature_number=True)
        res.save('{}/performance.pickle'.format(savedir))
    except: pass
