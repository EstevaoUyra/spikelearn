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
DSETS = ['medium_smoothed', 'medium_smoothed_norm',
        'narrow_smoothed', 'narrow_smoothed_norm']
CLFs = [(LogisticRegression(), 'LogisticRegression'),
            (GaussianNB(),'NaiveBayes') ]

basedir = 'data/results/double_recording'

ntrials_init_vs_after = 100
n_splits = 50
ntrials_total = 400


for label, dset, (clf, clfname) in product(SHORTCUTS['groups']['EZ'],
                                            DSETS, CLFs):
    savedir = '{}/{}/{}/{}'.format(basedir, clfname, dset, label)
    print(savedir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    data = io.load(label, dset)
    data = select(data, _min_duration=tmin, _max_duration=tmax)
    dataPFC = select(data, area='PFC')
    dataSTR = select(data, area='STR')

    baseline = io.load(label, 'baseline')
    data = remove_baseline(to_feature_array(data), baseline, .5)
    dataPFC = remove_baseline(to_feature_array(dataPFC), baseline, .5)
    dataSTR = remove_baseline(to_feature_array(dataSTR), baseline, .5)

    # In each compare first 50 with next 50.
    trials = data.reset_index().trial.unique()
    sep1 = trials[ntrials_init_vs_after]
    sep2 = trials[2*ntrials_init_vs_after+1]
    for df, area in zip([data, dataPFC, dataSTR], ['both', 'PFC', 'STR']):
        ini_df = select(df.reset_index(), _maxeq_trial=sep1)
        ini_df = ini_df.set_index(['trial','time'])
        end_df = select(df.reset_index(), _min_trial=sep1, _maxeq_trial=sep2)
        end_df = end_df.set_index(['trial','time'])
        res = shuffle_val_predict(clf, [ini_df, end_df], ['first50', 'next50'],
                                    n_splits=n_splits, feature_scaling='standard',
                                    cross_prediction=True, balance_feature_number=False)
        res.save('{}/{}_init_vs_after.pickle'.format(savedir,area))

    # Compare striatum with pfc
    res = shuffle_val_predict(clf, [dataPFC, dataSTR], ['PFC', 'STR'],
                                n_splits = n_splits, feature_scaling='standard')
    res.save('{}/performance.pickle'.format(savedir))
