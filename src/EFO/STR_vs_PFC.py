import pandas as pd
from itertools import product

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from spikelearn.models import shuffle_val_predict
from spikelearn.data import io, SHORTCUTS, to_feature_array, select


# For decoding
tmin = 1.5;
tmax = 10;
DSETS = ['medium_smoothed', 'medium_smoothed_norm']
CLFs = [LogisticRegression, GaussianNB]

ntrials_init_vs_after = 50




for label, dset, clf in product(SHORTCUTS['groups']['EZ'], DSETS, CLFs):
    data = io.load(label, dset)
    data = select(data, _min_duration=tmin, _max_duration=tmax)

    dataPFC = select(data, area='PFC')
    dataSTR = select(data, area='STR')

    trials = data.trial.unique()

    sep = trials[ntrials_init_vs_after]

    for df in [data, dataPFC, dataSTR]:
        
