from itertools import product

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
#from xgboost import XGBClassifier
import os
import sys
sys.path.append('.')
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from spikelearn.data import SHORTCUTS

# Parameters
tmin = 1.5;
tmax = 10;
DSETS = ['medium_smoothed', 'medium_smoothed_norm',
        'narrow_smoothed', 'narrow_smoothed_norm', 'wide_smoothed', 'huge_smoothed']
DSETS = ['huge_smoothed']
CLFs = [(LogisticRegression(), 'LogisticRegression'),
            (GaussianNB(),'NaiveBayes') ]
BLINE = [True, False]
import pandas as pd
AREAS = ['both', 'PFC', 'STR']
DAYS = [1,2]
basedir = 'data/results/double_recording'

RATS = SHORTCUTS['groups']['EZ']
#RATS = [rat for rat in RATS if '6' not in rat]

#fsavedir = lambda dset, clf, bline: 'reports/figures/double_recording/{}/{}/{}'.format(dset, clf, bline)
fsavedir = lambda dset, clf, bline: 'reports/figures/double_recording/{}/{}/{}'.format(dset, clf, str(bline))


ntrials_init_vs_after = 100
n_splits = 50
ntrials_total = 400

for dset, (clf, clfname), bline in product(DSETS, CLFs, BLINE):
    # Compare evolution @ first and second day
    all_scores = pd.DataFrame()
    all_probas = pd.DataFrame()
    for label, area in product(RATS, AREAS):

        loaddir = '{}/{}/{}/{}/{}'.format(basedir, clfname, dset, label, bline)

        res = pickle.load(open('%s/%s_init_vs_after.pickle'%(loaddir, area), 'rb'))

        proba = res.proba_matrix(plot=False)
        proba.loc[:,'day'] = 2 if '_2' in label else 1
        proba.loc[:,'area'] = area
        all_probas = pd.concat([all_probas, proba])

        score = res.score.groupby('trained_here').get_group(True)
        score.loc[:,'label'] = label
        score.loc[:,'day'] = 2 if '_2' in label else 1
        score.loc[:,'area'] = area
        all_scores = all_scores.append(score)

    all_probas = all_probas.reset_index().groupby(['area','day', 'true_label']).mean()

    savedir = fsavedir(dset, clfname, bline)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    ## Proba plots
    fig=plt.figure(figsize=(14,20))
    for day, (j, area) in product(DAYS, enumerate(AREAS)):
        ax=plt.subplot(len(AREAS),len(DAYS) , 2*j + day)
        sns.heatmap(all_probas.loc[area, day])
        plt.xlabel('Possible labels');
        plt.title('%s Electrodes, Day %d, probability matrix'%(area, day))
    plt.savefig('{}/probability_matrix.png'.format(savedir), dpi=200)
    plt.tight_layout()
    plt.close(fig)

    ## Evolution plots
    fig=plt.figure(figsize=(14,20))
    for day, (j, area) in product(DAYS, enumerate(AREAS)):
        local_data = all_scores.groupby(['day','area']).get_group((day, area))
        ax=plt.subplot(len(AREAS),len(DAYS) , 2*j + day)
        sns.pointplot(x='tested_on', y='score_mean',hue='label',
                        linestyles='--', data=local_data, legend=False)
        plt.legend().remove()
        sns.pointplot(x='tested_on', y='score_mean', color='k', n_boot=5000,
                        data=local_data, legend=False)
        plt.ylim(-.3,.65); plt.xlim(-.1,1.1); #plt.gca().legend_.remove()
        plt.title('%s Electrodes, Day %d, Change in decoding performance'%(area, day))

    plt.savefig('{}/evolution_both.png'.format(savedir), dpi=200)
    plt.tight_layout()
    plt.close(fig)

    # Striatum vs PFC
