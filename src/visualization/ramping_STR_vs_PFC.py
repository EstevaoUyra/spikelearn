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

SUBSETS = ['cropped', 'full']
RATS = list(SHORTCUTS['groups']['EZ'].keys())
names = ['Ramping PFC', 'Ramping STR', 'No ramps PFC', 'No ramps STR']

fsavedir = lambda dset: 'reports/figures/double_recording/ramping/{}'.format(dset)


ntrials_init_vs_after = 100
n_splits = 50
ntrials_total = 400

for dset, (clf, clfname), subset in product(DSETS, CLFs, SUBSETS):
    # # Compare evolution @ first and second day
    # all_scores = pd.DataFrame()
    # mean_probas = pd.DataFrame()
    #
    # for (i, label), name in product(RATS, enumerate(names)):
    #
    #     loaddir = '{}/{}/{}/{}'.format(basedir, clfname, dset, subset)
    #     res = pickle.load(open('{}/{}.pickle'.format(savedir, label), 'rb'))
    #
    #     res.proba_matrix(grouping=('tested_on', 'Ramping STR'))
    #
    #     score = res.score.groupby('trained_here').get_group(True)
    #     score.loc[:,'label'] = label
    #     score.loc[:,'day'] = 2 if '_2' in label else 1
    #     score.loc[:,'area'] = area
    #     all_scores = all_scores.append(score)
    #
    # all_probas = all_probas.reset_index().groupby(['area','day', 'true_label']).mean()
    #
    # savedir = fsavedir(dset, clfname)
    # if not os.path.exists(savedir):
    #     os.makedirs(savedir)
    #
    # ## Proba plots
    # fig=plt.figure(figsize=(14,20))
    # for day, (j, area) in product(DAYS, enumerate(AREAS)):
    #     ax=plt.subplot(len(AREAS),len(DAYS) , 2*j + day)
    #     sns.heatmap(all_probas.loc[area, day])
    #     plt.xlabel('Possible labels');
    #     plt.title('%s Electrodes, Day %d, probability matrix'%(area, day))
    # plt.savefig('{}/probability_matrix_{}.png'.format(savedir, subset), dpi=200)
    # plt.tight_layout()
    # plt.close(fig)
    label='all'
    if not os.path.exists(fsavedir(dset)):
        os.makedirs(fsavedir(dset))

    loaddir = '{}/{}/{}/{}'.format(basedir, clfname, dset, subset)
    res = pickle.load(open('{}/{}.pickle'.format(loaddir, label), 'rb'))
    # Plot Frankenstein rat
    fig=plt.figure(figsize=(14,14))
    for (i, ramping), (j, area) in product(enumerate(['Ramping ', 'No ramps ']), enumerate(['PFC', 'STR'])):
        which = ramping+area
        print(which,2*i+j+1)
        plt.subplot(2,2,2*i+j+1)
        if subset=='full':
            vmin,vmax = .063, .045
            res.proba_matrix(grouping=('tested_on', which), vmax=vmin,vmin=vmax, cbar=False)
            plt.xlim(5,20); plt.ylim(20,5)
        else:
            vmin,vmax = .12, .09
            res.proba_matrix(grouping=('tested_on', which), vmax=vmin,vmin=vmax, cbar=False)
        plt.title(which)
    ax=plt.axes((.95, .2, .05, .6), facecolor='w')
    sns.heatmap(np.linspace(vmin,vmax,100).reshape(-1,1),cbar=False, ax=ax)
    ax.yaxis.tick_right()
    ax.tick_params(rotation=0)
    ax.set_yticks(np.linspace(0,100,101)[::20]); ax.set_xticklabels(['']);
    ax.set_yticklabels(np.linspace(vmin,vmax,101)[::20].round(3));
    plt.savefig('{}/frankenstein_proba_{}.png'.format(fsavedir(dset), subset),bbox_inches='tight')
    plt.close(fig)

    # Bar scores comparison
    ramp_score = res.score[['tested_on','pearson_mean','cv']]
    ramp_score['area'] = ramp_score['tested_on'].apply(lambda x: x[-3:])
    ramp_score['cell'] = ramp_score['tested_on'].apply(lambda x: x[:-3])

    fig=plt.figure(figsize=(8,6))
    sns.barplot(x='area',hue='cell',y='pearson_mean', data=ramp_score)
    plt.title('Score with %d neurons in each group'%res.score.n_features.values[0], fontsize=16)
    plt.ylabel('Pearson r predicted vs true label')
    plt.savefig('{}/frankenstein_score_{}.png'.format(fsavedir(dset), subset))
    print('{}/frankenstein_score_{}.png'.format(fsavedir(dset), subset))
    plt.close(fig)


    # Striatum vs PFC
