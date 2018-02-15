import pandas as pd
import numpy as np

import sys
sys.path.append('.')

from spikelearn.data import SHORTCUTS

from numpy import dot
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product, count
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

# Load results data
data = pd.read_csv('data/results/across_trials/feature_importance.csv')
actv = pd.read_csv('data/results/across_trials/feature_importance_activity.csv')
score = pd.read_csv('data/results/across_trials/feature_importance_predictions.csv')

# WIll use a single train_size to compare scores
score = score.groupby('num_trials').get_group(50)

# Definitions used at generation
id_vars = ['rat','dataset','num_trials', 'unit', 'time','logC']

DRRD_RATS = SHORTCUTS['groups']['DRRD']
LOGCS = sorted(data.groupby('num_trials').get_group(50).logC.unique())[::-1]

# Preprocessing to get changes
each_mean = data.groupby(id_vars + ['when']).mean().reset_index(-1)
single_mean = data.groupby(id_vars).mean()
single_std = data.groupby(id_vars).std()

z_scored = (each_mean-single_mean)/single_std
z_scored['when'] = each_mean.when.values
data = z_scored.reset_index()

# Calculating score from predictions
score = score.groupby('num_trials').get_group(50)
score = pd.DataFrame(score.groupby(['rat', 'set', 'logC','cv','when']).apply(lambda x: x[['predictions','true']].corr().iloc[0,1]), columns=['score']).reset_index()


for rat, (fig_n, logC) in product(DRRD_RATS, enumerate(LOGCS)):
    one_rat = actv.groupby(['rat', 'logC', 'num_trials']).get_group((rat, logC, 50))
    fig = plt.figure(figsize=(12,24))
    for i, unit in enumerate(one_rat.unit.unique()):
        plt.subplot(one_rat.unit.nunique(), 3, 3*(i+1)-1)
        for when, c in [('init', 'b'),('end', 'r')]:
            local = one_rat.set_index(['unit','time']).groupby('when').get_group(when)
            local.loc[unit]['mean'].plot(color=c)
            plt.fill_between(local.loc[unit]['std'].index.values,
                             local.loc[unit]['mean'] + local.loc[0]['std'],
                             local.loc[unit]['mean'] - local.loc[0]['std'],alpha=.4,color=c)
            plt.axis('off');

    ax = plt.subplot(1,3,1)
    d = data.groupby(['rat', 'logC', 'num_trials','when']).get_group((rat, logC, 50, 'end'))
    d = d.pivot(index='unit', columns='time', values='value')
    sns.heatmap(d.values, ax=ax, vmin = -2, vmax = 2, cbar = False, cmap='RdBu_r')

    one_score = score.groupby(['rat','logC']).get_group((rat, logC)).groupby(['set','when']).mean()
    ax = plt.subplot(1,3,3)
    sns.barplot(x='set',y='score',hue='when',data=one_score.reset_index(), ax=ax,palette=['b','r'], hue_order=['init','end']); plt.ylim([0,1])
    plt.title('logC:%.3f'%logC, fontsize=30)

    plt.savefig('reports/figures/across_trials/feature_importance/feature_importance_{}_n{:3}_logC{:.3}.png'.format(rat, fig_n, logC), dpi=100)
    plt.close(fig)
