"""
This script generates plots for D' distribution along neurons,
for each side of the changepoint.

"""
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import os
import sys
sys.path.append('.')
from spikelearn.data import SHORTCUTS

# Data parameters
loaddir = 'data/results/duration/d_prime'
fresults = lambda label: '{}/{}_Dprime_cp.csv'.format(loaddir, label)

# Saving parameters
fsavedir = lambda label: 'reports/figures/changepoint/dprime'

# Create visualizations
for label in SHORTCUTS['groups']['DRRD']:
    print(label)
    savedir = fsavedir(label)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    data = pd.read_csv( fresults(label), index_col=0 ).reset_index()
    data = data.melt(id_vars='unit', value_name='dprime')
    maxD = data.groupby('unit').max().reset_index()

    fig = plt.figure()

    ax = plt.subplot(2,1,1)
    sns.barplot(x='unit', y='dprime', data=data, ax=ax)
    plt.title('Mean D\'')

    ax = plt.subplot(2,1,2)
    sns.barplot(x='unit', y='dprime', data=maxD, ax=ax)
    plt.title('Max D\'')

    plt.suptitle('Baseline Dprime around changepoint, {}'.format(label))
    figname = 'max_baseline_dprime.png'
    plt.savefig('{}/{}_{}'.format(savedir, label, figname), dpi=200)
    plt.close(fig)
