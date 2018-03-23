"""
This script generates plots for D' distribution along neurons,
for each side of the changepoint.

"""
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import sys
sys.path.append('.')
from spikelearn.data import SHORTCUTS

# Data parameters
loaddir = 'data/results'
fresults = lambda label: '{}/{}'.format(loaddir, label)

# Saving parameters
fsavedir = lambda label: 'reports/figures/'

# Create visualizations
for label in SHORTCUTS['groups']['DRRD']:
    print(label)
    savedir = fsavedir(label)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    data = pd.read_csv( fresults(label) )
    maxD = data.groupby('unit').max().reset_index()

    ax = plt.subplot(2,1,1)
    sns.barplot(x='unit', y='dprime', data=data, ax=ax)
    plt.title('Mean D\'')

    ax = plt.subplot(2,1,2)
    sns.barplot(x='unit', y='dprime', data=maxD, ax=ax)
    plt.title('Max D\'')

    plt.suptitle('Baseline Dprime around changepoint, {}'.format(label))
    figname = 'max_baseline_dprime.png'
    plt.savefig('{}/{}'.format(savedir, figname), dpi=200)
    plt.close(fig)
