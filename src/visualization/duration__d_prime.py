"""
This script generates one set of plots:
    1. Comparison of Trial_size and Trial_index Dprimes
"""

import os
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes("dark")


from spikelearn.data import SHORTCUTS, io, select

# Folders of interest
loaddir = 'data/results/duration/d_prime'
savedir = 'reports/figures/duration/d_prime'
filename = lambda label: '{}/{}_Dprime.csv'.format(loaddir, label)

sess, dur = 'is_init D\'', 'is_short D\''
det, det_ses = 'detrended Duration D\'', 'detrended Trial Index D\''
for label in ['DRRD 8']:#SHORTCUTS['groups']['DRRD']:
    data = pd.read_csv( filename(label) )


    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot(2,2,2)

    # Relation between Dprimes
    data.plot.scatter(sess, dur, ax=ax, s=2)
    ax.plot(data[sess], data['regress'], '-', color='r', alpha=.7)

    # Duration-specific D'
    ax = plt.subplot(2,2,1)
    dur_data = data[data[det]>0]
    sns.barplot(data=dur_data, y=det, x='unit',  ax = ax,
                 order = dur_data.sort_values(det).unit,
                 palette = 'copper_r')

    # Trial-index specific D'
    ax = plt.subplot(2,2,4)
    ses_data = data[ data[det_ses]>0 ]
    sns.barplot(data=ses_data, y=det_ses, x='unit',  ax = ax,
                 order = ses_data.sort_values(det_ses).unit,
                 palette = 'plasma_r')

    # Activity
    name = 'duration'
    ax = plt.subplot(2, 2, 3)
    act = pd.read_csv('{}/{}_{}'.format(loaddir, label, name))
    act = act.drop('is_init', axis=1).melt(id_vars = ['is_short', 'trial'])
    act.variable = act.variable.astype(float)
    sns.tsplot(data = act, condition='is_short', time='variable',
                unit='trial', value='value',
                    ax=ax)

    #act = act.groupby('is_init').mean().drop('is_short',axis=1).melt()
    #sns.tsplot()

    # General Aesthetics
    plt.suptitle('Duration and train selectivity', y=1.01)

    plt.tight_layout()
    plt.show()


    #plt.savefig('{}/{}'.format(savedir, label))
