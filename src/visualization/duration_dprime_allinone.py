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

fig, ax = plt.subplots(2,4, figsize=(17,7))
for i, label in enumerate(SHORTCUTS['groups']['DRRD']):
    data = pd.read_csv( filename(label) )

#     # Relation between Dprimes
#     data.plot.scatter(sess, dur, ax=ax, s=2)
#     ax.plot(data[sess], data['regress'], '-', color='r', alpha=.7)

    # Duration-specific D'
    dur_data = data[data[det]>0]
    sns.barplot(data=dur_data, y=det, x='unit',  ax = ax[0,i],
                 order = dur_data.sort_values(det).unit,
                 color='k')
    ax[0,i].set_ylim(0,.15)

#     # Trial-index specific D'
#     ax = plt.subplot(2,2,4)
#     ses_data = data[ data[det_ses]>0 ]
#     sns.barplot(data=ses_data, y=det_ses, x='unit',  ax = ax,
#                  order = ses_data.sort_values(det_ses).unit,
#                  palette = 'plasma_r')

    # Activity
    name = 'duration'
    act = pd.read_csv('{}/{}_{}'.format(loaddir, label, name))
    act = act.drop('is_init', axis=1).melt(id_vars = ['is_short', 'trial'])
    act['init'] = act.trial < act.trial.quantile(.5)
    act.variable = act.variable.astype(float)
#     act.value = act.value/act.value.max
    sns.lineplot(data = act, hue='is_short', x='variable',# style='init',
                 y='value', ax=ax[1,i])
    L=ax[1,i].legend(frameon=False)
    L.get_texts()[0].set_text('')
    L.get_texts()[1].set_text('Short Trials')
    L.get_texts()[2].set_text('Long Trials')
    
    
    if i !=0:
        ax[1,i].set_ylabel('')
        ax[0,i].set_ylabel('')
    ax[0,i].xaxis.set_visible(False)
    ax[1,0].set_ylabel('Firing rate (spikes/s)', fontsize=14)
    ax[1,i].set_xlabel('Time from onset (ms)', fontsize=14)
    ax[0,0].set_ylabel('max Cohen\'s D', fontsize=14)
    #act = act.groupby('is_init').mean().drop('is_short',axis=1).melt()
    #sns.tsplot()

    # General Aesthetics
#     plt.suptitle('Duration and train selectivity', y=1.01)

    plt.tight_layout()
#     plt.show()


plt.savefig('reports/figures/duration/d_prime/full.png', dpi=300)
