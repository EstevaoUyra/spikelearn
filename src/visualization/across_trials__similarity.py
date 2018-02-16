"""
This script generates

"""
import os
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import sys
sys.path.append('.')
from spikelearn.data import SHORTCUTS

from itertools import product

# Data parameters
DSETS = ['medium_smoothed', 'medium_smoothed_norm',
         'narrow_smoothed', 'narrow_smoothed_norm',
         'wide_smoothed']
TMIN = 1.5
WSIZE = 50
loaddir = 'data/results/across_trials/similarity/'
sim_filename = lambda label, dset: '{}/{}_w{}_t{}_unit_sim_evolution.csv'.format(loaddir+dset, label, WSIZE, TMIN)
pred_filename = lambda label, dset: '{}/{}_w{}_t{}_pred_init_end.csv'.format(loaddir+dset, label, WSIZE, TMIN)
weight_filename = lambda label, dset: '{}/{}_w{}_t{}_weight_init_end.csv'.format(loaddir+dset, label, WSIZE, TMIN)

# Saving parameters
fsavedir = lambda label, dset : 'reports/figures/across_trials/similarity/{}/{}'.format(dset, label)

# Identifier variables
id_vars = ['dset', 'logC', 'penalty']
id_combs = lambda df: product(weights.unique(df[id_]) for id_ in id_vars)

# Create visualizations
for label, dset in product(SHORTCUTS['groups']['DRRD'], DSETS):
    print(label, dset)
    savedir = fsavedir(label, dset)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    similarities = pd.read_csv(sim_filename(label, dset)).set_index(['unit','trial']).drop('dset',axis=1)
    units = similarities.reset_index().unit.unique()

    # One image for each neuron
    for unit in units:
        fig = plt.figure()
        sns.heatmap(similarities.loc[unit].values)
        plt.title('Unit {}, {}'.format(unit, label) )
        plt.savefig('{}/sim_evo_unit_{}.png'.format(savedir, unit),
                        dpi=500)
        plt.close(fig)

    # Plus one image for the mean
    fig = plt.figure()
    sns.heatmap(similarities.reset_index().groupby('trial').mean().drop('unit', axis=1))
    plt.title('Similarity with mean across units')
    plt.savefig('{}/sim_evo_mean.png'.format(savedir),
                    dpi=800)
    plt.close(fig)

    # Classifier-dependent analysis
    preds = pd.read_csv(pred_filename(label, dset))
    weights = pd.read_csv(weight_filename(label, dset))
    # Weighted mean & prediction evolution
    ids = product(*(weights[var].unique() for var in id_vars if var !='dset'))

    for id_ in ids:
        print(id_)
        id_u = (dset, *id_)
        # Calculate weighted mean similarity
        weight = weights.groupby(id_vars).get_group(id_u).set_index('unit')
        ponder = lambda df: df*weight.loc[df.unit.values[0], 'w']
        wm = similarities.reset_index().groupby('unit').apply(ponder)
        # Get prediction trialseries
        pred = preds.groupby(id_vars).get_group(id_u).sort_values('trial')
        # Create plot
        fig = plt.figure()
        ## Similarity
        ax = plt.subplot2grid((4,4),(1,0),rowspan=3,colspan=3)
        sns.heatmap((wm.groupby('trial').sum().drop('unit',axis=1)).values,
                                    ax=ax,cbar=False)
        ## Prediction evolution
        ax = plt.subplot2grid((4,4), (0,0), colspan=3)
        ax.plot(pred.predicted.values, linewidth=.5, alpha=.7)
        pred.predicted.rolling(20,center=True).mean().plot(linewidth=2)
        ## Trial in question
        ax = plt.subplot2grid((4,4), (1,3), rowspan=3)
        ax.plot(np.arange(len(pred.trial)), pred.trial)
        plt.xlabel("Trial in question")
        # Finalize plot and save
        plt.suptitle('Similarity weighted by Logistic Regression coefs')
        plt.savefig('{}/sim_evo_mean_weighted_{}.png'.format(savedir,'_'.join(map(str,id_))),
                        dpi=1000)
        plt.tight_layout()
        plt.close(fig)
