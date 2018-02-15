"""
This script generates

"""
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import sys
sys.path.append('.')
from spikelearn.data import SHORTCUTS

# Data parameters
TMIN = 1.5
WSIZE = 50
loaddir = 'data/results/across_trials/similarity/'
filename = lambda label: '{}{}_w{}_t{}_unit_sim_evolution.csv'.format(loaddir, label, WSIZE, TMIN)

# Saving parameters
savedir = 'reports/figures/across_trials/similarity/'

# Identifier variables
id_vars = ['logC', 'dset'. 'penalty']
id_combs = lambda df: product(weights.unique(df[id_]) for id_ in id_vars)

# Create visualizations
for label in SHORTCUTS['groups']['DRRD']:
    similarities = pd.read_csv(filename(label)).set_index(['unit','trial'])
    units = similarities.reset_index().unit.unique()

    # One image for each neuron
    for unit in units:
        fig = plt.figure()
        sns.heatmap(similarities.loc[unit])
        plt.title('Unit {}, {}'.format(unit, label) )
        plt.savefig('{}sim_evo_{}_unit_{}'.format(savedir, label, unit),
                        dpi=200)
        plt.close(fig)

    # Plus one image for the mean
    fig = plt.figure()
    sns.heatmap(similarities.reset_index().groupby('trial').mean())
    plt.title('Similarity with mean across units')
    plt.savefig('{}sim_evo_{}_mean'.format(savedir, label),
                    dpi=200)

    # Weighted mean
    weight = weights.groupby(id_vars).get_group(id_u)
    ponder = lambda df: df*weight.loc[df.unit.values[0], 'w']
    wm = similarities.groupby('unit').apply(ponder)
    fig = plt.figure()
    sns.heatmap(wm.groupby('trial').sum()/weight.sum())
    plt.title('Similarity with importances given by Logistic Regression')
    plt.savefig('{}sim_evo_{}_mean_weighted'.format(savedir, label),
                    dpi=200)
