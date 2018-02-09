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
