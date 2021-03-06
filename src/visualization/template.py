"""
This script generates visualizations

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

    # Plot
    plt.title('')
    figname = '___.png'
    plt.savefig('{}/{}'.format(savedir, figname), dpi=200)
    plt.close(fig)
