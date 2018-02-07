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
loaddir = 'data/results/'

# Saving parameters
savedir = 'reports/figures/'

# Create visualizations
for label in SHORTCUTS['groups']['DRRD']:
    pass
