import pandas as pd
import sys
sys.path.append('.')

from spikelearn.measures.similarity import unit_similarity_evolution
from spikelearn.data import io, to_feature_array, select, SHORTCUTS

from sklearn import LogisticRegression


# Parameters
DSET = 'medium_smoothed'
TMIN = 1.5
WSIZE = 50
folder = 'data/results/across_trials/similarity/'

# Run
for label in SHORTCUTS['groups']['DRRD']:
    data = select(io.load(label, DSET), _min_duration=TMIN, is_selected=True)
    data = to_feature_array(data, False)
    rat_sim = pd.DataFrame()
    for unit in data.columns:
        sim_mat = unit_similarity_evolution(data[unit], WSIZE)
        sim_mat['unit'] = unit
        rat_sim = rat_sim.append(sim_mat)

    filename = '{}_w{}_t{}_unit_sim_evolution.csv'.format(label, WSIZE, TMIN)
    rat_sim.reset_index().set_index(['unit','trial']).to_csv(folder + filename)
