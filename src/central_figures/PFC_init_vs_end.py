from spikelearn import io, select, to_feature_array, SHORTCUTS
from spikelearn import shuffle_val_predict
from spikelearn.data.selection import frankenstein

import os

from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV()

labels = SHORTCUTS['groups']['DRRD']

dfs = [io.load(label, 'wide_smoothed') for label in labels]

merged = frankenstein(dfs, _min_duration=1.5, is_tired=False)
size = merged.shape[0]

init = merged.iloc[ :size//2]
end = merged.iloc[size//2+1: ]

res = shuffle_val_predict(clf, [init, end],
                          ['init', 'end'], n_splits = 50,
                          feature_scaling='standard',
                         train_size=.8, test_size=.2, cross_prediction=False,
                         balance_feature_number = True)

if not os.path.exists('data/results/central_figures'):
    os.makedirs('data/results/central_figures')

res.save('data/results/central_figures/PFC_init_vs_end.pickle')
