from spikelearn import io, select, to_feature_array, SHORTCUTS
from spikelearn import shuffle_val_predict
from spikelearn.data.selection import frankenstein

import os

from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV()


labels = SHORTCUTS['groups']['day1']

dfs = [io.load(label, 'wide_smoothed') for label in labels]

short = frankenstein(dfs, _min_duration=1.5, _max_duration=3)
longer = frankenstein(dfs, _min_duration=4 )


res = shuffle_val_predict(clf, [short, longer],
                          ['short', 'long'], n_splits = 50,
                          feature_scaling='standard',
                         train_size=.8, test_size=.2, cross_prediction=True,
                         balance_feature_number = True)
                         

if not os.path.exists('data/results/central_figures'):
    os.makedirs('data/results/central_figures')

res.save('data/results/central_figures/STR_PFC_day1_vs_day2.pickle')
