from spikelearn import io, select, to_feature_array, SHORTCUTS
from spikelearn import shuffle_val_predict
from spikelearn.data.selection import frankenstein

import os

from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV()


day_1_labels = [l for l in SHORTCUTS['groups']['EZ'] if '_2' not in label]
dfs_day1 = [io.load(label, 'wide_smoothed') for label in day_1_labels]
pfc_day1 = frankenstein(dfs_day1, _min_duration=1.5, area = 'PFC')
str_day1 = frankenstein(dfs_day1, _min_duration=1.5, area = 'STR')

day_2_labels = [l for l in SHORTCUTS['groups']['EZ'] if '_2' in label]
dfs_day2 = [io.load(label, 'wide_smoothed') for label in day_2_labels]
pfc_day2 = frankenstein(dfs_day2, _min_duration=1.5, area = 'PFC')
str_day2 = frankenstein(dfs_day2, _min_duration=1.5, area = 'STR')


res = shuffle_val_predict(clf, [pfc_day1, str_day1, pfc_day2, str_day2],
                          ['pfc_day1', 'str_day1', 'pfc_day2', 'str_day2'], n_splits = 50,
                          feature_scaling='standard',
                         train_size=.8, test_size=.2, cross_prediction=False,
                         balance_feature_number = True)


if not os.path.exists('data/results/central_figures'):
    os.makedirs('data/results/central_figures')

res.save('data/results/central_figures/STR_PFC_day1_vs_day2.pickle')
