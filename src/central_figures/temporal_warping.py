import sys
sys.path.append('.')
import os
from spikelearn import frankenstein, shuffle_val_predict
from spikelearn.models import shuffle_cross_predict
from spikelearn.data import io, SHORTCUTS
from sklearn.linear_model import BayesianRidge



gb = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['DRRD']]
d1 = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['EZ'] if '_2' not in label]
d2 = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['EZ'] if '_2' in label]

gb_short = frankenstein(gb, _min_duration=1.5, _max_duration=3, is_selected=True, is_tired=False)
d1_short = frankenstein(d1, _min_duration=1.5, _max_duration=3, _min_quality=0, area='PFC')
d2_short = frankenstein(d2, _min_duration=1.5, _max_duration=3, _min_quality=0, area='STR')

gb_long = frankenstein(gb, _min_duration=3, _max_duration=6, is_selected=True, is_tired=False)
d1_long = frankenstein(d1, _min_duration=3, _max_duration=6, _min_quality=0, area='PFC')
d2_long = frankenstein(d2, _min_duration=3, _max_duration=6, _min_quality=0, area='STR')


res = {}
for name, (df_short, df_long) in zip(['DRRD', 'd1','d2'], zip([gb_short, d1_short, d2_short],
                                                              [gb_long, d1_long, d2_long])):
    clf = BayesianRidge()
    res[name] = shuffle_cross_predict(clf, [df_short, df_long], ['short','long'], feature_scaling='robust', problem='regression')