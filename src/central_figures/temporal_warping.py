import sys
sys.path.append('.')

import os

from spikelearn import frankenstein, shuffle_val_predict, io, SHORTCUTS

gb = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['DRRD']]
d1 = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['day1']]
d2 = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['day2']]


gb_short = frankenstein(gb, _min_duration=1.5, _max_duration=3, is_selected=True, is_tired=False)
d1_short = frankenstein(d1, _min_duration=1.5, _max_duration=3, _min_quality=0, area='PFC')
d2_short = frankenstein(d2, _min_duration=1.5, _max_duration=3, _min_quality=0, area='STR')

gb_long = frankenstein(gb, _min_duration=3, _max_duration=6, is_selected=True, is_tired=False)
d1_long = frankenstein(d1, _min_duration=3, _max_duration=6, _min_quality=0, area='PFC')
d2_long = frankenstein(d2, _min_duration=3, _max_duration=6, _min_quality=0, area='STR')

for df_short, df_long in zip([gb_short, d1_short, d2_short],
                             [gb_long, d1_long, d2_long]):
    res = shuffle_val_predict(clf, [df_short, df_long], ['short','long'])