import pickle
from itertools import product
from spikelearn.data import io, SHORTCUTS, to_feature_array, select
from spikelearn.models.mahalanobis import MahalanobisClassifier

DSETS = ['wide_smoothed', 'wide_smoothed_norm']
allsims = {}
for rat, dset in product(SHORTCUTS['group']['day1'], DSETS)
    data = select(io.load(rat, dset), _min_duration=1.5, is_tired=False)
    X, y, trial = to_feature_array(data, Xyt=True, subset='full')

    for tr_ in np.unique(y)[2:]:
        mah = MahalanobisClassifier(shared_cov=True)
        mah.fit(X[trial < tr_], y[trial < tr_])
        allsims[(rat, dset, tr_)] = mah.transform(X[trial == tr_], y[trial == tr_)

    pickle.dump(distances, open('data/results/consistency_of_activity.pickle', 'wb'))