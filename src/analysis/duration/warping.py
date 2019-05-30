from spikelearn.data import io, select, to_feature_array, SHORTCUTS
from spikelearn.models.shuffle_decoding import  shuffle_cross_predict
from catboost import CatBoostClassifier
from sklearn.linear_model import BayesianRidgeRegression
import pickle

allres = {}
for rat, dset in product(SHORTCUTS['group']['eletro'], DSETS):
    data = select(io.load(rat, dset), _min_duration=.5, is_tired=False)
    tercils = [data.duration.quantile(q) for q in [1/3, 2/3]]

    t1 = to_feature_array(select(data, _max_duration=tercils[0]), subset='full')
    t3 = to_feature_array(select(data, _min_duration=tercils[1]), subset='full')
    res = shuffle_cross_predict(reg, [t1,t3], ['short', 'long'], n_splits=5,
                                problem='regression', feature_scaling='robust')

    allres[(rat, dset)] = res
    
pickle.dump(open('data/results/warping.pickle', 'wb'))
    # TODO calculate bias and mean bias direction