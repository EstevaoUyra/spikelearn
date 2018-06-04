from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS
from scipy.spatial.distance import mahalanobis
import numpy as np

# TODO document classifiers

def similarity(X, y, group=None, n_splits = 1000, class_order=None,
                    split_size=None, distance='mahalanobis',
                    cov_estimator='oas', cov_method='class'):
    """
    Calculates similarity between points of each class.

    Parameters
    ----------
    X, y: arrays of same shape[0]
        Respectively the features and labels.

    group : array of shape equal to y, optional
        Half of groups will make each split.
        If None, y is used instead.

    n_splits : int, default 100
        How many times to repeat the separatioon and calculation of distances.

    class_order : list, optional
        Class ordering. If None, np.unique(y) is used

    split_size : int, optional
        size of each set on each split.
        if None, half of the groups are used in each (floor rounded)

    distance : {'mahalanobis', 'euclidean'}
        How to measure distance between split means.

    cov_estimator : {'oas', 'lw', 'ml'}
        Which method will decide regularization strength
        Ignored if distance is 'euclidean'

    cov_method : {'class', 'split', 'single'}
        single - only one covariance for whole dataset
        class - one covariance per class
        split - one covariance per class per split
        Ignored if distance is 'euclidean'
    """

    classes = np.unique(y) if class_order is None else class_order
    groups = classes if group is None else np.unique(group)

    split_size = len(groups)//2 if split_size is None else split_size
    sh = GroupShuffleSplit(n_splits, split_size, split_size)

    if distance is 'mahalanobis':
        clf = MahalanobisClassifier(classes=classes, estimator=cov_estimator,
                                        shared_cov= (cov_method == 'single'),
                                        assume_centered=False)
        if cov_method != 'split':
            clf.fit_cov(X, y)
    else:
        raise NotImplementedError

    results = pd.DataFrame()
    for cv_i, (idx_1, idx_2) in enumerate(sh.split(X, y, group)):
        X_1, X_2, y_1, y_2  = X[idx_1], X[idx_2], y[idx_1], y[idx_2]
        mean_1 = [X_1[y_1==yi].mean(axis=0) for yi in classes]
        mean_2 = [X_2[y_2==yi].mean(axis=0) for yi in classes]

        if cov_method == 'split':
            clf.fit_cov((X_1, X_2), (y_1, y_2), is_tuple=True)

        dists_1 = clf.fit(X_1, y_1).transform(mean_2)
        dists_2 = clf.fit(X_2, y_2).transform(mean_1)

        local_res = pd.DataFrame(np.vstack((dists1,dists2)),
                        index=pd.Index(np.hstack((classes, classes)),
                                                name='Real Time'),
                        columns=pd.Index(classes, name='Target time'))
        local_res['split'] = [1]*len(classes) + [2]*len(classes)
        local_res['cv'] = cv_i

        results = results.append(local_res)

    return results


class MahalanobisClassifier():
    """

    Attributes
    ----------
    estimator : {'oas', 'lw', 'ml'}
        Which kind of regularization to use

    Notes
    -----
    The _fit_ method can be supplied with tuples (X1, X2), (y1, y2)
    in which case it will remove the mean of each. It must be explicited
    by the is_tuple argument
    """
    def __init__(self, shared_cov=False, estimator='oas',
                    assume_centered=False, classes=None, **kwargs):
        self.shared_cov = shared_cov
        self.classes = classes

        base = self._set_estimator(estimator)
        self.estimator = base(assume_centered=assume_centered, **kwargs)

        self._cov_fitted = False

    def _set_estimator(self, estimator):
        if estimator.lower() in ['lw', 'ledoitwolf']:
            return LedoitWolf
        elif estimator.lower() == 'oas':
            return OAS
        elif estimator.lower() in ['ml', 'empirical', 'empiricalcovariance']:
            return EmpiricalCovariance

    def fit(self, X, y):
        if self.classes is None:
            self.classes = np.unique(y)

        self.center_ = {yi : X[y==yi].mean(axis=0) for yi in self.classes}

        if not self._cov_fitted:
            self.fit_cov(X, y)

        return self

    def fit_cov(self, X, y, is_tuple=False):
        if is_tuple:
            # Remove the mean of each
            assert len(X) == 2 and len(y) == 2
            x0 = pd.DataFrame(X[0], index=y[0])
            x0 = x0 - x0.reset_index().groupby('index').mean()
            x1 = pd.DataFrame(X[1], index=y[1])
            x1 = x1 - x1.reset_index().groupby('index').mean()

            X = np.vstack((x0.values, x1.values))
            y = np.vstack(y[0], y[1])

        if self.shared_cov:
            precision = self.estimator.fit(X).get_precision()
            self.precision_ = {yi:precision for yi in self.classes}
        else:
            prec = lambda X, yi: self.estimator.fit(X[y==yi]).get_precision()
            self.precision_ = {yi : prec(X, yi) for yi in self.classes}
        self._cov_fitted = True


    def _predictOne(self, x):
        return np.argmin(self._transformOne(x))

    def predict(self, X):
        return np.array([self._predictOne(x) for x in X])

    def transform(self, X, y = None):
        return np.array([self._transformOne(x) for x in X])

    def _transformOne(self, x):
        return np.array([mahalanobis(self.center_[yi], x, self.precision_[yi]) for yi in self.classes])

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X,y)

    def get_params(self, deep=True):
        return {'warm_start':self.warm}

    def set_params(self):
        return self

class EuclideanClassifier():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.center_ = [X[y==yi].mean(axis=0) for yi in np.sort(np.unique(y))]
        assert np.max(y) == (len(np.unique(y))-1)

    def _predictOne(self, x):
        return np.argmin(self._transformOne(x))

    def predict(self, X):
        return np.array([self._predictOne(x) for x in X])

    def transform(self, X, y=None):
        return np.array([self._transformOne(x) for x in X])

    def _transformOne(self, x):
        return np.array([euclidean(x, center) for center in self.center_])

    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X, y)

    def get_params(self, deep=True):
        return {}

    def set_params(self):
        return self
