from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS
from scipy.spatial.distance import mahalanobis
import numpy as np

class MahalanobisClassifier():
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

    def fit(self,X,y):
        if self.classes is None:
            self.classes = np.unique(y)

        self.center_ = {yi : X[y==yi].mean(axis=0) for yi in self.classes}

        if not self._cov_fitted:
            self.fit_cov(X, y)

        return self

    def fit_cov(self, X, y):
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

    def transform(self,X ,y = None):
        return np.array([self._transformOne(x) for x in X])

    def _transformOne(self, x):
        return np.array([mahalanobis(self.center_[yi], x, self.precision_[yi]) for yi in self.classes])

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X,y)

    def get_params(self, deep=True):
        return {'warm_start':self.warm}

    def set_params(self):
        return self
