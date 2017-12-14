from math import gcd
def gcd_mult(numbers):
    numbers=set(numbers)
    if len(numbers)==1:
        return numbers.pop()
    elif len(numbers)==2:
        return gcd(*numbers)
    else:
        return gcd_mult(set(gcd(*pair) for pair in combinations(numbers,2)))

from itertools import combinations
def combinationsDif_meanVar(Usc,Xc):
    """
    Compares each _deep_row_ of array X,
    and returns mean variance in the difference of rows
    Xc has shape ( Tcriterions, time, components )
    """
    return np.array([(square(Usc)@(xi-xj).transpose()).var() for xi,xj in combinations(Xc,2)]).mean()

from scipy.signal import decimate
def downscale(X, outSize):
    """
    Downscales the columns of X,
    returns newX of shape (outSize, X.shape[1])
    """
    downFactor =  X.shape[0]/outSize
    assert downFactor == int(downFactor)
    return decimate(X.transpose(),int(downFactor),n=2).transpose()

def square(matrix):
    n = int(sqrt(len(matrix)))
    return matrix.reshape(n,n)

from numpy.linalg import slogdet
from numpy import sqrt
def log_det(flat_array):

    sign, value = slogdet(square(flat_array))
    return sign*value

from scipy.optimize import minimize
def minimize_differences(Xc):
    constrain = ({'type': 'eq',
                  'fun' : log_det,
                  })
    return minimize(lambda U: combinationsDif_meanVar(U,Xc), np.eye(Xc.shape[2]).ravel(),constraints=constrain,method='SLSQP',jac=False)


from sklearn.decomposition import PCA
class SCA():
    def __init__(self,n_components=3,new_size=None,pca_to_use=None):
        self.new_size = new_size
        self.n_components = n_components
        self.pca_to_use = n_components if pca_to_use is None else pca_to_use
        
    def _norm(self, X_list, down=True):
        pca = PCA(n_components=self.pca_to_use)
        if down:
            return [downscale(pca.fit_transform(X), self.new_size) for X in X_list]
        else:
            return [pca.fit_transform(X) for X in X_list]
    def fit(self, X_list, y=None):
        """
        Each X in X_list has the form (time, features)
        """
        if self.new_size is None:
            sizes = set(X.shape[0] for X in X_list)
            self.new_size = gcd_mult(sizes)


        res = square(minimize_differences(np.stack(self._norm(X_list) ) ).x )
        self.components_ = res[np.argsort((res**2).sum(axis=1))[::-1]][:self.n_components,:]
        return self

    def transform(self, X_list, y=None):
        return [(self.components_@X.transpose()).transpose() for X in self._norm(X_list,down=False)]

    def fit_transform(self, Xc, y=None):
        return self.fit(Xc).transform(Xc)
