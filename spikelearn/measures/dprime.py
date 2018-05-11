import numpy as np
import pandas as pd

def pooled_var(var, sizes):
    assert len(var) == 2
    assert len(sizes) == 2
    return ((sizes[0] - 1)*var[0] + (sizes[1]-1)*var[0])/(sizes.sum()-2)

def cohen_d(value_var, id_var, df=None):
    """
    Calculates Dprimes between different values for id_var, pairwise.
    Currently only accepts id_var with two values.
    Deals quietly with unbalanced sample sizes.


    Parameters
    ----------
    df : DataFrame
        data holder

    value_var : string, or array
        Name of the column in which the values are
        Or the values themselves

    id_var : string, or list of strings, or array of values
        Name of the columns to use to differentiate
        Or array of ids

    """
    if df is None:
        df = pd.DataFrame({'values' : value_var,
                            'ids':id_var} )
        value_var = 'values'
        id_var = 'ids'
        print(df)

    val_mean = df.groupby(id_var).mean()[value_var]
    val_var = df.groupby(id_var).var()[value_var]
    sizes = df.groupby(id_var).apply(lambda x: x.shape[0])

    if type(id_var) == list and len(id_var) > 1:
        raise NotImplementedError
    elif df[id_var].unique().shape[0] == 1:
        return np.nan
    else:
        assert type(id_var) is str
        assert df[id_var].unique().shape[0] == 2
        dif = np.abs(val_mean.diff().values[1])
        pool_std = np.sqrt(pooled_var(val_var.values, sizes.values))
        return dif/pool_std

from scipy.optimize import curve_fit
from sklearn.metrics import explained_variance_score
from scipy.stats import norm

#TODO document functions

def dprime(pred):
    TP = pred.astype(int).sum()/len(pred)
    FP = (pred==False).astype(int).sum()/len(pred)
    return (norm.ppf(TP) - norm.ppf(FP))

def step(x, baixo, alto, th):
    return baixo*((x<th).astype(int)) + alto*((x>=th).astype(int))

def best_step(x, y):
    ths = x[2:-2]
    scores = np.array([((y - degrau(x, y[x<th].mean(), y[x>=th].mean(), th))**2).sum() for th in ths])
    th = ths[scores.argmin()]
    return y[x<th].mean(), y[x>=th].mean(), th
