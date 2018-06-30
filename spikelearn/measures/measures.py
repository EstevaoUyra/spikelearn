import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from scipy.stats import norm
from itertools import combinations

#TODO Organize location of functions


def step(x, low, high, th):
    """
    Step function is low if x<th, else it is high
    """
    return low*((x<th).astype(int)) + high*((x>=th).astype(int))

def best_step(x, y):
    """
    Fits a step function to the data.

    Returns
    -------
    low, high, threshold : floats
        The parameters of the step function


    """
    ths = x[2:-2]
    scores = np.array([((y - degrau(x, y[x<th].mean(), y[x>=th].mean(), th))**2).sum() for th in ths])
    th = ths[scores.argmin()]
    return y[x<th].mean(), y[x>=th].mean(), th

def jointPSTH_multiple(C, subtract=True):
    """
    Calculates Joint Peri-Stimulus Time Histogram,
    for all combinations of neurons

    Parameters
    ----------
    C : 3d array of shape (trials, times, neurons)

    Returns
    -------
    Jmultiple : dictionary of 2d arrays
        Indexed by tuples of the indices of neurons being compared
        Returns tuple of arrays if not subtract, see jointPSTH

    See Also
    --------
    jointPSTH

    """
    Jmultiple={}
    for i, j in combinations(np.arange(C.shape[2])):
        Jmultiple[(i,j)] = jointPSTH(C[:,:,i], C[:,:,j], subtract)

def jointPSTH(X1, X2, subtract=True):
    """
    Calculates Joint Peri-Stimulus Time Histogram
    A measure of functional interaction between neurons.

    Parameters
    ----------
    X1, X2 : 2d arrays of shape (trials, times)


    Returns
    -------
    J :  2d array
        The resultant histogram.
        With the shuffled-trials histogram subtracted if subtract is True.

    raw : 2d array
        Returned only if subtract is false.

    Source
    ------
    https://www.ncbi.nlm.nih.gov/pubmed/18839091
    """
    raise NotImplementedError
