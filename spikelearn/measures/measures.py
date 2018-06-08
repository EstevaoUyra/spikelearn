import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from scipy.stats import norm

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

def jointPSTH():
    """
    Calculates Joint Peri-Stimulus Time Histogram
    A measure of functional interaction between neurons.

    Source
    ------
    https://www.ncbi.nlm.nih.gov/pubmed/18839091
    """
    raise NotImplementedError
