import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import norm

def bracketing(arr, border_size=1, range=None):
    """
    A simplified measure of 'U-shapeness'. Negative values imply inverted U.
    Mean values at center subtracted from mean border values.

    Parameters
    ----------
    arr : 1-d array
        Activity upon which to calculate the bracketing.

    border_size : int, optional
        How many bins at beginning and end of range to use as border.
        Defaults to one.

    range : tuple, optional
        inclusive interval, defaults to full vector
    """
    if range is None:
        range = (0, len(arr)-1)
    center = np.arange(range[0]+border_size, range[1]-border_size+1)
    borders = np.hstack( (np.arange(range[0], range[0]+border_size),
                          np.arange(range[1]-border_size+1, range[1]+1)) )

    if any( np.isin(center, borders) ) or any(np.bincount(borders)>1):
        raise ValueError("Border size %d is causing overlap"%border_size)

    return arr[borders].mean() - arr[center].mean()

def unit_similarity_evolution(epoched_vector, window=1,
                                win_type='bartlett', **kwargs):
    """
    Calculates the similarity profile of a given neuron along all trials of a given segment. Note that epoched vector is given in the tidy format.

    Parameters
    ----------
    epoched_vector : Series or DataFrame
        Firing-rate indexed by 'trial' and 'time'

    window : int, optional
        number of trials to aggregate via moving average
        default 1

    win_type : string, optional
        Window type to be used. Default 'bartlett'. See more on:[]
        https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rolling.html

    **kwargs :
        Extra keyword arguments are passed to the window function.

    Returns
    -------
    similarity_profile : DataFrame (trial, trial)
        The pearson similarity along the session.
        If window is 1, it is the pairwise correlation matrix between trials.
        Elsewise, it reduces


    """
    return epoched_vector.reset_index().pivot('trial', 'time').rolling(window, center=True, win_type=win_type, **kwargs).mean().transpose().corr()

def ramping_trajectory():
    """
    Calculates the pearson coefficient for multiple durations.


    Returns
    -------
    ramping_p : array
        An array of log-odds for each duration.
    """
    raise  NotImplementedError

def ramping_p(firing_rate, times=None, range=None, return_r=False):
    """
    Calculates the pearson coefficient for the specified interval.

    Parameters
    ----------
    firing_rate : array
        Estimated firing rate in a single trial.
        Must have a single non-singleton dimension.

    times : array, optional
        The time relative to the firing rate.
        Must be of the same shape as firing rate.
        If not provided, defaults to range(len(firing_rate))

    range : tuple, optional
        The times to include in the analysis.
        Defaults to the full range.


    Returns
    -------
    ramping_p : array
        An array of log-odds for each duration.
    """
    if type(firing_rate) is pd.Series:
        firing_rate = firing_rate.values
    if type(times) is pd.Series:
        times = times.values
    if times is None:
        times = np.arange(len(firing_rate))
    if range is None:
        use_times = np.arange(len(firing_rate))
    else:
        use_times = np.logical_and(times>range[0], times<range[1])

    assert len(times) == len(firing_rate)
    r, p = pearsonr(firing_rate[use_times], times[use_times])

    if return_r:
        return r, p
    else:
        return p

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

def dprime(pred):
    """
    Calculates dprime (like effect size) in the case all provided samples are positive.
    """
    TP = pred.astype(int).sum()/len(pred)
    FP = (pred==False).astype(int).sum()/len(pred)
    return (norm.ppf(TP) - norm.ppf(FP))
