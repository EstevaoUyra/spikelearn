import numpy as np
from scipy.stats import pearsonr

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
    if times is None:
        times = range(len(firing_rate))
    if range is None:
        use_times = range(len(firing_rate))
    else:
        use_times = np.logical_and(times>range[0], times<range[1])

    assert len(times) == len(firing_rate)
    r, p = pearsonr(firing_rate[use_times], times[use_times])

    if return_r:
        return r, p
    else:
        return p
