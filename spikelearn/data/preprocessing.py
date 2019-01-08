import scipy.stats as st
import numpy as np
import pandas as pd
import collections

def kernel_smooth(spike_vector, sigma, edges, bin_size=None, padding='symmetric', border_correction = False):
    """
    Receives an array of spike times (point-process like), and smoothes it
    by convolving with a _gaussian_ kernel, of width *sigma*. The time
    position will be alocated with a time precision that is a ratio of sigma,
    given by tp = sigma/precision_factor.

    Parameters
    ----------
    spike_vector : array
        Point process like spike times, *in milisseconds*

    sigma : int
        Width of the window, in ms

    edges : tuple
        Starting and ending time of the window of interest, in ms.

    precision_factor : int, default 10
        Factor of the precision ratio sigma/temporal_precision

    bin_size : int, default None
        The size (in ms) of each step in the returning smoothed data.
        By default is the minimum, equal to 1ms.

    padding : str, default None
        The kind of padding on array edges. Possible values are
        'constant', 'edge', 'maximum', 'mean', 'median', 'minimum', 'reflect',
        'symmetric', 'wrap', or a <function>.

    border_correction : bool, default False
        whether to divide borders by spikevector true contribution
        Raises a ValueError if used adjoined with padding

    Returns
    -------
    smoothed_data : array
        The estimated firing rate as each interval of bin_size
        in *spikes per second*

    times : array
        The time at the left edge of each interval

    Notes
    -----
    Total kernel size is 6*sigma, 3 sigma for each size.

    See also
    --------
    numpy.pad for padding options and information.
    """


    tp = 1# int(sigma/precision_factor)
    if bin_size is None:
        bin_size = tp



    try:
        assert float(bin_size) == bin_size # Is multiple
    except AssertionError:
        raise ValueError("Bin size must be a multiple of temporal precision.")

    n_bins = int(bin_size*int((edges[1]-edges[0])/bin_size))
    edges= (edges[0], bin_size*int(n_bins/bin_size)+edges[0])
    if edges[1] <= edges[0]:
        return ([],[])

    if sigma is None:
        return np.histogram(spike_vector, bins=int((edges[1]-edges[0])/bin_size), range=edges)
        
    spike_count, times = np.histogram(spike_vector, bins=n_bins, range=edges)

    each_size_len = int(3*sigma + 1)
    if padding is not None:
        if border_correction:
            raise ValueError('Padding and correction cannot be used together')
        spike_count = np.pad(spike_count, each_size_len, padding)
    s=sigma # Just for one-lining below
    kernel = st.norm(0,s).pdf( np.linspace(-3*s, 3*s, 2*each_size_len + 1) )
    smoothed = np.convolve(spike_count, kernel,
                'valid' if padding is not None else 'same')

    if border_correction:
        contrib = st.norm(0,s).cdf(np.linspace(0, 3*s, each_size_len))
        smoothed[:each_size_len] /=  contrib
        smoothed[-each_size_len:]/= contrib[::-1]

    cs = np.hstack((0, smoothed.cumsum()))*1000/bin_size
    return np.diff(cs[::bin_size]), times[:-bin_size:bin_size]

def remove_baseline(activity, baseline, baseline_size=None):
    """
    Removes the mean baseline firing rate from the activity.

    Parameters
    ----------
    activity : DataFrame
        DataFrame of firing rates (in spikes/*s*)
        with a single Identifier column *by*, that will be used to select
        the corresponding baseline

    baseline : DataFrame
        Indexed in the same way as the important features of activity
        may be composed of the mean firing rate or the baseline spike times.
        BE CAREFUL: Do _NOT_ use smoothed firing rates

    baseline_size : number (default None)
        The duration of the baseline, *in seconds*.
        Ignored if firing rates are given
    """
    if isinstance(baseline.iloc[0,0], collections.Sized):
        assert baseline_size is not None
        firing_rate = lambda x: len(x)/baseline_size # Number of spikes per sec
        baseline = baseline.applymap(firing_rate)
    else:
        assert isinstance(baseline.iloc[0,0], float)

    return (activity - baseline)[activity.columns]
