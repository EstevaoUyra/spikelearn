import scipy.stats as st
import numpy as np

def kernel_smooth(spike_vector, sigma, edges, bin_size=None):
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
        By default is the minimum, equal to the temporal precision tp.
        Must be a multiple of tp (n*tp). The returning bin value is the *sum*
        of each n bins, with no superposition.

    Returns
    -------
    smoothed_data : array
        The estimated firing rate as each interval of bin_size

    times : array
        The time at the left edge of each interval

    Notes
    -----
    Convolution borders are edge-padded.
    Total kernel size is 6*sigma, 3 sigma for each size.
    """
    precision_factor = sigma
    tp = int(sigma/precision_factor)
    if bin_size is None:
        bin_size = tp

    nbins_to_agg = int(bin_size/tp)
    try:
        assert float(bin_size)/tp == nbins_to_agg # Is multiple
    except AssertionError:
        raise ValueError("Bin size must be a multiple of temporal precision.")

    n_bins = int( (bin_size*int(edges[1]/bin_size)-edges[0]) / tp )
    edges= (edges[0], bin_size*int(n_bins/nbins_to_agg)+edges[0])
    if edges[1] <= edges[0]:
        return ([],[])


    spike_count, times = np.histogram(spike_vector, bins=n_bins, range=edges)

    each_size_len = int(3*tp*precision_factor + 1)
    padded = np.pad(spike_count, each_size_len, 'edge')

    s=sigma # Just for one-lining below
    kernel = st.norm(0,s).pdf( np.linspace(-3*s, 3*s, 2*each_size_len + 1) )
    smoothed = np.convolve(padded, kernel, 'valid')

    cs = np.hstack((0, smoothed.cumsum()))
    return np.diff(cs)[::nbins_to_agg], times[:-nbins_to_agg:nbins_to_agg]
