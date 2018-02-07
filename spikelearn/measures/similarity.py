def unit_similarity_evolution(epoched_vector, window):
    """

    Parameters
    ----------
    epoched_vector : Series or DataFrame
        vector of activity indexed by 'trial' and 'time'

    window : int
        number of trials to aggregate via moving average

    """
    return epoched_vector.reset_index().pivot('trial', 'time').rolling(window, center=True).mean().transpose().corr()
