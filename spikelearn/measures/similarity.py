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
        Window type to be used. Default 'bartlett'. See more on:
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
