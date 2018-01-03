"""
Utilitary functions.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind



def reyes_cp(timeseries,full_window_size=60):
    """
    Change-point analysis algorithm. Window-walks a timeseries calculating
    the odds of a simple t_test between the window's first and second half.

    Parameters
    ----------
    timeseries : array-like
        Series of values upon which to calculate the Change-point
    full_window_size : int, optional, default: 60
        Size of the total walking window (sum of two halfs)

    Returns
    -------
    odds : Series
        Change-point odds for each timepoint
    """
    rollin_window = pd.Series(timeseries).rolling(full_window_size,center=True)
    return rollin_window.apply(half_ttest_half)

def half_ttest_half(series):
    """
    Divides series in half and returns the t_test odds between each
    """
    halfN = len(series)//2
    t, p = ttest_ind(series[:halfN],series[halfN:])
    return odds(p)

def odds(p):
    return (1-p)/p

def verticalize_series(series, idx):
    """
    Melts a series in which the values are not only in one row,
    but are in a single field in this row.

    Parameters
    ----------
    series : pandas Series
        A series of id variables, with one single field containing
        an array-like variable.
    idx : string
        The index of the array-like variable.

    Returns
    -------
    df : DataFrame
        Tidy (melted), with one value of idx variable per row
    """
    df = pd.DataFrame(row[idx],columns=[idx])
    for id_var in row.index.drop(idx):
        df[id_var]=row[id_var]
    return df

def verticalize_df(df, idx):
    """
    Melts a DataFrame in which the to-melt values are not distributed in the
    row, but are in a single field in each row.

    Parameters
    ----------
    df : DataFrame
        A DataFrame of id variables, with one single field containing
        an array-like variable.
    idx : string
        The index of the array-like variable.

    Returns
    -------
    df : DataFrame
        Tidy (melted), with one value of idx variable per row
    """
    return pd.concat([verticalize_series(row,idx) for _,row in df.iterrows()])

def slashed(path):
    assert type(path) is str
    if path=='':
        return '/'
    elif path[0] == '/':
        return path
    else:
        return '/'+path

def recursive_full_name_recovery(inside_folder_shortcuts):
    """
    Receives a nested dictionary in which keys are folder names,
    and values are filenames, and returns the fullpaths of each.
    """
    if type(inside_folder_shortcuts) is str:
        return [slashed(inside_folder_shortcuts)]
    elif type(inside_folder_shortcuts) is dict:
        paths = []
        for folder in inside_folder_shortcuts:
            for path in recursive_full_name_recovery(inside_folder_shortcuts[folder]):
                paths+= [slashed(folder)+path]
        return paths

def get_filepaths_from_shortcut(one_shortcuts):
    all_paths = []
    for folder in ['data','results']:
        all_paths+= ['{}/{}{}'.format(one_shortcuts['basepath'],folder,path) for path in recursive_full_name_recovery(one_shortcuts[folder])]
    return np.array(all_paths)
