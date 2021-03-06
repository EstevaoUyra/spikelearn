"""
Utilitary functions.
"""
import numpy as np
import pandas as pd

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
    for folder in ['data']:
        all_paths+= ['{}/{}{}'.format(one_shortcuts['basepath'],folder,path) for path in recursive_full_name_recovery(one_shortcuts[folder])]
    return np.array(all_paths)

def df_to_array(df, axis_fields, field_values=None):
    # Assert there is exactly one value for each combination of fields
    assert all(df.groupby(axis_fields).agg(lambda x: x.shape[0])==1)
    if field_values is None:
        field_values = {field:df[field].unique() for field in axis_fields}
    out_shape = (len(field_values[field]) for field in axis_fields)
    arr = np.full(out_shape, np.nan)
    raise NotImplementedError

def ndarray_to_df(arr, field_names):
    raise NotImplementedError
