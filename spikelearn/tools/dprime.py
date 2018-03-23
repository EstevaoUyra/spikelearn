import numpy as np

def Dprime(df, value_var, id_vars):
    """
    Calculates Dprimes between different id_vars, pairwise.

    Parameters
    ----------
    df : DataFrame
        data holder

    value_var : string
        Name of the column in which the values are

    id_vars : string, or list of strings
        Name of the columns to use to differentiate

    """

    val_mean = df.groupby(id_vars).mean()[value_var]
    val_var = df.groupby(id_vars).var()[value_var]

    if type(id_vars) == list and len(id_vars) > 1:
        raise NotImplementedError
    else:
        assert type(id_vars) is str
        assert df[id_vars].unique().shape[0] == 2
        dif = val_mean.diff().values[0]
        pool_std = np.sqrt((val_var**2).sum())
        return dif
