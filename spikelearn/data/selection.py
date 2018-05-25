"""
Contains the base classes for dealing with data directly.
The only data-dealing functions that
are accessible from other modules.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
import operator

class Batcher():
    """
    Implements an iterable cross-validator that groups all repeated indexes in the same group.

    Parameters
    ----------
    X : numpy ndarray or pandas DataFrame
        Data container to be sliced.

    y :  ndarray(n_examples, 1) or int, optional
        If ndarray, label of each example.
        If int, the axis that defines the label.

    axis : int or str
        If int, dimension to be sliced and served in batches,
        requiring ndarray.
        If string, field used to choose examples, requiring DataFrame.
        n_examples = X.shape[axis]

    group : np.ndarray (n_examples, 1) or string, optional, default None
        Use it to make examples always part of the same fold. Defaults to The position in axis (no preferential grouping).
        If string, X must be a DataFrame, and defines the grouping index

    mode : string or sklearn generator, optional, default
        Which kind of splitting will be used for batching.
        Currently accepts 'shuffle' and 'kfold'.

    flatten : bool
        Whether to return an Sklearn-style matrix.

    ylabels : tuple, optional
        If y is given as an int, its position on the axis is used to access
        the corresponding label in ylabels. defaults to the position itself

    Keyword arguments
    -----------------
    Arguments to be passed to the splitting function.
    Common kwargs are

    train_size : (int) or (float), optional, default: 0.5
        If int, total number of groups in the train set. If float, proportion of groups.

    test_size : (int) or (float), optional, default: 0.5
        If int, total number of groups in the test set. If float, proportion of groups.


    """
    #TODO make better 'yields' description
    def __init__(self, X, y=None, axis='Last', group=None, mode='sh',
                flatten=False, ylabels=None, **kwargs):

        ## Data container
        self.X = X
        if type(X) is np.ndarray:
            assert type(axis) is int
            if group is not None:
                assert type(group) is np.ndarray
                assert len(group) == X.shape[axis]
        elif type(X) is pd.DataFrame:
            assert type(axis) is str
            assert str in X.columns
            if group is not None:
                assert type

        # Labels
        self.y = y
        if y is not None:
            if type(y) is np.ndarray:
                # TODO assertion
                pass

        # Axis string to number
        self.axis = axis

        # Grouping
        if group is None:
            self.group = np.arange(X.shape[self.axis])
        else:
            assert len(group) == X.shape[self.axis]
            self.group = group

        # Start batching
        self.mode = mode
        self._init_mode(**kwargs)
        self._i = 0

    def _init_mode(self, **kwargs):
        """
        Initializes the splitting function.
        """
        if callable(self.mode):
            return NotImplementedError
        assert type(self.mode) is str

        indexes = np.unique(self.group)
        if  self.mode.lower() in ['sh', 'shuffle', 'shufflesplit', 'markov']:
            splitter = ShuffleSplit(n_splits=self.cv, test_size=self.test_size,train_size=self.train_size, **kwargs)

        elif self.mode.lower() in ['kfold', 'kf']:
            splitter = KFold(n_splits=self.cv, **kwargs)

        else:
            raise ValueError("The mode {} is not supported. Try 'kf' or 'sh'".format(self.mode))

        self._batcher_gen = splitter.split( np.unique( self.group))

    def _current_batch(self):
        """
        Returns
        -------
        _next_ train and test sets.

        Warning
        -------
        This function iterates the splitting, but does not increase Batcher's iterator number. It should *not* be used by itself.

        """
        train_groups, test_groups = self._batcher_gen.next()
        train_indices = np.isin(self.group, train_groups)
        test_indices = np.isin(self.group, test_groups)

        def x_and_y_from_idx(self, idx):
            if self.y is not None:
                return self.X.take(idx,axis=self.axis), self.y[idx]
            else:
                return (self.X.take(idx,axis=self.axis))

        return (*x_and_y_from_idx(train_indices), *x_and_y_from_idx(test_indices))

    def __iter__(self):
        return self

    def next(self):
        """
        Yields
        ------
        (train_size, n_features), (test_size, n_features)
        (train_size, n_features), (train_size,1), (test_size, n_features),(test_size,1)

        Examples
        -----
        >>> for X_train, X_test in Batcher(X, train_size=3, test_size=.4):
        >>>     assert X_train.shape[0] == 3
        >>>     assert X_test.shape[0] == 0.4 * X.shape[0]
        >>> for X_train, y_train, X_test, y_test in Batcher(X, y):
        >>>     assert X_train.shape[0] == y_train.shape[0]
        """
        if self._i < self.cv:
            self._i += 1
            return _current_batch
        else:
            raise StopIteration()


def select(dataframe, maxlen=None, takefrom=None, accept_smaller=False,
                **kwargs):
    """
    Parameters
    ----------
    dataframe : pandas DataFrame
        The data that will be under selection

    maxlen : int
        Maximum number of rows of resulting dataframe.
        Acts after all selection by kwargs.
        If dataframe is already smaller than maxlen by then,
        resides on accept_smaller parameter.

    takefrom : str
        Only functional when maxlen is not None.
        Specifies where to get the rows.
        May be one of 'shuffle', 'init', 'end'

    accept_smaller : bool
        Whether to silently accept dataframes smaller than maxlen.
        If False and it is smaller, raises an exception
        Defaults to False

    Keyword Arguments
    -----------------
    Key : string
        The index of any field in the DataFrame
        If type is numerical, may be preceded or succeded by _in_, _max_, _min_,
        _maxeq_ or _mineq_, inside underlines.


    Value : string or numerical
        The value used for selecting.

    Examples
    --------
    >>> selected_by_duration = select(data, _min_duration=1000)
    >>> selected_neurons = select(data, _in_unit=[1,2,3,8])

    Notes
    -----
    The wordparts _in_, _min_, _mineq_, _maxeq_, _max_ should not be
    part of identifier variables,
    and have to be reserved for using the comparisons.
    """
    localdata = dataframe.copy()
    ops = { '_mineq_': operator.ge,
            '_min_': operator.gt,
            '_maxeq_': operator.le,
            '_max_': operator.lt,
            '_in_': lambda x, y: x.isin(y)}

    # Select by the wanted values
    for key in kwargs:
        field = key; operation = None;
        for op in ops:
            if op in key:
                operation = ops[op]
                field = key.replace(op,'')
                assert field in dataframe.columns

        if operation is None:
            operation = operator.eq
        localdata = localdata[ operation(localdata[field],kwargs[key]) ]

    # Return dataframe of expected size
    size = localdata.shape[0]
    if maxlen is None or size == maxlen:
        return localdata
    elif size < maxlen:
        assert accept_smaller
        return localdata
    else:
        if takefrom is 'init':
            return localdata.iloc[:maxlen]
        elif takefrom is 'end':
            return localdata.iloc[-maxlen:]
        elif takefrom is 'shuffle':
            return localdata.sample(maxlen)

def to_feature_array(df, Xyt = False, subset='cropped'):
    """
    Receives an epoched smoothed dataframe, and transforms it to put the each
    unit into a column.
    """
    subset= [subset, subset+'_times']
    df = df.copy()[subset]

    bins = df.applymap(len).min()[0]
    df = df.applymap(lambda x: x[:bins])

    rates = pd.DataFrame(df[subset[0]].tolist(),
                index=df.index).reset_index().melt(['trial','unit'])
    times = pd.DataFrame(df[subset[1]].tolist(),
                index=df.index).reset_index().melt(['trial','unit'])

    rates['time'] = times.value
    rates = rates.drop('variable',axis=1)
    rates = rates.set_index(['trial', 'time','unit']).unstack()
    rates.columns = rates.columns.droplevel()
    if Xyt:
        X = rates.values
        rates = rates.reset_index()
        y = rates['time'].values
        trial = rates['trial'].values
        return X, y, trial
    else:
        return rates
