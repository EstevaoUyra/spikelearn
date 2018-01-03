"""
Contains the base classes for dealing with data directly.
The only data-dealing functions that
are accessible from other modules.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
import operator
#import spikeutils as spku



class Batcher():
    """
    Implements an iterable cross-validator that groups all repeated indexes in the same group.

    Parameters
    ----------
    X : (n_examples, n_features) ndarray
        Sklearn-style example matrix

    y : (n_examples, 1) ndarray
        Label of each example.

    axis : int or string
        Dimension to be sliced and served in batches. Accepts integer axis number, or 'last', 'first', and abbreviations.

    group : np.ndarray (n_examples, 1)
        Use it to make examples always part of the same fold. *Default:* The label itself (no preferential grouping).

    mode : string or sklearn generator, optional, default
        Which kind of splitting will be used for batching.
        Currently accepts 'shuffle' and 'kfold'.

    train_size : (int) or (float), optional, default: 0.5
        If int, total number of groups in the train set. If float, proportion of groups.

    test_size : (int) or (float), optional, default: 0.5
        If int, total number of groups in the test set. If float, proportion of groups.

    **kwargs :
        Extra keyword arguments will be passed to the splitting function.


    """
    #TODO make better 'yields' description
    def __init__(self, X, y=None, axis='Last', group=None, mode='sh', train_size=.5, test_size=.5, **kwargs):

        ## Attributes
        self.X, self.y = X, y
        self.train_size, self.test_size = train_size, test_size
        self.mode = mode
        self.cv = cv
        # Axis string to number
        if axis in ['Last', 'last', 'l', -1]:
            axis = len(X.shape)-1
        elif axis in ['First', 'first', 'f', 'row', 'rows']:
            axis = 0
        assert type(axis) is int
        self.axis = axis
        # Grouping
        if group is None:
            self.group = np.arange(X.shape[self.axis])
        else:
            assert len(group) == X.shape[self.axis]
            self.group = group

        ## Private attributes
        self._init_mode(**kwargs)
        self._i = 0

    def _init_mode(self, **kwargs):
        """
        Initializes the splitting function.
        """
        #TODO implement cross-validator function initializer
        indexes = np.unique(self.group)
        if  self.mode.lower() in ['sh', 'shuffle', 'shufflesplit', 'markov']:
            splitter = ShuffleSplit(n_splits=self.cv, test_size=self.test_size,train_size=self.train_size, **kwargs)

        elif self.mode.lower() in ['kfold', 'kf']:
            splitter = KFold(n_splits=self.cv, **kwargs)

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


def select(dataframe, maxlen=None, takefrom=None, **kwargs):
    """
    Parameters
    ----------
    dataframe : pandas DataFrame
    The data that will be under selection

    maxlen : int
    Maximum number of rows of resulting dataframe.
    Acts after all selection by kwargs.
    Has no effect if dataframe is already smaller than maxlen by then.

    takefrom : str
    Only functional when maxlen is not None.
    Specifies where to get the rows.
    May be one of 'shuffle', 'init', 'end'


    Keyword Arguments
    -----------------
    Key : string
    The index of any field in the DataFrame
    If type is numerical, may be preceded or succeded by max, min,
    maxeq or mineq, inside underlines
    it may also receive special key "maxshape", in which case


    Value : string or numerical
    The value used for selecting.

    Examples
    --------
    >>> select(data, _min_duration=1000)
    """
    localdata = dataframe.copy()
    ops = { '_mineq_': operator.ge,
            '_min_': operator.gt,
            '_maxeq_': operator.le,
            '_max_': operator.lt}

    # Select by the wanted values
    operation = None
    for key in kwargs:
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
    if maxlen is None or size <= maxlen:
        return localdata
    else:
        if takefrom is 'init':
            return localdata[:maxlen]
        elif takefrom is 'end':
            return localdata[-maxlen:]
        elif takefrom is 'shuffle':
            return localdata.sample(maxlen)
