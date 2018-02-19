import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, cross_val_predict, GroupKFold
from sklearn.base import clone

from numpy.random import permutation

def deprecated(clf, X, y, group=None,
                         cv='kf', n_splits = 5,
                         get_weights = True, **kwargs):
    # TODO documentation
    """

    """
    def get_predictions_or_proba(clf, X, mode):
        """
        Local helper function to ease the switching between predict_proba and predict
        """
        if mode == 'predict':
            return pd.DataFrame(clf.predict(X), columns=['predictions'])
        elif mode in ['proba','probability']:
            try:
                return pd.DataFrame(clf.predict_proba(X), columns=np.unique(y))
            except:
                return pd.DataFrame(clf.decision_function(X), columns=np.unique(y))

    if cv == 'kf':
        sh = GroupKFold(n_splits=n_splits, **kwargs)
    elif cv == 'sh':
        sh = GroupShuffleSplit(n_splits=n_splits, train_size=.8,test_size=.2, **kwargs)
    elif isinstance(cv, object):
        sh=cv

    weights = pd.DataFrame()
    results = pd.DataFrame()
    n_y = len(np.unique(y))
    for i, (train_idx, test_idx) in enumerate(sh.split(X, y, group)):
        clf_local = clone(clf)
        clf_local.fit(X[train_idx,:],y[train_idx])

        for train_or_test in ['train', 'test']:
            idx = test_idx  if train_or_test is 'test' else train_idx
            predictions = get_predictions_or_proba(clf_local, X[idx], 'proba')
            predictions['cv'] = i
            predictions['group'] = group[idx]
            predictions['true'] = y[idx]
            predictions['predictions'] = predictions.apply(lambda x: x.index[np.argmax(x.values[:n_y])], axis=1)
            predictions['mean'] = predictions.apply(lambda x: np.sum(predictions.columns[:len(np.unique(y))].values*x.values[:len(np.unique(y))]), axis=1)
            predictions['set'] = train_or_test
            results = results.append(predictions)

        if get_weights:
            w = pd.DataFrame(clf_local.coef_, columns = np.arange(X.shape[1]), index = pd.Index(np.unique(y).astype(int), name='time'))
            w = w.reset_index().melt(var_name='unit', id_vars=['time'])
            w['shuffle'] = i
            weights=weights.append(w)
    if get_weights:
        return results, weights
    return results

def shuffle_val_predict(clf,dfs, names, X, y, group=None,
                         cv='sh', n_splits = 5,
                         train_size=.8,test_size=.2,
                         get_weights = True, **kwargs):
    # TODO complete docs
    """
    Trains in each dataset, always testing on both, to calculate statistics
        about generalization between datasets.

    Parameters
    ----------
    X, y, group : indices [str[, str] ]
        The indices of each variable in the dataframes

    dfs : pandas DataFrames
        The data holders

    names : list of strings
        The ID variables of each DataFrame

    cv :

    n_splits :

    get_weights :

    Keyword Arguments
    -----------------
    Extra kwargs will be passed on to the cv function
    """
    n_train = int(min([df[group].unique().shape[0]*train_size for df in dfs]))
    n_test = int(min([df[group].unique().shape[0]*test_size for df in dfs]))


    if cv == 'kf':
        sh = GroupKFold(n_splits=n_splits, **kwargs)
    elif cv == 'sh':
        sh = GroupShuffleSplit(n_splits=n_splits,
                                train_size=n_train, test_size=n_test, **kwargs)
    elif isinstance(cv, object):
        sh=cv

    weights = pd.DataFrame()
    results = pd.DataFrame()
    classes = np.unique(dfs[0][y])
    n_y = len(classes)

    # Make the cross validation on each dataset
    print(len(dfs), names)
    for df, name in zip(dfs, names):
        print('training', name)
        for i, (train_idx, test_idx) in enumerate(sh.split(df[X], df[y], df[group])):
            clf_local = clone(clf)
            clf_local.fit( df[X].values[train_idx], df[y].values[train_idx] )

            # also test on each dataset
            for testdf, testname in zip(dfs, names):

                if testname == name:
                    train_or_test = 'test'
                    idx = test_idx
                else:
                    size = int(n_y*n_test)
                    idx = permutation(testdf.index)[:size]
                predictions = pd.DataFrame(clf_local.predict_proba( df[X].values[idx]), columns = classes)
                predictions['predictions'] = predictions.apply(lambda x: x.index[np.argmax(x.values[:n_y])], axis=1)

                predictions['mean'] = predictions.apply(lambda x: np.sum(predictions.columns[:len(np.unique(df[y].values))].values*x.values[:len(np.unique(df[y].values))]), axis=1)

                # Add identifiers
                predictions['cv'] = i
                predictions['group'] = df[group].values[idx]
                predictions['true'] = df[y].values[idx]
                predictions['train'] = name
                predictions['test'] = testname

                results = results.append(predictions)

            if get_weights:
                w = pd.DataFrame(clf_local.coef_,
                                    columns = np.arange(df[X].shape[1]),
                                    index = pd.Index( classes.astype(int),
                                                      name=y )  )
                w = w.reset_index().melt(var_name='unit', id_vars=['time'])
                w['cv'] = i
                w['train'] = name
                weights = weights.append(w)

        if get_weights:
            return results, weights
    return results
