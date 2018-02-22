import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, cross_val_predict, GroupKFold
from sklearn.base import clone

from numpy.random import permutation
from scipy.stats import pearsonr

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

pearson_score = lambda true, pred: pearsonr(true, pred)[0]
def shuffle_val_predict(clf, dfs, names, X, y, group=None,
                         cv='sh', n_splits = 5,
                         train_size=.8,test_size=.2,
                         get_weights = True, score=pearson_score,
                         **kwargs):
    """
    Trains in each dataset, always testing on both, to calculate statistics
        about generalization between datasets.

    Parameters
    ----------
    X, y, group : indices [str[, str] ]
        The indices of each variable in the dataframes

    dfs : list of pandas DataFrames
        The data holders

    names : list of strings
        The ID variables of each DataFrame

    cv : str or callable, default 'sh'
        The splitting method to use.

    n_splits : int
        Number of splits to be done.

    get_weights : bool
        Whether to save and return the weights of each model

    score : callable
        function( true_label, pred_label ) -> number
        Defaults to pearson's correlation

    Keyword Arguments
    -----------------
    Extra kwargs will be passed on to the cv function

    Returns
    -------
    results : DataFrame
        Columns are [*y, predictions_max, predictions_mean, true_label,
                        cv, group, trained_on, tested_on, trained_here]

    weights : DataFrame
        Columns are [*y, trained_on, cv]

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
                    trained_here = True
                    idx = test_idx
                else:
                    trained_here = False
                    size = int(n_y*n_test)
                    idx = permutation(testdf.index)[:size]

                predictions = pd.DataFrame(clf_local.predict_proba( df[X].values[idx]), columns = classes)
                predictions['predictions_max'] = predictions.apply(lambda x: x.index[np.argmax(x.values[:n_y])], axis=1)

                predictions['predictions_mean'] = predictions.apply(lambda x: np.sum(predictions.columns[:len(np.unique(df[y].values))].values*x.values[:len(np.unique(df[y].values))]), axis=1)

                # Add identifiers
                predictions['cv'] = i
                predictions['group'] = df[group].values[idx]
                predictions['true_label'] = df[y].values[idx]
                predictions['trained_on'] = name
                predictions['tested_on'] = testname
                predictions['trained_here'] = trained_here
                predictions['score_max'] = score(predictions['predictions_max'], df[y].values[idx])
                predictions['score_mean'] = score(predictions['predictions_mean'], df[y].values[idx])
                results = results.append(predictions)

            if get_weights:
                w = pd.DataFrame(clf_local.coef_,
                                    columns = np.arange(df[X].shape[1]),
                                    index = pd.Index( classes.astype(int),
                                                      name=y )  )
                w = w.reset_index().melt(var_name='unit', id_vars=['time'])
                w['cv'] = i
                w['trained_on'] = name
                weights = weights.append(w)

    # Calculate Z-score
    fields=['score_max', 'score_mean']
    scores = results.drop_duplicates(['trained_on', 'tested_on'])

    def SEM(df):
        return df.std()/np.sqrt(df.shape[0])
    def pool_SEM(df):
        pool = df.groupby('trained_here').get_group(True)[fields]
        gtest = df.groupby('tested_on')
        return gtest.apply(lambda df: SEM(pd.concat((df[fields], pool))))
    def Z_score(df):
        df_sem = pool_SEM(df)
        df_means = df.groupby('tested_on').mean()
        return df_means/df_sem
    stats = scores.groupby('trained_on').apply(Z_score)


    if get_weights:
        return results, weights, stats
    return results, stats
