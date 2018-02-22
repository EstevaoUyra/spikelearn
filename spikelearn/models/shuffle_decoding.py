import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, cross_val_predict, GroupKFold
from sklearn.base import clone

from numpy.random import permutation
from scipy.stats import pearsonr

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
    for traindf, name in zip(dfs, names):
        print('training', name)
        for i, (train_idx, test_idx) in enumerate(sh.split(traindf[X], traindf[y], traindf[group])):
            clf_local = clone(clf)
            clf_local.fit( traindf[X].values[train_idx], traindf[y].values[train_idx] )

            # also test on each dataset
            for testdf, testname in zip(dfs, names):

                if testname == name:
                    trained_here = True
                    idx = test_idx
                else:
                    trained_here = False
                    size = len(test_idx)
                    idx = permutation(testdf.shape[0])[:size]
                    
                predictions = pd.DataFrame(clf_local.predict_proba( testdf[X].values[idx]), columns = classes)
                predictions['predictions_max'] = predictions.apply(lambda x: x.index[np.argmax(x.values[:n_y])], axis=1)

                predictions['predictions_mean'] = predictions.apply(lambda x: np.sum(predictions.columns[:len(np.unique(testdf[y].values))].values*x.values[:len(np.unique(testdf[y].values))]), axis=1)

                # Add identifiers
                predictions['cv'] = i
                predictions['group'] = testdf[group].values[idx]
                predictions['true_label'] = testdf[y].values[idx]
                predictions['trained_on'] = name
                predictions['tested_on'] = testname
                predictions['trained_here'] = trained_here
                predictions['score_max'] = score(predictions['predictions_max'], testdf[y].values[idx])
                predictions['score_mean'] = score(predictions['predictions_mean'], testdf[y].values[idx])
                results = results.append(predictions)

            if get_weights:
                w = pd.DataFrame(clf_local.coef_,
                                    columns = np.arange(traindf[X].shape[1]),
                                    index = pd.Index( classes.astype(int),
                                                      name=y )  )
                w = w.reset_index().melt(var_name='unit', id_vars=['time'])
                w['cv'] = i
                w['trained_on'] = name
                weights = weights.append(w)

    # Calculate Z-score
    fields=['score_max', 'score_mean']
    scores = results.drop_duplicates(['trained_on', 'tested_on', 'cv'])
    def SEM(df):
        return df.std()/np.sqrt(df.shape[0])
    def pool_SEM(df):
        pool = df.groupby('trained_here').get_group(True)[fields]
        gtest = df.groupby('tested_on')
        return gtest.apply(lambda df: SEM(pd.concat((df[fields], pool))))
    def pool_MEAN(df):
        pool = df.groupby('trained_here').get_group(True)[fields]
        gtest = df.groupby('tested_on')
        return gtest.apply(lambda df: (pd.concat((df[fields], pool))).mean())
    def Z_score(df):
        df_sem = pool_SEM(df)
        df_means = df.groupby('tested_on').mean()[fields]
        df_pm = pool_MEAN(df)
        return (df_means-df_pm)/df_sem
    stats = scores.groupby('trained_on').apply(Z_score)


    if get_weights:
        return results, weights, stats
    return results, stats
