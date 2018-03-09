import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, cross_val_predict, GroupKFold
from sklearn.base import clone

from numpy.random import permutation
from scipy.stats import pearsonr

pearson_score = lambda true, pred: pearsonr(true, pred)[0]

class shuffle_val_results():
    """
    Organization dataholder for results from
    shuffle_val_predict
    """
    def __init__(self, n_splits, train_size,
                    test_size, scoring_function,
                    classes, groups, features, id_vars):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.scoring_function = scoring_function

        self.classes = classes
        self.groups = groups
        self.features = features
        self.id_vars =  id_vars
        self.fat_vars = id_vars + ['true_label', 'group']

        self.data = pd.DataFrame(self.id_vars)
        self.proba = pd.DataFrame(self.id_vars)
        self.weights = pd.DataFrame(self.id_vars)
        self.predictions = pd.DataFrame(self.id_vars)

    def _input_id_vars(self, df, **kwargs):
        for key in self.id_vars:
            df[key] = kwargs[key]
        return df

    def _thin(self, df, let_labels=False):
        if let_label:
            to_drop = self.fat_vars
        else:
            to_drop = self.id_vars

    return df.drop(to_drop, axis=1)

    def proba_matrix(self):
        raise NotImplementedError

    def confusion_matrix(self):
        raise NotImplementedError

    def append_probas(self, probas,
                        true_labels, groups, **kwargs):

        local = pd.DataFrame(probas, columns = self.classes)
        local['true_label'] = true_labels
        local = self._input_id_vars(local, **kwargs)
        self.proba = self.proba.append(local)

    def append_weights(self, weights, **kwargs):
        index = pd.Index(self.classes)
        local = pd.DataFrame(weights,
                                index = self.classes,
                                columns = self.features)
        local = local.reset_index().melt(var_name='feature',
                                      id_vars=[classes.name])
        local = self._input_id_vars(local, **kwargs)
        self.weights = self.weights.append(local)

    def calculate_predictions(self):
        pred_max = lambda x: self.classes[np.argmax(x)]
        self.predictions['predictions_max'] = self._thin(self.proba).apply(pred_max, axis = 1)

        pred_mean = lambda x: np.sum(predictions.drop('true_label',axis=1).columns.values*x.values)
        self.predictions['predictions_mean'] = self._thin(self.proba).apply(pred_mean, axis = 1)

        self.predictions[self.fat_vars] = self.proba[self.fat_vars]

    def score(self):


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
    # Number of training and testing is defined by the smallest dataframe
    size_smallest = min([df[group].unique().shape[0] for df in dfs])
    n_train = size_smallest * train_size
    n_test = size_smallest * test_size

    # Method of cross-validation
    if cv == 'kf':
        sh = GroupKFold(n_splits=n_splits, **kwargs)
    elif cv == 'sh':
        sh = GroupShuffleSplit(n_splits=n_splits,
                                train_size=n_train, test_size=n_test, **kwargs)
    elif isinstance(cv, object):
        sh=cv


    # Define the results format
    classes = pd.Index( np.unique(dfs[0][y]), name=y)
    id_vars = ['cv',
               'trained_on',
               'tested_on',
               'trained_here']
    res = shuffle_val_results(n_splits=n_splits,
                    train_size = n_train,
                    test_size = n_test,
                    scoring_function = score,
                    classes = classes,
                    groups = df[group].unique(),
                    features = X,
                    id_vars = id_vars)

    # Make the cross validation on each dataset
    for traindf, name in zip(dfs, names):
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

                probas = clf_local.predict_proba(testdf[X].values[idx])
                true_labels = testdf[y].values[idx]

                res.append_probas(probas, true_labels,
                                  cv=i, trained_on=name,
                                  tested_on=testname,
                                  trained_here=trained_here)



            if get_weights:
                res.append_weights(clf_local.coef_,
                                    cv=i, trained_on=name,
                                    tested_on= np.nan
                                    trained_here= np.nan)

                w = w.reset_index().melt(var_name='unit', id_vars=['time'])

                weights = weights.append(w)

    # TODO update statistics to results new format
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
