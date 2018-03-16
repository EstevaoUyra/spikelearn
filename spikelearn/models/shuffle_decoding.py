import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, cross_val_predict, GroupKFold
from sklearn.base import clone

from numpy.random import permutation
from scipy.stats import pearsonr

pearson_score = lambda true, pred: pearsonr(true, pred)[0]

class Results_shuffle_val():
    """
    Organization dataholder for results from
    shuffle_val_predict

    Attributes
    ----------
    n_splits, train_size, test_size : int
        Parameters used in the analysis.

    scoring_function : list of tuples [(name, callable), ]
        the scoring function used in the analysis

    classes, groups, features : arrays
        Characteristics of the dataset

    id_vars, fat_vars : list of string
        The name of identifier variables
        Including 'true_label' and 'group' in fat.

    data, proba, weights, predictions, scores, stats: DataFrames
        The results of the analysis


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

        self.proba = pd.DataFrame(columns = self.id_vars)
        self.weights = pd.DataFrame(columns = self.id_vars)
        self.predictions = pd.DataFrame(columns = self.id_vars)
        self.score = pd.DataFrame()
        self.stats = pd.DataFrame()

    def _input_id_vars(self, df, **kwargs):
        for key in self.id_vars:
            df[key] = kwargs[key]
        return df

    def _thin(self, df, let_labels=False):
        if let_labels:
            to_drop = self.id_vars
        else:
            to_drop = self.fat_vars
        return df.drop(to_drop, axis=1)

    def proba_matrix(self):
        raise NotImplementedError

    def confusion_matrix(self):
        raise NotImplementedError

    def append_probas(self, probas,
                        true_labels, groups, **kwargs):

        local = pd.DataFrame(probas, columns = self.classes)
        local['true_label'] = true_labels
        local['group'] = groups
        local = self._input_id_vars(local, **kwargs)
        self.proba = self.proba.append(local)

    def append_weights(self, weights, **kwargs):
        index = pd.Index(self.classes)
        local = pd.DataFrame(weights,
                                index = self.classes,
                                columns = self.features)
        local = local.reset_index().melt(var_name='feature',
                                      id_vars=[self.classes.name])
        local = self._input_id_vars(local, **kwargs)
        self.weights = self.weights.append(local)

    def calculate_predictions(self):

        pred_max = lambda x: np.argmax(x)
        self.predictions['predictions_max'] = self._thin(self.proba).apply(pred_max, axis = 1)

        pred_mean = lambda x: np.sum(self._thin(self.proba).columns.values*x.values)
        self.predictions['predictions_mean'] = self._thin(self.proba).apply(pred_mean, axis = 1)

        self.predictions[self.fat_vars] = self.proba[self.fat_vars]

    def compute_score(self):
        for which in ['max', 'mean']:
            # def scoring(df):
            #     print(df.columns)
            scoring = lambda df: self.scoring_function(df['predictions_'+which],
                                                        df['true_label'])
            self.score['score_'+which] = self.predictions.groupby(self.id_vars).apply(scoring)

    def add_identifiers(self, **kwargs):
        for key, val in kwargs.items():
            self.id_vars.append(key)
            self.fat_vars.append(key)
            self.data[key] = val
            self.proba[key] = val
            self.weights[key] = val


    def compute_stats(self):
        fields=['score_max', 'score_mean']
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
        self.stats = self.score.groupby('trained_on').apply(Z_score)


def shuffle_val_predict(clf, dfs, names=None, X=None, y=None, group=None,
                         cv='sh', n_splits = 5,
                         train_size=.8,test_size=.2,
                         get_weights = True, score=pearson_score,
                         id_kwargs, **kwargs):

    """
    Trains in each dataset, always testing on both, to calculate statistics
        about generalization between datasets.

    Parameters
    ----------
    clf : sklearn classifier instance
        The model which will be fitted and used to predict labels

    dfs : DataFrame, or list of DataFrames
        The data holders

    names : list of strings, optional, default range
        The ID variables of each DataFrame
        If not given, will default to 0, 1, ..., len(dfs)-1

    X, y, group : indices [str[, str] ], optional
        The indices of each variable in the dataframes
        If not given, will default to
        X -> df columns
        y -> second index name
        group -> first index name

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
    results : Results_shuffle_val
        dataholder for this results.
        consists in many dataframes for ease of access
        proba, weights, predictions, scores, stats

    See also
    --------
        Results_shuffle_val


    """
    # Dealing with other optional variables
    if type(dfs) == pd.DataFrame:
        dfs = [dfs]
    if X == None:
        assert group == None and y == None
        X = dfs[0].columns
        y = dfs[0].index.names[1]
        group = dfs[0].index.names[0]
        dfs = [df.reset_index() for df in dfs]
    if names == None:
        names = np.arange(len(dfs))

    # Number of training and testing is defined by the smallest dataframe
    size_smallest = min([df[group].unique().shape[0] for df in dfs])
    n_train = int(size_smallest * train_size)
    n_test = int(size_smallest * test_size)

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
    res = Results_shuffle_val(n_splits=n_splits,
                    train_size = n_train,
                    test_size = n_test,
                    scoring_function = score,
                    classes = classes,
                    groups = dfs[0][group].unique(),
                    features = pd.Index(X, name='unit'),
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
                pred_groups = testdf[group].values[idx]
                res.append_probas(probas, true_labels,
                                  cv=i, trained_on=name,
                                  tested_on=testname,
                                  trained_here=trained_here,
                                  groups = pred_groups)



            if get_weights:
                res.append_weights(clf_local.coef_,
                                    cv=i, trained_on=name,
                                    tested_on= np.nan,
                                    trained_here= np.nan)

    res.add_identifiers(id_kwargs)
    res.calculate_predictions()
    res.compute_score()
    res.compute_stats()

    return res
