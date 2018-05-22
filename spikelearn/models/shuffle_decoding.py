import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, cross_val_predict, GroupKFold
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from numpy.random import permutation
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

import pickle
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
                    test_size, scoring_metric,
                    classes, groups, features, id_vars):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.scoring_metric = scoring_metric


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

    # Internal
    def scoring_function(self, true, pred):
        if self.scoring_metric == 'pearson':
            return pearsonr(true, pred)[0]
        else:
            raise NotImplementedError

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

    # shuffle_val_predict usage
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

        pred_max = lambda x: x.idxmax()
        self.predictions['predictions_max'] = self._thin(self.proba).apply(pred_max, axis = 1)

        pred_mean = lambda x: np.sum(self._thin(self.proba).columns.values*x.values)
        self.predictions['predictions_mean'] = self._thin(self.proba).apply(pred_mean, axis = 1)

        self.predictions[self.fat_vars] = self.proba[self.fat_vars]

    def compute_score(self):
        for which in ['max', 'mean']:
            scoring = lambda df: self.scoring_function(df['true_label'], df['predictions_'+which])
            self.score['score_'+which] = self.predictions.groupby(self.id_vars).apply(scoring)
        self.score = self.score.reset_index()

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

    # Outside usage
    def proba_matrix(self, plot=True, grouping=None, **kwargs):
        if grouping is not None:
            df = self.proba.groupby(grouping[0]).get_group(grouping[1])
        else:
            df = self.proba
        df = df.groupby('true_label').mean().drop('group', axis=1)
        if plot:
            sns.heatmap(df, **kwargs)
            plt.title('Mean probability associated with labels')
            plt.xlabel('Possible labels'); plt.ylabel('True label')
        return df

    def confusion_matrix(self, plot=True, grouping=None, which='max', **kwargs):
        if grouping is not None:
            df = self.predictions.groupby(grouping[0]).get_group(grouping[1])
        else:
            df = self.predictions
        mat = confusion_matrix(df.true_label, df['predictions_'+which])
        if plot:
            sns.heatmap(mat, **kwargs)
            plt.title('Confusion matrix');
            plt.xlabel('Predicted label'); plt.ylabel('True label')
        return mat

    def save(self, filename):
        pickle.dump(self, open(filename, 'wb'))


def shuffle_val_predict(clf, dfs, names=None, X=None, y=None, group=None,
                         cv='sh', n_splits = 5, feature_scaling=None,
                         train_size=.8, test_size=.2, cross_prediction=False,
                         balance_feature_number = True,
                         get_weights = False, score='pearson',
                         id_kwargs=None, verbose=0, **kwargs):

    """
    Trains in each dataset, possibly testing on both, to calculate statistics
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
        See notes

    cv : str or callable, default 'sh'
        The splitting method to use.

    n_splits : int
        Number of splits to be done.

    feature_scaling : string
        The kind of scaling to apply to features.
        Implemented via pipeline (fitted only during training)
        One of 'standard', 'minmax', 'robust'.

    get_weights : bool
        Whether to save and return the weights of each model

    score : callable
        function( true_label, pred_label ) -> number
        Defaults to pearson's correlation

    Keyword Arguments
    -----------------
    Extra kwargs will be passed on to the cv function

    Notes
    -----
    While the number of features can differ in some cases,
    the variables y and group MUST be the same for all dfs

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
    if names is None:
        names = np.arange(len(dfs))

    if type(dfs) == pd.DataFrame:
        dfs, names = [dfs], [names]
    if X is None:
        assert group == None and y == None
        if cross_prediction or get_weights:
            assert  len(np.unique([len(df.columns) for df in dfs] )) == 1
            assert all( [(df.columns == dfs[0].columns).all() for df in dfs] )
            X = dfs[0].columns
        else:
            X = {name:df.columns for df, name in zip(dfs,names)}
        y = dfs[0].index.names[1]
        group = dfs[0].index.names[0]

        dfs = [df.reset_index() for df in dfs]

    # Number of training and testing is defined by the smallest dataframe
    size_smallest = min([df[group].unique().shape[0] for df in dfs])
    n_train = int(size_smallest * train_size)
    n_test = int(size_smallest * test_size)

    # Number of features is also defined by the smallest
    if balance_feature_number:
        assert not cross_prediction
        n_feats = min([df.shape[1] for df in dfs]) - 2


    # Method of cross-validation
    if cv == 'kf':
        sh = GroupKFold(n_splits=n_splits, **kwargs)
    elif cv == 'sh':
        sh = GroupShuffleSplit(n_splits=n_splits,
                                train_size=n_train, test_size=n_test, **kwargs)
    elif isinstance(cv, object):
        sh=cv

    # Scaling
    if feature_scaling is None:
        pass
    elif feature_scaling == 'minmax':
        clf = Pipeline([('minmaxscaler', MinMaxScaler((-1,1))),
                        ('classifier', clf)])
    elif feature_scaling == 'standard':
        clf = Pipeline([('standardscaler', StandardScaler()),
                        ('classifier', clf)])
    elif feature_scaling == 'robust':
        clf = Pipeline([('robustscaler', RobustScaler()),
                        ('classifier', clf)])
    else:
        raise ValueError('%s scaling is not accepted.\n Lookup the documentation for accepted scalings'%feature_scaling)

    # Define the results format
    classes = pd.Index( np.unique(dfs[0][y]), name=y)
    id_vars = ['cv',
               'trained_on',
               'tested_on',
               'trained_here',
               'n_features']
    res = Results_shuffle_val(n_splits=n_splits,
                    train_size = n_train,
                    test_size = n_test,
                    scoring_metric = score,
                    classes = classes,
                    groups = dfs[0][group].unique(),
                    features = pd.Index(X, name='unit'),
                    id_vars = id_vars)

    # Make the cross validation on each dataset
    for traindf, name in zip(dfs, names):
        if verbose>0: print('\n-------\nDataset %s'%name)
        for i, (train_idx, test_idx) in enumerate(sh.split(traindf[y], traindf[y], traindf[group])):
            if verbose>1:print(i,end=', ')

            if cross_prediction:
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
                                      groups = pred_groups,
                                      n_features = len(X))
            else:
                if get_weights:
                    shufX = X
                elif balance_feature_number:
                    shufX = np.random.permutation(X[name])[:n_feats]
                else:
                    shufX = X[name]

                clf_local = clone(clf)
                clf_local.fit( traindf[shufX].values[train_idx], traindf[y].values[train_idx] )

                if verbose >= 4:
                    print("Has %d features"%len(X[name]),end=', ')
                    print('now using %s'%shufX)

                trained_here = True
                testdf, testname, idx = traindf, name, test_idx
                probas = clf_local.predict_proba(testdf[shufX].values[idx])
                true_labels = testdf[y].values[idx]
                pred_groups = testdf[group].values[idx]
                res.append_probas(probas, true_labels,
                                  cv=i, trained_on=name,
                                  tested_on=testname,
                                  trained_here=trained_here,
                                  groups = pred_groups,
                                  n_features = len(shufX))


            if get_weights:
                res.append_weights(clf_local.coef_,
                                    cv=i, trained_on=name,
                                    tested_on= np.nan,
                                    trained_here= np.nan)
        if id_kwargs is not None:
            res.add_identifiers(id_kwargs)

    res.calculate_predictions()
    res.compute_score()
    #res.compute_stats()
    return res
