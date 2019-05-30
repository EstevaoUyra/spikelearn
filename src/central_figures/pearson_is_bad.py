import pandas as pd
from sklearn.model_selection import cross_validate, GroupShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_validate

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
pearson = lambda t, p: pearsonr(t, p)[0]

from sklearn.model_selection import GroupShuffleSplit

from spikelearn.data.selection import select, to_feature_array, frankenstein
from spikelearn.data import io, SHORTCUTS
from spikelearn.models import shuffle_val_predict
from sklearn.metrics import make_scorer, cohen_kappa_score
from sklearn.linear_model import BayesianRidge

scoring={'Explained variance':'explained_variance', "Pearson's r":make_scorer(pearson)}

def analysis(df, clf, n_splits=2)
    res = cross_validate(clf, df.values, df.reset_index().time, df.reset_index().trial, 
                         cv = GroupShuffleSplit(n_splits), scoring=scoring, return_train_score=False)

    res = pd.DataFrame(res).filter(regex='test').melt()
    res.variable = res.variable.apply(lambda s: s[5:])
    
    

    
bootres = pickle.load(open('data/results/central_figures/pearson_vs_var.pickle', 'rb'))    

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(6,3), dpi=200)

sns.boxplot(x='model', y='value', data=bootres[bootres.variable=="Pearson's r"], ax=ax[0])
sns.boxplot('model', 'value', data=bootres[bootres.variable=="Explained variance"], ax=ax[1])

ax[0].set_ylabel("Pearson's r")
ax[1].set_ylabel("Explained variance")
plt.tight_layout()