from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_validate

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
pearson = lambda t, p: pearsonr(t, p)[0]

rom sklearn.model_selection import GroupShuffleSplit

from spikelearn.data.selection import select, to_feature_array, frankenstein
from spikelearn.data import io, SHORTCUTS
from spikelearn.models import shuffle_val_predict
from sklearn.metrics import make_scorer, cohen_kappa_score


# Merging rats
DR = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['DRRD']]
EZ = [io.load(label, 'wide_smoothed') for label in SHORTCUTS['groups']['EZ']]

sp_pfc = frankenstein(DR, _min_duration=1.5, is_selected=True, is_tired=False, subset='full')
sp_pfc = sp_pfc[(sp_pfc.reset_index('time').time>=200).values & (sp_pfc.reset_index('time').time<1300).values]

ez_pfc = frankenstein(EZ, _min_duration=1.5, _min_quality=0, area='PFC', subset='full')
ez_pfc = ez_pfc[(ez_pfc.reset_index('time').time>=200).values & (ez_pfc.reset_index('time').time<1300).values]

ez_str = frankenstein(EZ, _min_duration=1.5, _min_quality=0, area='STR', subset='full')
ez_str = ez_str[(ez_str.reset_index('time').time>=200).values & (ez_str.reset_index('time').time<1300).values]
merged_rats = [sp_pfc, ez_pfc, ez_str]


# Calculating scores
ldares=pd.DataFrame()
scoring={'Explained Variance':'explained_variance', "Pearson's r":make_scorer(pearson), "Cohen's kappa":make_scorer(cohen_kappa_score)}
for rat_label, df in zip(['sp_pfc', 'ez_pfc', 'ez_str'],
                         [sp_pfc,   ez_pfc,   ez_str]):
    clf = LinearDiscriminantAnalysis()
    local = pd.DataFrame(cross_validate(clf, df.values, df.reset_index().time, df.reset_index().trial, 
                                        cv = GroupShuffleSplit(10), scoring=scoring, return_train_score=False))
    local['rat'] = rat_label    
    ldares = ldares.append(local)
    
res = ldares.filter(regex='test').rename(columns=lambda s: s[5:]).reset_index(drop=True).join(ldares.reset_index(drop=True).rat)

# Plotting
plt.figure(figsize=(6,6))
sns.barplot(data=res.melt('rat'), x='variable', y='value', hue='rat')

