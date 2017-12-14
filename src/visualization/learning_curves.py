import pickle
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score,explained_variance_score,mean_squared_error,accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
prediction_dir = '/home/tevo/Documents/UFABC/Spikes/predictions_xgboost'
sys.path.append('../')
sys.path.append('/home/tevo/Documents/UFABC/SingleUnit Spike Learning/src/')

from models.makeClassifierList import makeClassifierList

def agg_and_score(df, mode):
    if mode == 'kappa':
        scoring = lambda x,y: cohen_kappa_score(x,y,weights='quadratic')
    elif mode == 'corr':
        scoring = lambda x,y: pearsonr(x,y)[0]
    elif mode == 'explained variance':
        scoring = explained_variance_score
    elif mode == 'mse':
        scoring = mean_squared_error
    elif mode == 'acc':
        scoring = accuracy_score
    return scoring(np.hstack(df.true),np.hstack(df.predictions))


def make_one_score_per_field(df, field='shuffle',time=False):
    score_df = pd.DataFrame(df.groupby(['classifier','train_size', field]).apply(agg_and_score,'corr'))
    score_df.columns=['Pearson r']
    score_df['Cohen $\kappa$'] = df.groupby(['classifier','train_size', field]).apply(agg_and_score, 'kappa')
    score_df['Accuracy'] = df.groupby(['classifier','train_size', field]).apply(agg_and_score,'acc')
    score_df['Explained variance'] = df.groupby(['classifier','train_size', 'shuffle']).apply(agg_and_score,'explained variance')
    if time:
        score_df['Time'] = df.groupby(['classifier','train_size', 'shuffle']).apply(lambda x: x['time'].values[0])
    #score_df['Mean Squared Error'] = df.groupby(['classifier','train_size', 'shuffle']).apply(agg_and_score,'mse')

    return score_df#corr_df.merge(kappa_df,left_index=True,right_index=True)



def get_predictions(rat_number, one_score_per_field=None,bootstrap=False,time=False, mode='fullShuffle'):
    bootstrap_flag = 'bootstrap' if bootstrap else 'results'

    if mode in ['fullShuffle', 'full']:
        mode_flag = 'decoding_alltrials'
    elif mode in ['init','fix_init']:
        mode_flag = 'from_beginning'
    else:
        raise ValueError('The mode %s is unknown'%mode)

    filename = '%s/%s_%s_for_training_sizes_rat_%d.pickle'%(prediction_dir, mode_flag, bootstrap_flag, rat_number)
    df = pickle.load(open(filename,'rb'))
    if one_score_per_field is not None:
        df = make_one_score_per_field(df, field=one_score_per_field,time=time)
        df['rat'] = rat_number
        return df
    else:
        df['rat'] = rat_number
        return df


def clf_learning_curve(score_per_shuffle, classifier, ci=False, scores = 'all',ax=None,inset=False,explained_variance=False,**kwargs):
    if scores == 'all':
        scores = score_per_shuffle.columns
    else:
        columns=[col for col in score_per_shuffle.columns if col in scores ]

    if ax is None:
        ax = plt.subplot(1,1,1)


    clf_score = score_per_shuffle.groupby('classifier').get_group(classifier).reset_index()
    clf_score = clf_score.melt(id_vars=['classifier','train_size', 'shuffle','rat'], var_name='score',
                                value_name='performance')
    clf_score['shuffle'] = clf_score['shuffle'].apply(str) + clf_score['rat'].apply(str)


    ax = sns.tsplot(clf_score[clf_score['score']!='Explained variance'], time='train_size',condition='score',value='performance',unit='shuffle', ax=ax,ci=ci,**kwargs)
    plt.legend(loc='lower right')
    plt.xlabel('Train size (trials used)')
    plt.xlim([-5,505])
    plt.ylim([-.02,.4])
    if explained_variance:
        ax2 = ax.twinx()
        ax2 = sns.tsplot(clf_score[clf_score['score']=='Explained variance'], time='train_size',condition='score',value='performance',unit='shuffle', ax=ax2,ci=ci,color='r',**kwargs);ax2.set_ylabel('Explained variance')
        plt.legend(loc='upper right')
    return ax


def allRats_clf_learning_curve(classifier, ci=False, scores = 'all',ax=None, bootstrap=False,inset=False,**kwargs):
    styles = ['-','--','-.',':']
    #fig = plt.figure(figsize=(20,20))


    for i, rat_number in enumerate([7,8,9,10]):
        score_per_shuffle = get_predictions(rat_number,one_score_per_field='shuffle', bootstrap=bootstrap)
        ax = clf_learning_curve(score_per_shuffle,classifier,ci=ci,linestyle=styles[i],ax=ax,legend= i==0,**kwargs)
    plt.title(classifier+' learning curve')
    sns.despine(left=True,bottom=True)
    return plt.gcf(), ax

def mean_learning_curve(classifier, ci=False, scores = 'all',ax=None,inset=False, bootstrap=False,**kwargs):
    #fig = plt.figure(figsize=(20,20))

    r = pd.concat([get_predictions(i,bootstrap=bootstrap,one_score_per_field='shuffle') for i in [7,8,9,10]])
    ax = clf_learning_curve(r,classifier,ci=ci,scores=scores,ax=ax,inset=inset,**kwargs)
    sns.despine(left=True,bottom=True)
    plt.legend(loc='lower right')
    plt.title(classifier+' learning curve')
    return plt.gcf(), ax

if __name__=='__main__':
    for classifier in makeClassifierList()[-1:]:
        print(classifier['name'])

        sns.set_style('whitegrid',{ 'axes.labelcolor': 'w', 'text.color': 'w', 'ytick.color': 'w', 'xtick.color': 'w'})
        sns.set_palette('deep')
        fig, ax = mean_learning_curve(classifier['name'],ci=95,linewidth=2,marker='.',markersize=6)
        plt.rc("axes.spines", top=False, right=False,left=False,bottom=False)
        plt.legend(loc='upper left')
        plt.savefig('../../reports/figures/Learning curves/mean_'+ classifier['name'].replace(' ','_') +'_learningcurve.png',
                    bbox_inches='tight',transparent=True,dpi=1000)
        plt.close(fig)

        fig, ax = allRats_clf_learning_curve(classifier['name'],ci=95,linewidth=2,marker='.',markersize=6)
        plt.rc("axes.spines", top=False, right=False,left=False,bottom=False)
        plt.legend(loc='upper left')
        plt.savefig('../../reports/figures/Learning curves/each_'+ classifier['name'].replace(' ','_') +'_learningcurve.png',
                    transparent=True,dpi=1000)
        plt.close(fig)
