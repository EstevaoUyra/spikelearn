from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from pyriemann.utils.mean import mean_covariance
import sys
sys.path.append('/home/tevo/Documents/UFABC/Spikes')
sys.path.append('/home/tevo/Documents/UFABC/SingleUnit Spike Learning/src/models/')
import os
os.chdir('/home/tevo/Documents/UFABC/Spikes')
from spikeHelper.loadSpike import Rat


import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import pandas as pd
import numpy as np

# TODO 6 per rat
# All sigma vs non sigma (x2)
#   Beginning true training
#   After changepoint
#   Autoshape
# TODO Line of maximum
# Along any axis

savename = 'other_same_mean_size'

CPs = {7:157, 8:314, 9:176, 10:119}

def cross_temporal_generalization(df, other, method='pearson'):
    """

    """
    n_times = df.shape[1]
    fat_mat = df.join(other,lsuffix='f',rsuffix='s')
    pairwise_sim = fat_mat.corr()
    relevant_sim = pairwise_sim.iloc[:n_times,n_times:].values
    return np.nan_to_num(relevant_sim)

def similarity_matrix(cMat, n_splits = 200, method = 'pearson', **kwargs):
    """
    Calculates the similarity profile along the zeroth axis (axis=0) of a matrix,
    treating the axis 1 as features and axis 2 as trial/unit, marginalizing n_splits times
    each of two halfs along the axis 2


    Parameters
    -----
    cMat, ndarray of the shape (features, time, trials)
    n_splits, int

    Returns
    -----
    similarity, ndarray of shape (time, time)

    Notes
    -----
    Currently only accepts method pearson
    """
    sh = ShuffleSplit(n_splits=n_splits,**kwargs)

    trials = np.arange(cMat.shape[2])
    n_times = cMat.shape[1]
    all_sim = np.full((n_times,n_times,0),1)
    for f_half, s_half in sh.split(trials):
        mean_f, mean_s = pd.DataFrame(cMat[:,:,f_half].mean(axis=2)), pd.DataFrame(cMat[:,:,s_half].mean(axis=2))
        all_sim = np.dstack((cross_temporal_generalization(mean_f, mean_s),
                                all_sim))
    return all_sim

def plot_sim(df,title):
    fig = plt.figure(figsize=(10,12))
    df = df[df['label']!= '32 Neurons']
    for j,sigma in enumerate(df.sigma.unique()):
        for i,label in enumerate(['Autoshape','Early','AfterCP']):
            ax = plt.subplot2grid((3,2),(i,j),fig=fig)
            sns.heatmap(df.loc[(df.label==label).values &  (df.sigma==sigma).values].similarity.values[0],ax=ax)
            plt.title('%s, sigma %s'%(label,str(sigma)))
            plt.xticks(np.arange(12),np.arange(0,1.5,.12),rotation=30); plt.xlabel('Time from nosepoke (s)')
            plt.yticks(np.arange(12),np.arange(0,1.5,.12),rotation=30); plt.ylabel('Time from nosepoke (s)')
        plt.tight_layout()
        plt.suptitle(title,x=.5,y=1.02,fontsize=20)
    return fig

if __name__=='__main__':
    all_results = pd.DataFrame()
    for rat_number in [7,8,9,10]:
        for sigma in [None, 100]:
            auto = Rat(rat_number, sigma=sigma, binSize=120, label='autoshape', method = 32)
            true32 = Rat(rat_number, sigma=sigma, binSize=120, method = 32)
            trueLate = Rat(rat_number, sigma=sigma, binSize=120, method = 32)
            trueEarly = Rat(rat_number, sigma=sigma, binSize=120, method = 32)

            auto.selecTrials({'minDuration':1300,'maxDuration':1700},zmax=3)
            true32.selecTrials({'minDuration':1300,'maxDuration':1700},zmax=3)
            trueLate.selecTrials({'minDuration':1300,'maxDuration':1700, 'trialMin': CPs[rat_number]},zmax=3)
            trueEarly.selecTrials({'minDuration':1300,'maxDuration':1700, 'trialMax':CPs[rat_number]},zmax=3)

            for r,label in zip([auto, true32, trueLate, trueEarly],['Autoshape','32 Neurons', 'AfterCP', 'Early']):
                r.selecTimes(-30,1300,False)
                sim_mat = similarity_matrix(r.cubicNeuronTimeTrial(),train_size=5,test_size=5)
                print(sim_mat.swapaxes(0,2).shape,np.isnan(sim_mat).sum())
                results = pd.DataFrame({'Rat':rat_number, 'label':label, 'sigma': sigma,
                #'similarity':[mean_covariance(sim_mat.swapaxes(0,2))]})
                'similarity':[sim_mat.mean(axis=2)]})
                all_results = all_results.append(results)

    all_results.loc[all_results.sigma.isnull(),'sigma'] = 'None'
    pickle.dump(all_results, open('similarity_measures_samesize.pickle','wb'))

    for rat_number in [7,8,9,10]:
        fig = plot_sim(all_results.groupby('Rat').get_group(rat_number),'Rat %d'%rat_number)
        plt.savefig('Figures/similarities_%s_%s'%(savename,rat_number), dpi=400)
