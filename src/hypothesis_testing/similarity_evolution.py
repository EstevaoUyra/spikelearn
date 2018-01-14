from pyriemann.utils.distance import distance
from similarity import similarity_matrix
import sys
sys.path.append('/home/tevo/Documents/UFABC/Spikes')
sys.path.append('/home/tevo/Documents/UFABC/SingleUnit Spike Learning/src/models/')
import os
os.chdir('/home/tevo/Documents/UFABC/Spikes')
from spikeHelper.loadSpike import Rat
import pandas as pd
import numpy as np
from numpy.linalg import eig, norm
ds = ['riemann', 'euclid', 'logdet', 'kullback', 'kullback_sym']
import pickle
from scipy.stats import pearsonr
from scipy.spatial.distance import directed_hausdorff

##TODO group trials by similarity and compute tgen matrices
 # group MATRICES by similarity and compute
 # k-means trial matrices

## Measure distance between each single-trial generalization matrix and each one of the others
# Bonus: get 2-trial and 5-trial mean matrices
iti_best = {7:400, 8:550, 9:300, 10:400}
n_trials_for_mean_sim = 20
all_res = pd.DataFrame()

templates = pd.DataFrame()
for rat_number in [7,8,9,10]:
    r = Rat(rat_number, sigma=None, binSize=120)
    #({'minDuration':1300,'maxDuration':1700},zmax=3)
    r.selecTrials({'minDuration':1300,'maxDuration':1700, 'trialMax':iti_best[rat_number]})
    r.selecTimes(0,1300)
    early_sim = similarity_matrix(r.cubicNeuronTimeTrial()[:,:,:n_trials_for_mean_sim],
     n_splits = 100, method = 'pearson').mean(axis=2)
    late_sim = similarity_matrix(r.cubicNeuronTimeTrial()[:,:,-n_trials_for_mean_sim:],
     n_splits = 100, method = 'pearson').mean(axis=2)
    templates=templates.append(pd.DataFrame({'early':[early_sim],'late':[late_sim],'rat':rat_number}))

    for trial in np.unique(r.trial):
        one_trial_activity = r.X[r.trial==trial,:].transpose()
        one_trial_gen = np.nan_to_num(pd.DataFrame(one_trial_activity).corr().values)
        one_trial_res = {#'to early':norm(one_trial_gen - early_sim),
                         #'to late':norm(one_trial_gen - late_sim),
                         'to early':pearsonr(one_trial_gen.ravel(), early_sim.ravel())[0],
                         'to late':pearsonr(one_trial_gen.ravel(), late_sim.ravel())[0],

                         'trial': trial, 'rat_number':rat_number,
                         'matrix': [one_trial_gen]}
        all_res = all_res.append(pd.DataFrame(one_trial_res))


pickle.dump(templates, open('similarity_templates_cp_corr_smoothNO.pickle','wb'))
pickle.dump(all_res, open('similarity_results_cp_corr_smoothNO.pickle','wb'))

# s = all_res.drop(['rat_number','matrix'],axis=1).set_index('trial')
# (s['to early']-s['to late']).plot()
# plt.fill_betweenx([-1,1],s.index[n_trials_for_mean_sim],s.index[0],color='g',alpha=.5)
# plt.fill_betweenx([-1,1],s.index[-1],s.index[-n_trials_for_mean_sim],color='r',alpha=.5)
# plt.show()
