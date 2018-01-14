import sys
sys.path.append('/home/tevo/Documents/UFABC/Spikes')
sys.path.append('/home/tevo/Documents/UFABC/SingleUnit Spike Learning/src/models/')
import os
os.chdir('/home/tevo/Documents/UFABC/Spikes')
from spikeHelper.loadSpike import Rat
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from spikeHelper.loadSpike import Rat

n_shuffles_for_each = 100
n_bootstrap_shuffles = 0
CPs = {7:157, 8:314, 9:176, 10:119} # Gallistel
iti_best = {7:800, 8:550, 9:600, 10:900}
binSize = 120
tmin = 200; tmax = 1200


true_codes = pd.DataFrame()
boot_codes = pd.DataFrame()
for train_size, test_size in [(.9,.1)]:# zip([27,.9],[1,.1]):
    for sigma in [None]:
        for rat_number in [7,8,9,10]:
            auto = Rat(rat_number, label='autoshape', sigma=sigma, binSize=binSize)
            boot = Rat(rat_number, label='autoshape', sigma=sigma, binSize=binSize)
            early = Rat(rat_number, sigma=sigma, binSize=binSize)#,method=32)
            after = Rat(rat_number, sigma=sigma, binSize=binSize)#,method=32)label='32%d'

            auto.selecTrials({'minDuration':1500}); auto.selecTimes(tmin=tmin,tmax=tmax)
            boot.selecTrials({'minDuration':1500}); boot.selecTimes(tmin=tmin,tmax=tmax)
            early.selecTrials({'minDuration':1500,'ntrials':100}); early.selecTimes(tmin=tmin,tmax=tmax)
            after.selecTrials({'minDuration':1500,'ntrials':100,'trialMin':CPs[rat_number],'trialMax':iti_best[rat_number]}); after.selecTimes(tmin=tmin,tmax=tmax)


            for r, id in zip([auto,early,after], ['autoshape', 'begin session', 'after changepoint']):
                true_codes = true_codes.append(
                        r.decode(LogisticRegression(), mode='fullShuffle', train_size=train_size, test_size=test_size, n_shuffles=n_shuffles_for_each,
                            predict_or_proba='predict',scoring=True, pca=4,
                            id_kwargs={'results':'predictions', 'data from':id,'smoothing': True if sigma == 100 else False,'train_size':train_size}) )
                true_codes = true_codes.append(
                        r.decode(LogisticRegression(), mode='fullShuffle', train_size=train_size, test_size=test_size, n_shuffles=n_shuffles_for_each,
                            predict_or_proba='proba',scoring=False, pca=4,
                            id_kwargs={'results':'probability', 'data from':id,'smoothing': True if sigma == 100 else False,'train_size':train_size}) )

            for n_boot in range(n_bootstrap_shuffles):
                boot.shuffleTimes()
                boot_codes = boot_codes.append(
                        boot.decode(LogisticRegression(), mode='fullShuffle', train_size=27, test_size=1,
                        n_shuffles=n_shuffles_for_each, predict_or_proba='predict',scoring=True,
                            id_kwargs={'results':'predictions', 'data from':True}) )
                boot_codes = boot_codes.append(
                        boot.decode(LogisticRegression(), mode='fullShuffle', train_size=27, test_size=1,
                        n_shuffles=n_shuffles_for_each, predict_or_proba='proba',scoring=False,
                            id_kwargs={'results':'probability', 'data from':True}) )

pickle.dump(true_codes, open('Autoshape_comparison_100_pca.pickle','wb'))
pickle.dump(boot_codes, open('Autoshape_bootstrap_100_pca.pickle','wb'))
