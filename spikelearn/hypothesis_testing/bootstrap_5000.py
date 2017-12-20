import time
import sys
sys.path.append('/home/tevo/Documents/UFABC/Spikes')
import os
os.chdir('/home/tevo/Documents/UFABC/Spikes')
import pandas as pd

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()

from spikeHelper.loadSpike import Rat


n_bootstrap = 200
n_shuffles = 25
training_sizes = [2,7,50,200,496]



for rat_number in [8,9,10]:
    boot = Rat(rat_number)
    boot.selecTrials({'minDuration':1000})
    boot.selecTimes(tmin=200,tmax=700,z_transform=True)

    boot_results = pd.DataFrame()
    for train_size in training_sizes:
        clock = time.time()
        for boot_i in range(n_bootstrap):
            boot.shuffleTimes()
            results = boot.decode(clf, mode='fullShuffle', predict_or_proba = 'predict', test_size=.2,
                                    train_size=train_size, scoring=True, n_shuffles=n_shuffles,
                                    id_kwargs={'Bootstrap number':boot_i})
            boot_results = boot_results.append(results)
        print('Took %.2f seconds for train size %d'%(time.time()-clock, train_size))
    boot_results.to_csv('logistic_bootstrap_5000_%d.csv'%rat_number)

    #import pickle
    #pickle.dump(boot_results, open('logistic_bootstrap_5000_%d.pickle'%rat_number,'wb'))
