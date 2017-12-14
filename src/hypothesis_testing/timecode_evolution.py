from sklearn.linear_model import LogisticRegression
from spikeHelper import Rat



for rat_number in [7,8,9,10]:
    true = Rat(rat_number)
    true.selecTrials('minDuration':1000)
    true.selecTimes(tmin=200,tmax=700)

    boot = Rat(rat_number)
    boot.selecTrials('minDuration':1000)
    boot.selecTimes(tmin=200,tmax=700)

    results = true.decode(clf, mode='init', predict_or_proba = 'proba', train_size=200)
    for boot_i in n_bootstrap:
        boot.shuffleTimes()
        results = boot.decode(clf, mode='init', predict_or_proba = 'proba', train_size=200)
