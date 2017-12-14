from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

import time
import pickle
import sys
sys.path.append('/home/tevo/Documents/UFABC/Spikes')
import os
os.chdir('/home/tevo/Documents/UFABC/Spikes')

import matplotlib.pyplot as plt
from skopt import load
from skopt.plots import plot_objective

import numpy as np
import pandas as pd
from spikeHelper.loadSpike import Rat
from spikeHelper.makeClassifierList import makeClassifierList
classifiers = makeClassifierList()

## Simulation parameters

saveDir = 'predictions_7bins'; os.makedirs(saveDir)
# For decoding
min_trial_duration = 1000
tmin, tmax = 200, 700
sim_test_size = .2 ; max_train_size = .8 ;
# For statistics
n_bootstrap = 5; n_bootstrap_shuffles = 5;
n_shuffles_true = 25;
number_of_training_sizes = 10

assert sim_test_size + max_train_size <= 1

## Functions
def number_of_trials_for_each_rat(restrictions):
    number_of_trials_for_each = []
    for rat_number in [7,8,9,10]:
        true_times_Rat = Rat(rat_number)
        true_times_Rat.selecTrials({'minDuration':min_trial_duration})
        number_of_trials_for_each.append(true_times_Rat.trialsToUse.sum())
    return number_of_trials_for_each

def load_parameters_from_tuning(filename):
    params = pd.read_csv(filename, index_col=0)
    params['parameters'] = params['parameters'].apply(string_to_dict)
    return params

def string_to_dict(dictstring):
    eachPair = dictstring.replace('{','').replace('}','').replace('"','').replace("'",'').strip().split(',')
    all_keys_and_values = [keyvalue.split(':') for keyvalue in eachPair]
    final_dict = {}
    for pair in all_keys_and_values:
        try:
            if int(pair[1]) == float(pair[1]):
                final_dict[pair[0].strip()] = int(pair[1])
            else:
                final_dict[pair[0].strip()] = float(pair[1])
        except:
            try:
                final_dict[pair[0].strip()] = float(pair[1])
            except:
                final_dict[pair[0].strip()] = pair[1].strip()
    return final_dict


## Simulation start
pre_calculated_hyperparameters = load_parameters_from_tuning('hyperparameter_tuning_1811/best_parameters_from_all_optimizations.csv')

max_shareable_trial_number = np.min( number_of_trials_for_each_rat(restrictions = {'minDuration':min_trial_duration}) )
training_sizes = np.unique( np.logspace( 0, np.log10(.8*max_shareable_trial_number), number_of_training_sizes).astype(int) )
print('Training sizes are %s'%training_sizes)

for rat_number in [7,8,9,10]:
    print('Calculating for rat %d'%rat_number)

    full_shuffle_true_predictions = pd.DataFrame(columns = ['trial', 'shuffle', 'predictions','true',
    'classifier','rat','train size'])
    full_shuffle_bootstrap = pd.DataFrame(columns = ['trial', 'shuffle', 'predictions','true',
    'classifier','rat','train size', 'n boot'])

    init_true_predictions = pd.DataFrame(columns = ['trial', 'shuffle', 'predictions','true',
    'classifier','rat','train size'])
    init_bootstrap = pd.DataFrame(columns = ['trial', 'shuffle', 'predictions','true',
    'classifier','rat','train size','n boot'])

    true_times_Rat = Rat(rat_number)
    true_times_Rat.selecTrials({'minDuration':min_trial_duration})
    true_times_Rat.selecTimes(tmin,tmax, z_transform=True)

    bootstrap_Rat = Rat(rat_number)
    bootstrap_Rat.selecTrials({'minDuration':min_trial_duration})
    bootstrap_Rat.selecTimes(tmin,tmax, z_transform=True)

    for classifier in classifiers:
        print('    Using classifier %s'%classifier['name'])

        if classifier['name'] in pre_calculated_hyperparameters['classifier'].values:
            params = pre_calculated_hyperparameters[ pre_calculated_hyperparameters['classifier']==classifier['name'] ] ['parameters'][rat_number]
            print('     %s'%dict(params))
            clf = classifier['func'](**params)
        else:
            clf = classifier['func']()


        for train_size in training_sizes:
            try:
                print('        Using train size %d'%train_size)
                fullShuffle_time = time.time()
                local_results = true_times_Rat.decode(clf, 'fullShuffle', train_size=train_size, test_size=sim_test_size, n_shuffles=n_shuffles_true)
                local_results['time'] = time.time() - fullShuffle_time;
                local_results['rat'] = rat_number;
                local_results['classifier'] = classifier['name'];
                local_results['train size'] = train_size;
                full_shuffle_true_predictions = full_shuffle_true_predictions.append(local_results)


                local_beg_results = true_times_Rat.decode(clf, 'fixInit_shuffleEnd', train_size=train_size,
                                        test_size=sim_test_size, n_shuffles=n_shuffles_true,init_size=30)
                local_beg_results['rat'] = rat_number;
                local_beg_results['classifier'] = classifier['name'];
                local_beg_results['train size'] = train_size;
                init_true_predictions = init_true_predictions.append(local_beg_results)

                begin_boot_time = time.time()
                for bootstrap in range(n_bootstrap):
                    bootstrap_Rat.shuffleTimes()
                    local_results = bootstrap_Rat.decode(clf, 'fullShuffle', train_size=train_size, test_size=sim_test_size, n_shuffles=n_bootstrap_shuffles)
                    local_results['rat'] = rat_number
                    local_results['classifier'] = classifier['name']
                    local_results['train size'] = train_size
                    local_results['n boot'] = bootstrap
                    full_shuffle_bootstrap = full_shuffle_bootstrap.append(local_results)

                    local_beg_results = bootstrap_Rat.decode(clf, 'fixInit_shuffleEnd', train_size=train_size,
                                        test_size=sim_test_size, n_shuffles=n_bootstrap_shuffles,init_size=30)
                    local_beg_results['rat'] = rat_number
                    local_beg_results['classifier'] = classifier['name']
                    local_beg_results['train size'] = train_size
                    local_beg_results['n boot'] = bootstrap
                    init_bootstrap = init_bootstrap.append(local_beg_results)
                print('            Took %.1f seconds for bootstraping'%(time.time()-begin_boot_time))
            except Exception as e:
                print(e)

    full_shuffle_true_predictions.to_csv('%s/decoding_alltrials_results_for_training_sizes_rat_%d.csv'%(saveDir,rat_number))
    full_shuffle_bootstrap.to_csv('%s/decoding_alltrials_bootstrap_for_training_sizes_rat_%d.csv'%(saveDir,rat_number))
    init_true_predictions.to_csv('%s/from_beginning_results_for_training_sizes_rat_%d.csv'%(saveDir,rat_number))
    init_bootstrap.to_csv('%s/from_beginning_bootstrap_for_training_sizes_rat_%d.csv'%(saveDir,rat_number))

    pickle.dump(full_shuffle_true_predictions,open('%s/decoding_alltrials_results_for_training_sizes_rat_%d.pickle'%(saveDir,rat_number),'wb') )
    pickle.dump(full_shuffle_bootstrap,open('%s/decoding_alltrials_bootstrap_for_training_sizes_rat_%d.pickle'%(saveDir,rat_number),'wb'))
    pickle.dump(init_true_predictions,open('%s/from_beginning_results_for_training_sizes_rat_%d.pickle'%(saveDir,rat_number),'wb'))
    pickle.dump(init_bootstrap,open('%s/from_beginning_bootstrap_for_training_sizes_rat_%d.pickle'%(saveDir,rat_number),'wb'))
