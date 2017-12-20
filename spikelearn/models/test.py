from tune_hyperparameters import *
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os
os.chdir('/home/tevo/Documents/UFABC/Spikes')

from spikeHelper.loadSpike import Rat


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
                print(pair[1])
                final_dict[pair[0].strip()] = pair[1].strip()
    return final_dict

all_params = load_parameters_from_tuning('hyperparameter_tuning_1811/best_parameters_from_all_optimizations.csv')
