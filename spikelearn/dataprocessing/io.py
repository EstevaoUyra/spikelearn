"""
Input (loading) and output (saving) functions.
Lower level functions, usually called from the api
"""

import pandas as pd
import json
from scipy.io import loadmat
from datetime import datetime

def spikes_behavior_from_mat(filepath):
    """
    Loads a mat-file into two DataFrames

    Parameters
    ----------

    Returns
    -------
    spikes : DataFrame (n_units, 3)
        Contains three ndarray fields, indexed by the unit (neuron).
        Each ndarray has the form (n_spikes_i,) being different for each row.
        'times' holds the absolute times of each spike from the session begin.
        'trial' holds the trial number of each corresponding spike from times.
        'trial_time' has the relative time of each spike from trial onset.

    behavior : DataFrame (n_trials, 3)
        Contains five number fields of trial attributes.
        'onset' is the time of trial beginning
        'offset' is the end of the trial
        'duration' is equal to offset - onset
    """
    #TODO decide if it would make sense to keep sortIdx and sortLabel
    data = loadmat(filepath)

    spikes = data['dados'][0,0][1]
    behavior = data['dados'][0,0][0]

    spikes = pd.DataFrame([[ spikes[0,i][0][:,0], spikes[0,i][0][:,1]] for i in range(spikes.shape[1]) if spikes[0,i][0].shape[1]==2], columns=['times','trial'])

    behavior = pd.DataFrame(np.transpose(behavior[0,0][0]),   columns=['one','onset','offset','zero', 'duration', 'sortIdx', 'sortLabel']).drop(['one', 'zero', 'sortIdx', 'sortLabel'], axis=1)

    # Calculate relative spike time
    spikes['trial_time'] = pd.DataFrame(np.transpose([spikes.times[i] - behavior.iloc[spikes.trial[i]-1].onset.as_matrix() for i in range(spikes.shape[0])]))

    return spikes, behavior

class DataShortcut():
    """
    Loads and saves data into DataFrame, getting shortcuts from
    a pre-edited JSON file shortcuts.json

    Parameters
    ----------
    base_label : string
    Base dataset label. If not contained in shortcuts.json, then the filepath
    parameter is needed.

    dataset_name : string or list of strings
    Name of the specific dataset to load. If list of strings,
        enters the hierarchy from left to right, at last loading
        dataset[-1].
        Listing is only needed when the dataset name is repeated in shortcuts

    filepath : string, path-to-file, optional
    Name of the path of datafile. If provided jointly with label, the
    overwrite flag must be explicitly set to True.

    basepath : string, optional
    full path of some folder in filepath hierarchy
    serves to make shortcuts.json identation less polluted

    overwrite : bool, optional, default False
    Which to overwrite label with filepath

    annotations : str, optional
    Observations to add to the file description

    Keyword Arguments
    -----------------
    Add any number of identifier variables with keyword arguments,
    which must be numbers or strings.

    Notes
    -----
    Uses the datafiles.json to search and load data using simple
    name labels.
    """
    def __init__(self,base_label, dataset_name, filepath=None, basepath=None, overwrite=False, data = None, annotations='', **kwargs):
        self.base_label = label
        self.dataset_name = dataset
        if filepath is None:
            self.filepath = self._get_filepath
        else:
            self.filepath = filepath

        self.overwrite = overwrite

        # Make sure inputs are consistent
        self._assertions()

        # Load the shortcuts
        self.shortcuts = json.load(open('shortcuts.json','r') )
        if label in shortcuts:
            self._soi_fullpath = recursive_full_name_recovery(self.shortcuts[label])
            self._dataset_exists = True
        else:
            self._dataset_exists = False
            assert basepath is not None
            self.basepath = basepath

        # Load data at last
        if data is None:
            self.data = self._load()
            self._functioning = "loader"
        else:
            self.data = data
            self._functioning = "creator"

        # Add identifiers
        for key in kwargs:
            if key in self.columns:
                raise ValueError("Identifier {} already exists")
            else:
                self.data[key] = kwargs[key]

        # Overwritting


    def _assertions(self):
        """
        Assert class inputs are consistent.
        """
        # TODO correct full function

        assert type(label) is str
        if filepath is not None:
            if overwrite or dataset not in _soi_fullpath:
                print('Overwritting dataset {}\n Previous dataset still stored in trash'.format(dataset, self.base_label))
            elif label not in _soi_fullpath:
                print('Adding {} to dataset'.format(label))
            else:
                raise IOError('When not overwritting, do not input filepath')
        else:
            assert overwrite is False
            try:
                assert label in _soi_fullpath
            except:
                raise IOError('If label is not pre-saved, must provide filepath')
        for key in kwargs:
            try:
                assert type(kwargs[key]) is float or type(kwargs[key]) is str or type(kwargs[key]) is int
            except:
                raise TypeError('keyword argument {} was not accepted, because it was not integer/float/string'.format(key))


    def _get_filepath(self):
        all_datasets = np.array([name.split('/')[-2] for name in self._soi_fullpath])
        if type(dataset_name) is str:
            assert len(all_datasets == self.dataset_name)
            return self._soi_fullpath[all_datasets==self.dataset_name]
        elif type(dataset_name) is list:
            raise NotImplementedError
        else:
            return TypeError('The dataset_name shoud be str or list, not {}'.format(type(dataset_name)))

    def _detect_extension(self):
        ext = self.filepath.split('.')[-1]
        if ext in ['pickle', 'pkl', 'pk']:
            if self._functioning is "loading":
                return lambda f: pickle.load(open(f, 'rb'))
            else:
                return lambda d, f: pickle.dump(d, open(f,'wb'))
        elif ext in ['csv']:
            if self._functioning is "loading":
                return lambda f: pd.load_csv(f)
            else:
                return lambda d, f: pd.to_csv(d, f)
        elif ext in ['h5', 'hdf5', 'h5py']:
            raise NotImplementedError
        else:
            raise ValueError("The detected extension {} is not supported. Please input a pickle, csv or hdf5 file".format(ext))

    def _load(self):
        return self._detect_extension()(self.filepath)

    def _overwrite(self):
        # TODO overwrite file, saving previous one if filename
        # or carrying annotations if just adding identifiers
        pass
    def _create(self):
        if self._dataset_exists:
            self.basepath = self.shortcuts[self.label]['basepath']
        assert self.basepath is not None
        assert path[:len(self.basepath)] == self.basepath

        # Updating shortcuts
        path = self.filepath.replace(self.basepath,'').split('/')
        assert type(path) is list
        insidecut = self.shortcuts[self.label][self.basepath]

        for folder in path[:-1]:
            if folder not in insidecut:
                insidecut[folder] = {}
            insidecut = insidecut[folder]

        insidecut[path[-1]] = 'Created in {}, {}'.format(str(datetime.today()),self.annotations)
        json.dump(self.shortcuts, open('shortcuts.json','w'), indent='\t')

        # Creating datafile
        self._detect_extension()(self.data, self.filepath)
