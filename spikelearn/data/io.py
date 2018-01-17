"""
Input (loading) and output (saving) functions.
Lower level functions, usually called from the api
"""

import pandas as pd
import json

from datetime import datetime
from .utils import get_filepaths_from_shortcut
import os
import numpy as np
import pickle

DATA_SUBFOLDERS = ('raw', 'interim', 'processed')
RESULTS_SUBFOLDERS = ()

class DataShortcut():
    """
    Loads and saves data into DataFrame, getting shortcuts from
    a pre-edited JSON file shortcuts.json

    Parameters
    ----------
    base_label : string
        Base dataset label. If not contained in shortcuts.json, then the filename
        parameter is needed.

    dataset_name : string, default None
        Name of the specific dataset to load.
        if None (default), raises TypeError and
        prints a list of all datasets

    dataset_type : string, default 'auto'
        Can be "raw", "interim", "processed", "results" or 'auto'
        If 'auto' searches for the dataset in all folders

    filename : string, path-to-file, optional
        Name of the path of datafile. If provided jointly with label, the
        overwrite flag must be explicitly set to True.

    basepath : string, optional
        full path of some folder in filename hierarchy
        serves to make shortcuts.json identation less polluted

    overwrite : bool, optional, default False
        Whether to overwrite label with filename

    annotations : str or list of str, optional
        Observations to add to the file description

    getpath : bool, optional, default False
        Whether to get only the path to file,
        Avoiding loading and/or creating

    extension : string, optional, default 'auto'
        Must be specified for creating or overwritting datasets.
        Options are 'pickle', 'csv'

    Keyword Arguments
    -----------------
    Add any number of identifier variables with keyword arguments,
    which must be numbers or strings.

    Notes
    -----
    Uses the datafiles.json to search and load data using simple
    name labels.
    """
    def __init__(self, base_label, dataset_name=None, dataset_type='auto', filename=None, basepath=None, overwrite=False, data = None, annotations='', getpath=False, extension='auto', **kwargs):
        self.base_label = base_label
        self.dataset_name = dataset_name
        self.overwrite = overwrite
        self.getpath = getpath
        self.annotations = annotations

        if dataset_type in ['raw', 'interim', 'processed']:
            self.dataset_type = 'data/'+dataset_type
        else:
            self.dataset_type = dataset_type

        # Make sure inputs are consistent
        #TODO self._assertions()

        # Load the shortcuts
        self.shortcuts = json.load(open('shortcuts.json','r') )
        if base_label in self.shortcuts:
            self._soi_fullpath = get_filepaths_from_shortcut(self.shortcuts[base_label])
            self._dataset_exists = True
            self.basepath = self.shortcuts[base_label]['basepath']
        else:
            self._dataset_exists = False
            assert basepath is not None
            self.basepath = basepath
            self._new_label()

        # Specific case of no dataset listed
        if dataset_name is None:
            dataset_names = np.array([name.split('/')[-2] for name in self._soi_fullpath])
            raise TypeError('You must input the dataset name. For {}, possible datasets are {}'.format(base_label, dataset_names))

        # Get data at last
        if data is None:
            self.filename = self._get_filename()
            self._get_extension(extension)
            self._functioning = "loader"
            self.data = self._load()

        elif not overwrite: # Dataset is new
            self.data = data
            self._functioning = "creator"
            if filename is None:
                self.filename = '_'.join([dataset_name,base_label])
            else:
                self.filename = filename
            self._get_extension(extension)
            self._create()

        # Add identifiers
        # TODO identifiers in numpy arrays
        # TODO identifiers before saving
        for key in kwargs:
            if key in self.data.columns:
                raise ValueError("Identifier {} already exists")
            else:
                self.data[key] = kwargs[key]

        # Overwritting
        if self.overwrite:
            self._overwrite()

    def _new_label(self, basepath):
        self.shortcuts[self.base_label] = {}
        labeldict = self.shortcuts[self.base_label]
        labeldict['basepath'] = basepath
        labeldict['data'] = {subfolder : {} for subfolder
                                    in DATA_SUBFOLDERS}
        labeldict['results'] = {subfolder : {} for subfolder
                                    in RESULTS_SUBFOLDERS}

    def _get_extension(self, extension):
        if extension == 'auto':
            self.extension = self.filename.split('.')[-1]
        else:
            self.extension = extension

    def _assertions(self):
        """
        Assert class inputs are consistent.
        """
        # TODO correct whole function

        assert type(label) is str
        if filename is not None:
            if overwrite or dataset not in _soi_fullpath:
                print('Overwritting dataset {}\n Previous dataset still stored in folder'.format(dataset, self.base_label))
            elif label not in _soi_fullpath:
                print('Adding {} to dataset'.format(label))
            else:
                raise IOError('When not overwritting, do not input filename')
        else:
            assert overwrite is False
            try:
                assert label in _soi_fullpath
            except:
                raise IOError('If label is not pre-saved, must provide filename')
        for key in kwargs:
            try:
                assert type(kwargs[key]) is float or type(kwargs[key]) is str or type(kwargs[key]) is int
            except:
                raise TypeError('keyword argument {} was not accepted, because it was not integer/float/string'.format(key))


    def _get_filename(self):
        dataset_names = np.array([name.split('/')[-2] for name in self._soi_fullpath])

        # Last ([-1]) is filename, next to last ([-2]) is dataset_name
        if type(self.dataset_name) is str:
            if self.dataset_type == 'auto':
                index = dataset_names==self.dataset_name
            else:
                index = np.logical_and((dataset_names==self.dataset_name,
                          [self.dataset_type in name for name in self._soi_fullpath]))
            if sum(index) == 1:
                return self._soi_fullpath[index][0]
            elif sum(index) == 0:
                raise IOError(0, self.dataset_name)
            elif sum(index) > 1:
                raise IOError(999, self.dataset_name)
        elif type(self.dataset_name) is list:
            raise NotImplementedError
        else:
            return TypeError('The dataset_name shoud be str or list, not {}'.format(type(dataset_name)))

    def _enforce_extension(self):
        if self.extension in ['pickle', 'pkl']:
            if self._functioning is "loader":
                return lambda f: pickle.load(open(f, 'rb'))
            else:
                return lambda d, f: pickle.dump(d, open(f,'wb'))
        elif self.extension in ['csv']:
            if self._functioning is "loader":
                return lambda f: pd.read_csv(f)
            else:
                return lambda d, f: pd.to_csv(d, f)
        elif self.extension in ['h5', 'hdf5', 'h5py']:
            raise NotImplementedError
        elif self.extension == 'mat':
            raise NotImplementedError
        else:
            raise ValueError("The detected extension .{} is not supported. Please input a pickle, csv or hdf5 file".format(self.extension))

    def _load(self):
        if self.getpath:
            return self.filename
        else:
            return self._enforce_extension()(self.filename)

    def _overwrite(self):
        # TODO overwrite file, saving previous one if filename
        # or carrying annotations if just adding identifiers
        pass

    def _create_path(self):
        if self.basepath not in self.filename:
            self.filename = '{}/{}/{}/{}.{}'.format(self.basepath, self.dataset_type, self.dataset_name, self.filename,self.extension)

    def _create_annotations(self):
        with open(''.join(self.filename.split('.')[:-1])+'_annotations.txt', 'w') as f:
            f.write('Created in {}'.format(str(datetime.today())))
            if type(self.annotations) is str:
                f.write('\n{}'.format(self.annotations))
            elif type(self.annotations) is list:
                [f.write('\n{}'.format(annot)) for annot in self.annotations]
            else:
                raise TypeError('Annotations must be strings or lists of strings')

    def _create(self):
        self.basepath = self.shortcuts[self.base_label]['basepath']
        self._create_path()

        assert self.basepath is not None
        assert self.filename[:len(self.basepath)] == self.basepath

        # Updating shortcuts
        path = self.filename.replace(self.basepath,'').split('/')
        assert type(path) is list
        insidecut = self.shortcuts[self.base_label]
        for folder in path[1:-1]:
            folderpath = self.filename[:self.filename.index(folder)]+folder
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            if folder not in insidecut:
                insidecut[folder] = {}
            if folder is not path[-2]:
                insidecut = insidecut[folder]

        insidecut[self.dataset_name] = path[-1]
        json.dump(self.shortcuts, open('shortcuts.json','w'), indent='\t')

        # Creating datafile
        self._enforce_extension()(self.data, self.filename)
        self._create_annotations()



def load(base_label, dataset_name, getpath=False, dataset_type='auto'):
    """
    Lightweight loader that used DataShortcut under the hood.

    See also
    --------
    DataShortcut
    """
    return DataShortcut(base_label, dataset_name, getpath=getpath, dataset_type=dataset_type).data


def save(data, base_label, dataset_name, dataset_type= 'processed', extension='pickle', annotatons='', **kwargs):
    """
    Lightweight saver that used DataShortcut under the hood.

    See also
    --------
    DataShortcut
    """
    DataShortcut(base_label, dataset_name, dataset_type=dataset_type,  annotatons=annotatons, data=data, extension=extension, **kwargs)

def dataset_exist(base_label, dataset_name):
    try:
        load(base_label, dataset_name, getpath=True)
        return True
    except IOError as e:
        if e.errno == 0:
            return False
        elif e.errno > 1:
            raise NotImplementedError
