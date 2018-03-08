"""
A module for data loading, processing and saving.
"""

import glob
import json
import numpy as np
from .utils import get_filepaths_from_shortcut
from .selection import select, to_feature_array
from .preprocessing import remove_baseline

#module_path = os.path.abspath(os.path.dirname(__file__))
#shortcuts_path = module_path+'/shortcuts.json'
#print(module_path)


# Check if shortcuts.json exists
if len(glob.glob('shortcuts.json')):
    pass
else: # if not, create with default example
    default_example = {
        "Name_example":{
          "basepath" : "/home/user/Documents/project_name/data",
          'data':{
            "raw" : "filename located in /raw/",
            "interim" : "filename located in /interim/",
            "processed" : {
              "data_label" : "filename",
              "example" : "example.mat #located in base/processed",
              "another_example" : "smoothed.pickle"
            },
            "results" : "another_filepath_or_dict"
          }
        }
    }
    json.dump(default_example, open('shortcuts.json','w'), indent='\t')

SHORTCUTS = json.load(open('shortcuts.json','r') )
if 'Name_example' in SHORTCUTS:
    del SHORTCUTS['Name_example']
# If there are saves in shortcuts, make sure folders and files exist.
for label in SHORTCUTS:
    if label != 'groups':
        all_files = get_filepaths_from_shortcut(SHORTCUTS[label])
        try:
            file_exists = np.array([len(glob.glob(filepath))==1 for filepath in all_files if len(filepath)>0])
            assert all(file_exists)
        except AssertionError:
            raise IOError('Filepaths %s do not exist'%all_files[np.where(file_exists == 0)])
