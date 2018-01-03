"""
A module for data loading, processing and saving.
"""

import glob
import json
from .utils import get_filepaths_from_shortcut
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
          "/home/user/Documents/project_name/data":{
            "raw" : "filename located in /raw/",
            "interim" : "filename located in /interim/",
            "processed" : {
              "data_label" : "corresponding file full path/name",
              "example" : "example.mat #located in base/processed",
              "another_example" : "smoothed.pickle"
            }
          },
          "results" : "another_filepath_or_dict"
        }
    }
    json.dump(default_example, open('shortcuts.json','w'), indent='\t')


SHORTCUTS = json.load(open('shortcuts.json','r') )
del SHORTCUTS['Name_example']
# If there are saves in shortcuts, make sure folders and files exist.
for label in SHORTCUTS:
    all_files = get_filepaths_from_shortcut(SHORTCUTS[label])
    assert all([len(glob.glob(filepath))==1 for filepath in all_files if len(filepath)>0])
