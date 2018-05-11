"""
Generates the shortcut dictionary that will be used for loading and saving
data for the project.

Run this script from the base of directory. For example, see ../../Makefile
"""
import sys
sys.path.append('.')
import os

import json
from spikelearn.data import io, SHORTCUTS

# Label for each animal
DRRD_RATS = ['DRRD 7','DRRD 8','DRRD 9','DRRD 10']
AUTOSHAPE_RATS = ['Autoshape 7','Autoshape 8','Autoshape 9','Autoshape 10']

EZ1_RATS = ['ELI 3', 'ELI 4', 'ELI 5', 'ELI 6']
EZ2_RATS = ['ELI 3_2', 'ELI 4_2', 'ELI 5_2', 'ELI 6_2']
EZ_RATS = EZ1_RATS + EZ2_RATS

GB_RATS = DRRD_RATS + AUTOSHAPE_RATS
ALL_RATS = GB_RATS + EZ_RATS

shortcuts = SHORTCUTS
basepath = os.getcwd()
# Add raw data
for rat in ALL_RATS:
    shortcuts[rat] = {}
    shortcuts[rat]['basepath'] = basepath
    shortcuts[rat]['data'] = {}
    shortcuts[rat]['data']['raw'] = {}
    shortcuts[rat]['data']['external'] = {}

    shortcuts[rat]['data']['raw']['spikesorted'] = rat+'.mat'

# Add external data
for rat in DRRD_RATS:
    shortcuts[rat]['data']['external']['selected_neurons'] = 'selected_neurons_{}.pickle'.format(rat)
    shortcuts[rat]['data']['external']['tiredness'] = '{}.csv'.format(rat)
    shortcuts[rat]['data']['external']['changepoint'] = '{}.csv'.format(rat)

# Add group for easy access
shortcuts['groups'] = {}
shortcuts['groups']['DRRD'] = {rat :'' for rat in DRRD_RATS}
shortcuts['groups']['Autoshape'] = {rat :'' for rat in AUTOSHAPE_RATS}
shortcuts['groups']['GB'] = {rat :'' for rat in GB_RATS}
shortcuts['groups']['EZ'] = {rat :'' for rat in EZ_RATS}
shortcuts['groups']['ALL'] = {rat :'' for rat in ALL_RATS}

json.dump(shortcuts, open('shortcuts.json','w'), indent='\t')
