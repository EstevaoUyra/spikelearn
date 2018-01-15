import os
import sys
sys.path.append('.')


import json
from spikelearn.dataprocessing import io, SHORTCUTS

DRRD_RATS = ['DRRD 7','DRRD 8','DRRD 9','DRRD 10']
AUTOSHAPE_RATS = ['Autoshape 7','Autoshape 8','Autoshape 9','Autoshape 10']
ALL_RATS = DRRD_RATS + AUTOSHAPE_RATS

shortcuts = SHORTCUTS
basepath = os.getcwd()
for rat in ALL_RATS:
    shortcuts[rat] = {}
    shortcuts[rat]['basepath'] = basepath
    shortcuts[rat]['data'] = {}
    shortcuts[rat]['data']['raw'] = {}
    shortcuts[rat]['data']['external'] = {}

    shortcuts[rat]['data']['raw']['spikesorted'] = rat+'.mat'

for rat in DRRD_RATS:
    shortcuts[rat]['data']['external']['selected_neurons'] = 'selected_neurons_{}.pickle'.format(rat)

json.dump(shortcuts, open('shortcuts.json','w'), indent='\t')
