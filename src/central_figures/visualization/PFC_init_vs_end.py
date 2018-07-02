import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

mlp.rcParams['font.size'] = 18
label_size, title_size = 22, 25
vmin, vmax = 0.08,.13

savedir = 'reports/figures/central_figures'

fig = plt.figure(figsize=(10,10))
pfc_1 = fig.add_axes([0,.45, .4, .4])
pfc_2 = fig.add_axes([.42,.45, .4, .4])
cbar = plt.axes((.88, .1, .05, .6), facecolor='w')

import pickle
res = pickle.load('data/results/central_figures/PFC_init_vs_end.pickle')
# Plot results
res.proba_matrix(grouping=('tested_on','pfc_day1'), vmin=vmin, vmax=vmax, cbar=False, ax=pfc_1);
res.proba_matrix(grouping=('tested_on','pfc_day2'), vmin=vmin, vmax=vmax, cbar=False, ax=pfc_2);

# Plot colorbar
sns.heatmap(np.linspace(vmin,vmax,100).reshape(-1,1)[::-1],cbar=False, ax=ax)
cbar.yaxis.tick_right()
cbar.tick_params(rotation=0)
cbar.set_yticks(np.linspace(0,100,101)[::-20]); cbar.set_xticklabels(['']);
cbar.set_yticklabels(np.linspace(vmin,vmax,101)[::20].round(3));

pfc_1.set_title('PFC, day 1',fontsize=title_size)
pfc_2.set_title('PFC, day 2',fontsize=title_size)

pfc_1.set_xticks([]); pfc_1.set_xlabel('')
pfc_2.set_yticks([]); pfc_2.set_ylabel('')
pfc_2.set_xticks([]); pfc_2.set_xlabel('')

pfc_1.set_ylabel('Time (s)', fontsize=label_size)
pfc_1.set_xlabel('Decoded time (s)', fontsize=label_size)
pfc_2.set_xlabel('Decoded time (s)', fontsize=label_size)

if not os.path.exists(savedir):
    os.makedirs(savedir)

plt.savefig('{}/PFC_init_vs_end.png', dpi=300,
                                        bbox_inches='tight').format(savedir)
plt.savefig('{}/PFC_init_vs_end.eps', dpi=300,
                                        bbox_inches='tight').format(savedir)
