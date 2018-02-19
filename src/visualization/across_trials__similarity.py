"""
This script generates

"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes("dark")
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from spikelearn.data import SHORTCUTS, io, select
from spikelearn.visuals.visuals import to_video

from sklearn.preprocessing import MinMaxScaler
from itertools import product

# Data parameters
DSETS = ['medium_smoothed', 'medium_smoothed_norm',
         'narrow_smoothed', 'narrow_smoothed_norm',
         'wide_smoothed']
TMIN = 1.5
WSIZE = 50
loaddir = 'data/results/across_trials/similarity/'
sim_filename = lambda label, dset: '{}/{}_w{}_t{}_unit_sim_evolution.csv'.format(loaddir+dset, label, WSIZE, TMIN)
pred_filename = lambda label, dset: '{}/{}_w{}_t{}_pred_init_end.csv'.format(loaddir+dset, label, WSIZE, TMIN)
weight_filename = lambda label, dset: '{}/{}_w{}_t{}_weight_init_end.csv'.format(loaddir+dset, label, WSIZE, TMIN)
scaler = MinMaxScaler((-1,1))


# Saving parameters
fsavedir = lambda label, dset : 'reports/figures/across_trials/similarity/{}/{}'.format(dset, label)

# Identifier variables
id_vars = ['logC', 'penalty']
id_combs = lambda df: product(weights.unique(df[id_]) for id_ in id_vars)

# Create visualizations
for label, dset in product(SHORTCUTS['groups']['DRRD'], DSETS):
    print(label, dset)
    savedir = fsavedir(label, dset)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # Loading data
    similarities = pd.read_csv(sim_filename(label, dset)).set_index(['unit','trial'])
    units = similarities.reset_index().unit.unique()
    behav = io.load(label, 'behav_stats').reset_index()
    behav = select(behav, trial_in_=similarities.reset_index()['trial'])

    # One image for each neuron
    for unit in units:
        fig = plt.figure()
        sns.heatmap(similarities.loc[unit].values)
        plt.title('Unit {}, {}'.format(unit, label) )
        plt.savefig('{}/sim_evo_unit_{}.png'.format(savedir, unit),
                        dpi=500)
        plt.close(fig)

    # Plus one image for the mean
    fig = plt.figure()
    sns.heatmap(similarities.reset_index().groupby('trial').mean().drop('unit', axis=1))
    plt.title('Similarity with mean across units')
    plt.savefig('{}/sim_evo_mean.png'.format(savedir),
                    dpi=800)
    plt.close(fig)

    # Classifier-dependent analysis (Weighted mean & prediction evolution)
    ## Loading data
    preds = pd.read_csv(pred_filename(label, dset))
    weights = pd.read_csv(weight_filename(label, dset))
    ## Prepare list of figures to video
    img_order = []
    ## Analysis
    ids =  weights.drop_duplicates(subset=id_vars)[id_vars]
    ### bar limits
    maxbar = ids.groupby('penalty').min().transpose()
    minbar = ids.groupby('penalty').max().transpose()
    for id_ in map(tuple, ids.values):
        print(id_)
        ## Calculate weighted mean similarity
        weight = weights.groupby(id_vars).get_group(id_).set_index('unit')
        weight
        ponder = lambda df: df*weight.loc[df.unit.values[0], 'w']
        wm = similarities.reset_index().groupby('unit').apply(ponder)
        wm['trial'] = similarities.index.get_level_values('trial')
        wm['unit'] = similarities.index.get_level_values('unit')
        ## Get prediction trialseries
        pred = preds.groupby(id_vars).get_group(id_).sort_values('trial')

        # Create plot
        fig = plt.figure()

        ## Similarity
        ax = plt.subplot2grid((4,4),(1,0),rowspan=3,colspan=3)
        wm = wm.groupby('trial').sum().drop('unit',axis=1)
        scaler.fit(wm.values.reshape(-1, 1));
        scale = lambda x: scaler.transform(x.values.reshape(-1, 1)).reshape(-1)
        sns.heatmap( wm.apply(scale), ax=ax,cbar=False,
                        vmin=-1, vmax=1, cmap='RdBu_r')

        ## Prediction evolution
        ax = plt.subplot2grid((4,4), (0,0), colspan=3)
        trials = pred.trial.unique()
        ax.plot(trials, pred.predicted.values, linewidth=.5, alpha=.7)
        ax.plot(trials, pred.predicted.rolling(20,center=True).mean().values, linewidth=2)
        plt.ylim([0, 1]); plt.xticks([]); plt.ylabel('P(init)')

        ax.fill_betweenx((0,1), trials[0], trials[WSIZE], color='g', alpha=.5)
        ax.fill_betweenx((0,1), trials[-WSIZE], trials[-1], color='g', alpha=.5)

        ## Behavior (increasing from above)
        behav= behav.sort_values("trial", ascending=False)
        ax = plt.subplot2grid((4,4), (1,3), rowspan=3)
        ax.plot(behav['duration'].values, behav['trial'].values,
                    'b.', label='Duration', markersize=.5, alpha=.5)
        ax.plot(behav['duration'].rolling(20,center=True).mean().values, behav['trial'],
                    color='b', linewidth=2)
        plt.xlim([0,7]); plt.xlabel("Trial duration", color='b');
        ax.yaxis.tick_right(); ax.invert_yaxis()
        ax = ax.twiny()
        ax.plot(behav['intertrial_interval'], behav['trial'], 'g.', label='Intertrial interval', markersize=.5, alpha=.5)
        ax.plot(behav['intertrial_interval'].rolling(20,center=True).mean(), behav['trial'],
                    color='g', linewidth=2)
        plt.xlabel("Intertrial interval", color='g');
        plt.xlim([0,100]); plt.ylim([trials[0], trials[-1]])

        ## Identifiers
        ax = plt.subplot2grid((4,4), (0,3), rowspan=1); plt.axis('off')
        ax.text(0.2, 0.7, 'Penalty: {}'.format(id_[1]))
        ax.text(0.2, 0.5, 'logC: {:.2f}'.format(id_[0]))
        color = 'c' if id_[1] == 'l1' else 'm'
        # plt.fill_betweenx((.6,.8 ), 0,1, alpha=.5, color=color)
        size = (id_[0] - minbar[id_[1]])/(maxbar[id_[1]] - minbar[id_[1]])
        plt.barh(.65, size, .5, color=color, alpha=size.values[0])
        plt.ylim([0,1]); plt.xlim([0,1])

        # Finalize plot and save
        plt.suptitle('Similarity weighted by Logistic Regression coefs')
        figname = '{}/sim_evo_mean_weighted_{}.png'.format( savedir,'_'.join(map(str,id_)))
        plt.savefig(figname, dpi=500)
        plt.tight_layout()
        plt.close(fig)
        img_order.append(figname)
    ordfile = '{}/img_order'.format(savedir)
    np.savetxt(ordfile, np.array(img_order), fmt='%s')
    to_video(ordfile, '{}/by_regularization'.format(savedir))
