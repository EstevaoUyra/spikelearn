import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from itertools import product
import subprocess
import matplotlib as mpl

def to_video(listfile, outputfile,
                fps=3, w=800, h=600, type='png'):
    """
    Simply runs mencoder to merge images into a .avi video.

    Parameters
    ----------
    listfile : string, path-to-file
        Path to a file containing the image names in its rows, in the order
        that they should appear on the video.

    outputfile : string
        The name of the output video.
        Obs: the name will be completed by .avi

    fps, w, h : int
        Parameters of the video. defaults 3, 800, 600
        Frame rate, width and height.

    type : string, default 'png'
        Type of the images being transformed.
    """
    subprocess.run(['mencoder',
                    'mf://@{}'.format(listfile)]+
                    '-mf w={}:h={}:fps={}:type={} -ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell -oac copy -o'.format(w, h, fps, type).split(' ') +
                    ['{}.avi'.format(outputfile)])

def raster_multiple(spike_trains, time='time', yaxis='trial', xlim=None,
        col=None, row=None, rows=None, hue=None,hue_order=None, title=None,sharey=False,
        palette='deep', density=False, n_yticks=None, is_list=False,
        kde_kwargs=None, fig_kwargs=None, **kwargs):
    """
    Parameters
    ----------
    spike_trains : DataFrame
        Each row contains identifier variables, and the time column should
        contain a single time or a list of times for each row.
        Note: if the format is list of times, the function will take longer,
        and this must be flagged at is_lista

    time : string, optional, default: 'time'
        DataFrame column index containing the times or time lists.
        If not 'time', the name of the column is added to the xlabel
        default label is 'Time (ms)', and it turns ('Time ({})').format(time)

    yaxis : string, optional, default: 'trial'
        Variable that defines the vertical position of the row's data.

    xlim: tuple or list of tuples, optional
        Limits of the horizontal axis of each subplot.
        If list, applies one for each *row*

    col : string, optional
        Defines which variable is unique in each subplot column

    row : string, optional
        Defines which variable is unique in each subplot row

    rows : list, optional
        The order in which to plot the fields in the rows

    hue : string, optional
        Defines which variable is unique in each color

    hue_order : list, optional
        The order in which hues will get colored

    palette : list or str, optional
        list of colors or name of seaborn color_palette. Defaults to 'Set3'

    density : bool or str, optional
        Whether to show the time density of spikes on top of each subplot.
        If str, selects column to groupby and plot multiple densities.
        If 'hue', plots a density for each color.

    n_yticks : int, optional, default: None
        Number of ticks on the vertical axis of each subplot

    is_list : bool, optional, default: False
        Flags the rows as containing list of times, instead of values.
        Making the transformation takes time, and it is preferable to do
        this transformation outside the plotting function.
        DEPRECATED, now only accepts list format



    See also
    --------
    raster_plot

    """
    ## Mechanics
    # Make local copy
    data = spike_trains.copy()
    # Make mutables non-persistent
    if kde_kwargs is None:
        kde_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

    # Auxilliary columns
    if col is None:
        col = 'aux_col'
        data['aux_col'] = 1
    if row is None:
        row = 'aux_row'
        data['aux_row'] = 1
    if hue is None:
        hue = 'aux_hue'
        data['aux_hue'] = 1

    ## Define plot aesthetics
    # Geometry
    cols=data[col].unique()
    n_cols = len(cols)
    if rows is None:
        rows = data[row].unique()
    n_rows = len(rows)

    subtitle = lambda _r, _c: '{}: {}, '.format(row, _r)*(row!='aux_row')+'{}: {} '.format(col, _c)*(col!='aux_col')

    if 'figsize' not in fig_kwargs:
        fig_kwargs['figsize'] = (6*n_cols, 8*n_rows)

    fig = plt.figure(**fig_kwargs)
    # Colors
    if hue_order is None:
        hue_order = data[hue].unique()
    n_hue = hue_order.shape[0]
    if type(palette) is str:
        palette = sns.palettes.color_palette(palette, n_hue)
    else:
        assert len(palette) is n_hue

    all_axes = np.array(fig.subplots(n_rows, n_cols, sharey=sharey)).reshape(n_rows,n_cols)

    # Text
    xlabel = 'Time ({})'.format('ms' if time=='time' else time)
    ylabel = yaxis

    ## Plotting per se
    # Each subplot
    for row_i, col_i in product(range(n_rows), range(n_cols)):
        ax = all_axes[row_i,col_i]
        sub_data = data.groupby([row, col]).get_group((rows[row_i], cols[col_i]))

        raster_plot(sub_data,time=time,yaxis=yaxis, hue=hue, hue_order=hue_order, density=density, ax=ax, palette=palette,kde_kwargs=kde_kwargs, **kwargs)

        plt.sca(ax)
        if n_yticks is not None:
            ticks = np.linspace(sub_data[yaxis].min(),sub_data[yaxis].max(), n_yticks).astype(int)
            plt.yticks(ticks,ticks)


        plt.title(subtitle(rows[row_i], cols[col_i]))
        plt.ylabel(ylabel); plt.xlabel(xlabel)
        if type(xlim) is tuple: plt.xlim(xlim)
        elif type(xlim) is list:
            if xlim[row_i] is not None:
                plt.xlim(xlim[row_i])
        if title is not None:
            plt.suptitle(title,y=1.01, x=.5,fontsize=16)
    plt.tight_layout()
    return fig

def raster_plot(data, time='time', yaxis='trial', hue=None, hue_order=None, density=False, ax=None, palette='deep',  is_list=True, kde_kwargs=None, **kwargs):
    """
    data :

    time : string, optional, default: 'time'
        DataFrame column index containing the times or time lists.
        If not 'time', the name of the column is added to the xlabel
        default label is 'Time (ms)', and it turns ('Time ({})').format(time)

    yaxis : string, optional, default: 'trial'
        Variable that defines the vertical position of the row's data.

    hue : string, optional
        Defines which variable is unique in each color

    hue_order : list, optional
        The order in which hues will get colored

    density : bool or str, optional
        Whether to show the time density of spikes on top of each subplot.
        If str, selects column to groupby and plot multiple densities.
        If 'hue', plots a density for each color.

    palette : list or str, optional
        list of colors or name of seaborn color_palette. Defaults to 'Set3'

    ax : matplotlib axes, optional
        The axis on which to plot

    """
    data = data.copy()
    if kde_kwargs is None:
        kde_kwargs = {}
    if ax==None:
        ax = plt.subplot(1,1,1)
    if hue is None:
        hue = 'aux_hue'
        data['aux_hue'] = 1
    if hue_order is None:
        hue_order = data[hue].unique()
    n_hue = hue_order.shape[0]

    if type(palette) is str:
        palette = sns.palettes.color_palette(palette, n_hue)
    else:
        assert len(palette) is n_hue

    if density:
        axd = ax.twinx()
    # Each color
    for hue_i, hue_l in enumerate(hue_order):
        if hue_l in data[hue].values:
            hue_data = data.groupby(hue).get_group(hue_l)
            if is_list:
                for _, list_data in hue_data.iterrows():
                    ylist = list_data[yaxis]*np.ones(len(np.array(list_data[time])))
                    ax.scatter(list_data[time], ylist, color=palette[hue_i], **kwargs)
            else:
                ax.scatter(hue_data[time], hue_data[yaxis], color=palette[hue_i], **kwargs)
            # Density
            if density:
                if is_list:
                    sns.kdeplot(np.hstack(hue_data[time].values), ax=axd, color=palette[hue_i], **kde_kwargs)
                else:
                    sns.kdeplot(hue_data[time].values, ax=axd, color=palette[hue_i], **kde_kwargs)
    plt.sca(ax)
    plt.ylim([data[yaxis].min(), data[yaxis].max()])
    return ax


def pretty_neurons(spike_vector=None, show_spikes=True, sigma=0,
                    n_neurons=8, rate = 1, tmax=10,
                    spacing = 1, palette='Set2', bin_fill=None,
                    alpha_multiplier = 'auto', alpha_power = 1/2,
                    binpalette='plasma', bin_sep=None, fig_kwargs=None,):
    """ Plots multiple point processes in parallel lines.

    Parameters
    ----------
    spike_vector : array-like (n_neurons, max_n_spikes)
        Each row contains the spike times of one unit, and will be plotted in
        one line. May contain NANs, which are ignored.

    show_spikes : bool, default True
        Which to show spikes or only smoothed activity.
        If True and sigma > 0, plot will be empty.

    sigma : int
        Width of gaussian kernel to smooth spike-series.

    n_neurons : int
        Number of units to be generated.
        Is overwritten by spike_vector

    rate : float
        Firing rate of each generated unit.
        Active only if spike_vector is None

    tmax : float
        Maximum time until which to simulate activity, in seconds.
        Active only if spike_vector is None

    Other Parameters
    ----------------
    spacing : float, default 1
        Vertical distance between each baseline

    palette : string
        If len>1, names a Seabon color_palette
        If char (len==1), is the color of all neurons

    bin_fill : tuple, (int, int)
        Position (x,y) of bin to be color-filled.
        According to the value of the binning.

    alpha_multiplier : float or 'auto'
        How much to multiply each bin fill alpha value.
        'auto' selects a number such that the maximum is one.

    alpha_power : float
        How much to exponentiate each bin's firing rate to get the alpha.
        Smaller values approximate big values and decrease difference.

    bin_sep: int
        How much vertical separators to use in the bin separation.

    binpalette : str
        Which palette to
        Note: It does not correspond to bin filling,
        only to vertical separators.


    """

    # Produce random spike vector if not provided
    if spike_vector is None:
        spike_vector = np.random.exponential(1/rate,(n_neurons,int(1.5*tmax*rate))).cumsum(axis=1)
    else:
        n_neurons = spike_vector.shape[0]

    # Define color of each unit
    assert type(palette) is str
    if len(palette) > 1:
        neuroncolors = sns.palettes.color_palette(palette,n_neurons)
    else:
        neuroncolors = palette*n_neurons

    ## Plotting per se
    fig=plt.figure(**fig_kwargs)
    ax=plt.subplot(1,1,1)

    # Base horizontal lines
    ax.hlines((spacing)*np.arange(n_neurons), -.4,11,colors=neuroncolors);

    # Point process
    if show_spikes:
        for i in range(n_neurons):
            ax.vlines(spike_vector[i,:],(spacing)*i,(spacing)*i+.7, colors=neuroncolors[i], linewidth=3)

    # Smoothed activity
    nbins = (tmax+2)*1000
    s=np.zeros((n_neurons,nbins))
    if sigma >= 2:
        for i in range(n_neurons):
            v, t = np.histogram(spike_vector[i,:], bins=nbins,range=(-1,tmax+1))
            s[i,:] = kernelSmooth(v, sigma)
            plt.plot(t[:-1],s[i,:]*2.6*(sigma**(2/3))+(spacing)*i,linewidth=5,color = neuroncolors[i])
    # Vertical separators
    bincmap = sns.palettes.color_palette(binpalette,tmax)
    if bin_sep is not None:
        bincmap = sns.palettes.color_palette('plasma',tmax+1)
        plt.vlines(np.arange(0,bin_sep+1),-1,n_neurons,linewidth=8,colors=bincmap,linestyle='-')
    # Each bin value in its color alpha
    if bin_fill is not None:
        for j in bin_fill[0]:
            for i in bin_fill[1]:
                plt.fill_between([i,i+1],j,j+1,color = neuroncolors[j], alpha=(binarize(s[j,:],1000)[1+i]*alpha_multiplier)**alpha_power)
    # Aesthetics
    ax.set_yticks((spacing)*np.arange(n_neurons)+(spacing)*.5)
    ax.set_yticklabels(['Neuron %d'%i for i in range(1,n_neurons+1)],fontsize=12)
    ax.set_xlim(-.4,tmax+.4); ax.set_ylim(-0.2,max(s[-1,:].max()+n_neurons,(spacing)*n_neurons+.2))
    return fig, spike_vector, s


def singleRatBehaviorPlot(durations, tVec=None, threshold='max',cpax=True, s=20,
                            Tc = 1.5, kde='fl', kdeN = 100, ticksize=14,
                            axislabel_size=16, reverse=False, tmax = 5,
                            figsize = 4):
    #TODO document function
    """
    Plots the responses along the trials, together with the
    """
    f = plt.figure(figsize=(figsize,figsize))
    mpl.rcParams['font.size']=ticksize

    trialNumber = np.arange(len(durations))
    if tVec is None:
        tVec = (np.arange(len(durations)) == len(durations)//2).astype(int)

    # Behavioral dots
    nonRewarded = (durations < Tc)
    plt.scatter(durations[nonRewarded],trialNumber[nonRewarded],s=s, marker='o',facecolors='none',edgecolor='k')#=(0.6, 0.036000000000000004, 0.0))#(0.48, 0.1416, 0.12))



    rewarded = (durations >= Tc)
    plt.scatter(durations[rewarded],trialNumber[rewarded],color='k', s=s, marker='o')#(0.0, 0.6, 0.18600000000000008)

    plt.ylim([0,len(durations)]); plt.xlim([0,tmax]);
    plt.ylabel('Trial number',fontsize=axislabel_size);
    if not reverse:
        plt.xlabel('Time from nosepoke onset (s)',fontsize=axislabel_size)
    pcax = plt.gca()
    # Changepoint lines
    if threshold == 'max':
        plt.axhline(np.argmax(tVec), linestyle='-.', linewidth=6, color='k')
    else:
        pass
        plt.hlines(np.nonzero(tVec>threshold)[0],0,max(durations),'k',linestyle='-.',linewidth=6)

    # Changepoint lateral
    if cpax:
        cpax = f.add_axes([1,.045,1,.945])
        print(trialNumber)
        cpax.plot(tVec,trialNumber,'k'); plt.yticks([]);
        plt.ylim([0,len(durations)]); plt.xlabel('Odds',fontsize=axislabel_size)
        pos1 = pcax.get_position() # get the original position
        pos2 = [pos1.x0  + .85, pos1.y0,  pos1.width, pos1.height]
        cpax.set_position(pos2) # set a new position
    # Kernel density
    if kde is not False:
        kdax = f.add_axes([.06,1,.925,1])
        pos1 = pcax.get_position() # get the original position
        if reverse:
            pos2 = [pos1.x0 , pos1.y0 - .85,  pos1.width, pos1.height]
            plt.xlabel('Time from nosepoke onset (s)',fontsize=axislabel_size)
        else:
            pos2 = [pos1.x0 , pos1.y0 + .85,  pos1.width, pos1.height]
        plt.ylabel('Probability density', fontsize=axislabel_size)
        kdax.set_position(pos2) # set a new position
    if kde in  ['fl','firstlast']:
        first = durations[:kdeN]
        last = durations[-kdeN:]

        sns.kdeplot(first,label='First %d trials'%kdeN,color='b',linewidth=2);
        sns.kdeplot(last, label = 'Last %d trials'%kdeN,color = 'magenta',linewidth=4);
        plt.xlim([0,tmax]);plt.ylim([0,0.8])

        pcax.axhline(kdeN,linestyle='--',color='b',linewidth=4);
        pcax.arrow(1,kdeN,0,-30, head_width=0.05, head_length=10, fc='b', ec='b',linewidth=2)
        pcax.arrow(4,kdeN,0,-30, head_width=0.05, head_length=10, fc='b', ec='b',linewidth=2)

        pcax.axhline(len(durations)-kdeN,linestyle='--',color = 'magenta',linewidth=4)
        pcax.arrow(1,len(durations)-kdeN,0,30, head_width=0.05, head_length=10, fc='magenta', ec='magenta',linewidth=2)
        pcax.arrow(4,len(durations)-kdeN,0,30, head_width=0.05, head_length=10, fc='magenta', ec='magenta',linewidth=2)

    elif kde in ['cp','fromcp']:
        first = durations[np.argmax(tVec)-kdeN:np.argmax(tVec)]
        last = durations[np.argmax(tVec):np.argmax(tVec)+kdeN]

        sns.kdeplot(first,label='First %d trials'%kdeN,color='b',linewidth=2);
        sns.kdeplot(last, label = 'Last %d trials'%kdeN,color = 'magenta',linewidth=4);
        plt.xlim([0,tmax]);plt.ylim([0,0.8])

        pcax.axhline(np.argmax(tVec)-kdeN,linestyle='--',color='b',linewidth=4);
        pcax.arrow(1,np.argmax(tVec)-kdeN,0,30, head_width=0.05, head_length=10, fc='b', ec='b',linewidth=2)
        pcax.arrow(4,np.argmax(tVec)-kdeN,0,30, head_width=0.05, head_length=10, fc='b', ec='b',linewidth=2)

        pcax.axhline(np.argmax(tVec)+kdeN,linestyle='--',color = 'magenta',linewidth=4)
        pcax.arrow(1,np.argmax(tVec)+kdeN,0,-30, head_width=0.05, head_length=10, fc='magenta', ec='magenta',linewidth=2)
        pcax.arrow(4,np.argmax(tVec)+kdeN,0,-30, head_width=0.05, head_length=10, fc='magenta', ec='magenta',linewidth=2)


    #plt.suptitle('Behavior evolution\n in one session',y = 1.34, fontsize=axislabel_size*1.1, x=1.35)
#    plt.tight_layout()
