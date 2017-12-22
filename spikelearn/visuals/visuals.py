import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def raster_plot(spike_trains, time='time', yaxis='trial', xlim=None,
        subplots=None, hue=None, n_cols='auto', n_rows='auto', title=None,
        palette='deep', density=False, n_yticks=5, is_list=False,
        kde_kwargs=None, fig_kwargs=None, **kwargs):
    """
    Parameters
    ----------
    spike_trains : DataFrame
        Each row contains identifier variables, and the time column should
        contain a single time or a list of times for each row.
        Note: if the format is list of times, the function will take longer,
        and this must be flagged at is_list

    time : string, optional, default: 'time'
        DataFrame column index containing the times or time lists.
        If not 'time', the name of the column is added to the xlabel
        default label is 'Time (ms)', and it turns ('Time ({})').format(time)

    yaxis : string, optional, default: 'trial'
        Variable that defines the vertical position of the row's data.

    xlim: tuple, optional
        Limits of the horizontal axis of each subplot

    subplots : string, optional
        Defines which variable is unique in each subplot

    hue : string, optional
        Defines which variable is unique in each color

    palette : list or str, optional
        list of colors or name of seaborn color_palette. Defaults to 'Set3'

    density : bool or str, optional
        Whether to show the time density of spikes on top of each subplot.
        If str, selects column to groupby and plot multiple densities.
        If 'hue', plots a density for each color.

    n_yticks : int, optional, default: 5
        Number of ticks on the vertical axis of each subplot

    is_list : bool, optional, default: False
        Flags the rows as containing list of times, instead of values.
        Making the transformation takes time, and it is preferable to do
        this transformation outside the plotting function.



    Notes
    -----


    """
    ## Mechanics
    # Make local copy, verticalyzing if necessary
    if is_list:
        data = verticalize_df(epc,time)
    else:
        data = spike_trains.copy()
    # Make mutables non-persistent
    if kde_kwargs is None:
        kde_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}
    # Auxilliary columns
    if subplots is None:
        subplots = 'aux_subplots'
        data['aux_subplots'] = 1
    if hue is None:
        hue = 'aux_hue'
        data['aux_hue'] = 1
    if density is True:
        density = 'aux_density'
        data['aux_density'] = 1
    ## Define plot aesthetics
    # Geometry
    n_subplots = data[subplots].unique().shape[0]
    assert n_cols is 'auto' or n_rows is 'auto'
    if n_cols is 'auto':
        n_cols = np.ceil(np.sqrt(n_subplots))
    if n_rows is 'auto':
        n_rows = np.ceil(n_subplots/n_cols)
    if 'figsize' not in fig_kwargs:
        fig_kwargs['figsize'] = (3*n_cols, 4*n_rows)
        print(n_cols,n_rows)
    else:
        print('test')
    fig = plt.figure(**fig_kwargs)
    # Colors
    n_hue = data[hue].unique().shape[0]
    if type(palette) is str:
        palette = sns.palettes.color_palette(palette, n_hue)
    else:
        assert type(palette) is list
        assert len(palette) is n_hue
    # Text
    xlabel = 'Time ({})'.format('ms' if time=='time' else time)
    ylabel = yaxis
    ## Plotting per se
    # Each subplot
    for sub_i, sub_l in enumerate(data[subplots].unique()):
        ax = fig.add_subplot(n_rows, n_cols, sub_i+1)
        if density:
            axd = ax.twinx()
        sub_data = data.groupby(subplots).get_group(sub_l)
        # Each color
        for hue_i, hue_l in enumerate(sub_data[hue].unique()):
            hue_data = sub_data.groupby(hue).get_group(hue_l)
            ax.scatter(hue_data[time], hue_data[yaxis], color=palette[hue_i], **kwargs)
            # Density
            if density:
                sns.kdeplot(hue_data[time].values, ax=axd, color=palette[hue_i], **kde_kwargs)
        plt.sca(ax)
        ticks = np.linspace(sub_data[yaxis].min(),sub_data[yaxis].max(), n_yticks).astype(int)
        plt.yticks(ticks,ticks)
        plt.title(sub_l)
        plt.ylabel(ylabel); plt.xlabel(xlabel)
        if xlim is not None: plt.xlim(xlim)
        if title is not None:
            plt.suptitle(title,y=1.01, x=.5,fontsize=16)
    plt.tight_layout()


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
