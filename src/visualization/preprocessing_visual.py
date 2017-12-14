import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/home/tevo/Documents/UFABC/Spikes')
import os
os.chdir('/home/tevo/Documents/UFABC/Spikes')

from spikeHelper.filters import kernelSmooth,binarize
from spikeHelper.loadSpike import Rat

r = Rat(8)
sns.set_style('darkgrid')

def pretty_neurons(N_neurons=8, sigma=0, spike_vector=None,rate = 1, tmax=10, figsize=(26,16), superimpose = 0,
                    palette='Set2',binpalette='plasma',bin_fill=None,bin_sep=None,alpha_multiplier = 1/20):
    fig=plt.figure(figsize=figsize); s=None

    ax=plt.subplot(1,1,1)
    neuroncolors = sns.palettes.color_palette(palette,N_neurons)
    if spike_vector is None:
        spike_vector = np.random.exponential(1/rate,(N_neurons,int(1.5*tmax*rate))).cumsum(axis=1)


    ax.hlines((1-superimpose)*np.arange(N_neurons), -.4,11,colors=neuroncolors);

    if sigma in [0,1,2,100,1000]:
        if sigma==2:
            linewidth=7
        else:
            linewidth=3
        for i in range(N_neurons):
            ax.vlines(spike_vector[i,:],(1-superimpose)*i,(1-superimpose)*i+.7,colors=neuroncolors[i],linewidth=linewidth)

    nbins = (tmax+2)*1000
    s=np.zeros((N_neurons,nbins))
    if sigma >= 2:
        for i in range(N_neurons):
            v, t = np.histogram(spike_vector[i,:], bins=nbins,range=(-1,tmax+1))
            s[i,:] = kernelSmooth(v, sigma)
            plt.plot(t[:-1],s[i,:]*2.6*(sigma**(2/3))+(1-superimpose)*i,linewidth=5,color = neuroncolors[i])


    bincmap = sns.palettes.color_palette(binpalette,tmax)
    if bin_sep is not None:
        bincmap = sns.palettes.color_palette('plasma',tmax+1)
        plt.vlines(np.arange(0,bin_sep+1),-1,N_neurons,linewidth=8,colors=bincmap,linestyle='-')

    if bin_fill is not None:
        for j in range(bin_fill[0]):
            for i in range(bin_fill[1]):
                plt.fill_between([i,i+1],j,j+1,color = neuroncolors[j],alpha=min(np.sqrt(binarize(s[j,:],1000)[1+i]*alpha_multiplier),1))

    ax.set_yticks((1-superimpose)*np.arange(N_neurons)+(1-superimpose)*.5)
    ax.set_yticklabels(['Neuron %d'%i for i in range(1,N_neurons+1)],fontsize=12)
    ax.set_xlim(-.4,tmax+.4); ax.set_ylim(-0.2,max(s[-1,:].max()+N_neurons,(1-superimpose)*N_neurons+.2)); #ax.set_ylim(-0.2,(1-superimpose)*N_neurons+.2);
    return fig, spike_vector, s

if __name__ == '__main__':
    # Original
    fig, spike_vector, s = pretty_neurons(rate=8);
    plt.savefig('Figures/preprocessing/original.png',transparent=True,dpi=200)
    plt.close(fig)
    # Grow smoothing
    for i in range(2,101,7):
        fig, spike_vector, s =pretty_neurons(spike_vector=spike_vector,sigma=i);
        plt.savefig('Figures/preprocessing/smooth_%d.png'%i,transparent=True,dpi=200)
        plt.close(fig)
    # Cut bins
    for binsep in range(11):
        fig, spike_vector, s = pretty_neurons(sigma=100,spike_vector=spike_vector,bin_sep=binsep,
        alpha_multiplier=1/15)
        plt.savefig('Figures/preprocessing/binsep_%d.png'%binsep,transparent=True,dpi=200)
        plt.close(fig)

    # Make values
    for time in range(1,11):
        fig, spike_vector, s = pretty_neurons(sigma=100,spike_vector=spike_vector,bin_sep=binsep,
        alpha_multiplier=1/15,bin_fill=(1,time))
        plt.savefig('Figures/preprocessing/timebin%d.png'%time,transparent=True,dpi=200)
        plt.close(fig)

    for neuron in range(1,9):
        fig, spike_vector, s = pretty_neurons(sigma=100,spike_vector=spike_vector,bin_sep=binsep,
        alpha_multiplier=1/15,bin_fill=(neuron,10))
        plt.savefig('Figures/preprocessing/neuronbin%d.png'%neuron,transparent=True,dpi=200)
        plt.close(fig)
