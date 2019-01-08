import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append('/home/tevo/Documents/UFABC/Spikes')

from spikeHelper.filters import kernelSmooth,binarize
from spikeHelper.loadSpike import Rat

sns.set_style('darkgrid')



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
