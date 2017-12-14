import numpy as np
import pandas as pd



class Neuron():
    def __init__(self, spike_times, **kwargs):
        self.spike_times = spike_times
        self.spike_attributes = pd.DataFrame(kwargs)

    def firing_rate_estimate(self, window):
        return


class NeuronGroup(Neuron):


class Trial(NeuronGroup):
    def __init__(self):
class
