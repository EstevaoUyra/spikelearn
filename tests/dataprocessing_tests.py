import unittest
import numpy.testing as npt
import sys
sys.path.append('../')
from spikelearn import data

class KernelSmoothingTest(unittest.TestCase):
    """Unit tests for kernel smoothing"""

    def setUp(self):
        self.area = .9973 # Three std's for each side

    def test_borders(self):
        data.preprocessing.kernel_smooth()
        # Spikes at equal intervals of 10ms
        N=100
        spike_times = (10*np.arange(N)); fr = np.ones(N)
        smoothed = data.preprocessing.kernel_smooth(spike_times, 100, (0,1000)
                                                        10, 10)[0]
        npt.assertAlmostEqual(self.area*fr, smoothed, 4)

    def test_size(self):
        pass

    def test_edges(self):
        pass





if __name__ == '__main__':
    unittest.main()
