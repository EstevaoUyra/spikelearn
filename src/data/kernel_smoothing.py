"""
From the epoched_spikes, derives four smoothed datasets, according to the
following parameters:

(kernel) sigma : 100, 50
    For comparison of the default (100) with the smaller (50)

histogram binsize : 100, 20
    The first one (100) for default applications and analysis,
    the second one (20) for visualization purposes.

The dataset names will accord to the following:
    {100, 100} wide_smoothed
    { 50, 100} narrow_smoothed
    {100,  20} wide_smoothed_viz
    { 50,  20} narrow_smoothed_viz

"""
import sys
sys.path.append('.')
from spikelearn.data.preprocessing import kernel_smooth
from spikelearn.data import io, SHORTCUTS
import pandas as pd

BASELINE = -500
DSET_PARAMS = {'wide_smoothed' : { 'sigma' : 100,
                                  'bin_size' : 100},

              'narrow_smoothed' : { 'sigma' : 50,
                                    'bin_size' : 100},

              'wide_smoothed_viz' : { 'sigma' : 100,
                                      'bin_size' : 20},

              'narrow_smoothed_viz' : { 'sigma' : 50,
                                        'bin_size' : 20}
                }


for rat_label in SHORTCUTS:
    epoched = io.load(rat_label, 'epoched_spikes')
    epoched = epoched.groupby('is_selected').get_group(True)
    epoched = epoched.reset_index().set_index(['trial','unit'])
    for dset_name, params in DSET_PARAMS.items():
        # Create dataset and add identifiers
        smoothed_dataset = pd.DataFrame(index=epoched.index)

        # Put firing_rates with motor activity and baseline
        f = lambda x: kernel_smooth( 1000*x['with_baseline'], **params,
                                      edges=(BASELINE, 1000*x.duration))
        out = epoched.reset_index().apply(f, axis=1)
        out = pd.DataFrame(out.tolist(), index = epoched.index,
                            columns = ['full', 'full_times'])
        smoothed_dataset = smoothed_dataset.join(out)

        # Then without motor activity (200, -300)
        f = lambda x: kernel_smooth( 1000*x['time'], **params,
                                      edges=(200, 1000*x.duration-300))
        out = epoched.reset_index().apply(f, axis=1)
        out = pd.DataFrame(out.tolist(), index = epoched.index,
                            columns = ['cropped', 'cropped_times'])
        smoothed_dataset = smoothed_dataset.join(out)

        behav = io.load(rat_label, 'behav_stats')
        smoothed_dataset = smoothed_dataset.reset_index().set_index('trial').join(behav)

        # At last save the data
        io.save(data = smoothed_dataset.reset_index().set_index(['trial','unit']),
                base_label = rat_label,
                dataset_name = dset_name)
