from spikelearn.models.mahalanobis import similarity
from spikelearn import io, frankenstein

DSETS = ['wide_smoothed_viz', 'no_smoothing_viz', 'no_smoothing_norm_viz', 'no_smoothing']


for dset in DSETS:
    # Merging rats
    DR = [io.load(label, dset) for label in SHORTCUTS['groups']['DRRD']]
    EZ_d1 = [io.load(label, dset) for label in SHORTCUTS['groups']['EZ'] if '_2' not in label]
    EZ_d2 = [io.load(label, dset) for label in SHORTCUTS['groups']['EZ'] if '_2' in label]

    sp_pfc = frankenstein(DR, _min_duration=1.5, is_selected=True, is_tired=False)
    ez_pfc_d1 = frankenstein(EZ_d1, _min_duration=1.5, _min_quality=0, area='PFC')
    ez_pfc_d2 = frankenstein(EZ_d2, _min_duration=1.5, _min_quality=0, area='PFC')
    ez_str_d1 = frankenstein(EZ_d1, _min_duration=1.5, _min_quality=0, area='STR')
    ez_str_d2 = frankenstein(EZ_d2, _min_duration=1.5, _min_quality=0, area='STR')

    merged_rats = [sp_pfc, ez_pfc, ez_str]
    for rat_label, df in zip(['sp_pfc', 'ez_pfc_d1', 'ez_pfc_d2', 'ez_str_d1', 'ez_str_d2'],
                             [sp_pfc,   ez_pfc_d1,   ez_pfc_d2, ez_str_d1, ez_str_d2]):
        