
from mtuq.misfit.waveform import WaveformMisfit

from mtuq.misfit.polarity import PolarityMisfit, polarities_from_dict

from mtuq.misfit.relative_amplitude import (
    PolarityRelPampMisfit,
    GeometryList,
    relative_pamp_polarity_from_dict,
    relative_pol_pamp_spamp_from_dict,
)

#
# for backward compatibility
#
Misfit = WaveformMisfit

