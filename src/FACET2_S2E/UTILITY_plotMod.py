"""Deprecated location of UTILITY_plotMod, kept so old imports keep working.

The code was split into:
    FACET2_S2E.plotting.phase_space
    FACET2_S2E.plotting.floorplan
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .plotting.phase_space import (
    plotMod,
    slicePlotMod,
)
from .plotting.floorplan import (
    colorlist,
    colorlist2,
    floorplan_sorter,
    floorplan_patches,
    floorplan_plot_partial,
    format_longitudinal_plot,
    floorplanPlot,
)

warnings.warn(
    "FACET2_S2E.UTILITY_plotMod is deprecated and will be removed; import "
    "from FACET2_S2E.plotting.phase_space, FACET2_S2E.plotting.floorplan or"
    " use FACET2_S2E.<name> instead.",
    FutureWarning,
    stacklevel=2,
)
