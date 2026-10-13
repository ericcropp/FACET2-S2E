"""Deprecated location of UTILITY_QPAD, kept so old imports keep working.

The code moved to:
    FACET2_S2E.simulators.qpad
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .simulators.qpad import (
    QPAD_sim,
    generate_Li_oven_profile,
    eq,
    get,
    present,
    run_QPAD,
    combined_dset_mod,
    plotQPAD,
    saveAllQPADFigures,
    plotInteractiveQPADFigure,
    plotPlasmaProfile,
)

warnings.warn(
    "FACET2_S2E.UTILITY_QPAD is deprecated and will be removed; import from"
    " FACET2_S2E.simulators.qpad or use FACET2_S2E.<name> instead.",
    FutureWarning,
    stacklevel=2,
)
