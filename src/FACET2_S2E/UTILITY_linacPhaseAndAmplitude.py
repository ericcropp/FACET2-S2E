"""Deprecated location of UTILITY_linacPhaseAndAmplitude, kept so old imports keep working.

The code moved to:
    FACET2_S2E.simulators.bmad.lattice.linac
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .simulators.bmad.lattice.linac import (
    getLinacMatchStrings,
    getEnergyChangeFromElements,
    setLinacGradientAuto,
    setLinacPhase,
    matchStringWrapper,
    printArbValues,
)

warnings.warn(
    "FACET2_S2E.UTILITY_linacPhaseAndAmplitude is deprecated and will be "
    "removed; import from FACET2_S2E.simulators.bmad.lattice.linac or use FACET2_S2E.<name>"
    " instead.",
    FutureWarning,
    stacklevel=2,
)
