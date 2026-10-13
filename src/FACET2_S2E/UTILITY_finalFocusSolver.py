"""Deprecated location of UTILITY_finalFocusSolver, kept so old imports keep working.

The code moved to:
    FACET2_S2E.simulators.bmad.lattice.final_focus
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .simulators.bmad.lattice.final_focus import (
    finalFocusSolverObjective,
    finalFocusSolver,
)
from .simulators.bmad.lattice.set_lattice import (
    setLattice,
    getBendkG,
    getQuadkG,
    getSextkG,
    setBendkG,
    setQuadkG,
    setSextkG,
    setXOffset,
    setYOffset,
)

warnings.warn(
    "FACET2_S2E.UTILITY_finalFocusSolver is deprecated and will be removed;"
    " import from FACET2_S2E.simulators.bmad.lattice.final_focus or use FACET2_S2E.<name> "
    "instead.",
    FutureWarning,
    stacklevel=2,
)
