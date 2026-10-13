"""Deprecated location of UTILITY_setLattice, kept so old imports keep working.

The code moved to:
    FACET2_S2E.simulators.bmad.lattice.set_lattice
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .simulators.bmad.lattice.set_lattice import (
    setLattice,
    setLinacsHelper,
    setBendkG,
    getBendkG,
    setQuadkG,
    getQuadkG,
    setSextkG,
    getSextkG,
    setXOffset,
    setYOffset,
    setKickerkG,
    getKickerkG,
    setBendGeVc,
    getBendGeVc,
    setAllInjectorQuads,
    setAllFinalFocusQuads,
    setAllWChicaneBends,
    setAllWChicaneQuads,
    setAllWChicaneSextupoles,
    setAllWChicaneMovers,
    setAllFinalFocusKickers,
    setXTCAV,
    setWChicaneLaunchQuads,
)
from .simulators.bmad.lattice.linac import (
    getLinacMatchStrings,
    setLinacPhase,
    setLinacGradientAuto,
)

warnings.warn(
    "FACET2_S2E.UTILITY_setLattice is deprecated and will be removed; "
    "import from FACET2_S2E.simulators.bmad.lattice.set_lattice or use FACET2_S2E.<name> "
    "instead.",
    FutureWarning,
    stacklevel=2,
)
