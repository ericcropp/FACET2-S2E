"""Deprecated location of UTILITY_quickstart, kept so old imports keep working.

The code was split into:
    FACET2_S2E.simulators.bmad.core
    FACET2_S2E.beam.manipulation
    FACET2_S2E.beam.analysis
    FACET2_S2E.simulators.bmad.lattice.optics
    FACET2_S2E.beam.microbunching
    FACET2_S2E.simulators.bmad.config
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .simulators.bmad.core import (
    initializeTao,
    applyBMADCollectiveEffectSettings,
    trackBeam,
    trackBeamHelper,
    getBeamAtElement,
    writeBeam,
    makeBeamActiveBeamFile,
)
from .beam.manipulation import (
    ballisticPropagation,
    nudgeMacroparticleWeights,
    getDriverAndWitness,
    centerBeam,
    collimateBeam,
    sortIndices,
    sliceBeam,
    getSingleBeamSlice,
    modifyAndSaveInputBeam,
)
from .beam.analysis import (
    calcBMAG,
    smallestInterval,
    smallestIntervalImpliedSigma,
    smallestIntervalImpliedEmittanceModelFunction,
    smallestIntervalImpliedEmittance,
    emittance,
    getBeamSpecs,
    generalizedEmittanceSolverObjective,
    generalizedEmittanceSolver,
)
from .simulators.bmad.lattice.optics import (
    displayMatrix,
    getMatrix,
    getMatrixLEGACY,
    setLatticeAndGetMatrix,
    launchTwissCorrectionObjective,
    launchTwissCorrection,
)
from .beam.microbunching import (
    addLHmodulation,
)
from .simulators.bmad.config import (
    loadConfig,
    disableAutoQuadEnergyCompensation,
    disableAutoMagnetEnergyCompensation,
    applyOtherConfig,
)
from .plotting.phase_space import (
    plotMod,
    slicePlotMod,
)
from .plotting.floorplan import (
    floorplanPlot,
)
from .simulators.bmad.lattice.linac import (
    getLinacMatchStrings,
    setLinacPhase,
    setLinacGradientAuto,
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
    setKickerkG,
    getKickerkG,
    setBendGeVc,
    getBendGeVc,
)
from .simulators.impact import (
    runImpact,
)
from .simulators.bmad.lattice.final_focus import (
    finalFocusSolver,
)
from .simulators.qpad import (
    QPAD_sim,
    run_QPAD,
)

# filePathGlobal is not re-exported: it is module state of
# FACET2_S2E.simulators.bmad.core, and a copy here would be stale.
# Use tao.filePathGlobal.

warnings.warn(
    "FACET2_S2E.UTILITY_quickstart is deprecated and will be removed; "
    "import from FACET2_S2E.simulators.bmad.core, FACET2_S2E.beam.manipulation, "
    "FACET2_S2E.beam.analysis, FACET2_S2E.simulators.bmad.lattice.optics, "
    "FACET2_S2E.beam.microbunching, FACET2_S2E.simulators.bmad.config or use "
    "FACET2_S2E.<name> instead.",
    FutureWarning,
    stacklevel=2,
)
