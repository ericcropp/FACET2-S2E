"""
FACET2_S2E: Simulation tools for FACET-II start-to-end beam dynamics
"""

from .UTILITY_quickstart import (
    # Core initialization and tracking
    initializeTao,
    trackBeam,
    trackBeamHelper,
    getBeamAtElement,
    ballisticPropagation,
    
    # Beam manipulation and analysis
    nudgeMacroparticleWeights,
    getDriverAndWitness,
    writeBeam,
    makeBeamActiveBeamFile,
    centerBeam,
    collimateBeam,
    sliceBeam,
    getSingleBeamSlice,
    
    # Statistical analysis
    smallestInterval,
    smallestIntervalImpliedSigma,
    smallestIntervalImpliedEmittance,
    smallestIntervalImpliedEmittanceModelFunction,
    emittance,
    getBeamSpecs,
    
    # Matrix and lattice operations
    displayMatrix,
    getMatrix,
    getMatrixLEGACY,
    setLatticeAndGetMatrix,
    
    # Beam modulation
    addLHmodulation,
    calcBMAG,
    
    # Configuration utilities
    loadConfig,
    
    # Optimization and correction
    launchTwissCorrection,
    launchTwissCorrectionObjective,
    generalizedEmittanceSolver,
    generalizedEmittanceSolverObjective,
    
    # Lattice configuration
    disableAutoQuadEnergyCompensation,
    disableAutoMagnetEnergyCompensation,
    applyOtherConfig,
    
    # Helper utilities
    sortIndices,
)


from .UTILITY_QPAD import (
    # QPAD visualization and analysis
    plotInteractiveQPADFigure,
    saveAllQPADFigures,
    plotPlasmaProfile,
)

# Import commonly used functions from other utility modules
from .UTILITY_plotMod import plotMod, slicePlotMod, floorplanPlot
from .UTILITY_linacPhaseAndAmplitude import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto
from .UTILITY_setLattice import (
    setLattice, 
    getBendkG, getQuadkG, getSextkG,
    setBendkG, setQuadkG, setSextkG,
    setXOffset, setYOffset,
    getKickerkG, setKickerkG,
    getBendGeVc, setBendGeVc
)
from .UTILITY_finalFocusSolver import finalFocusSolver

__all__ = [
    # Core initialization and tracking
    'initializeTao',
    'trackBeam',
    'trackBeamHelper',
    'getBeamAtElement',
    'ballisticPropagation',
    
    # Beam manipulation and analysis
    'nudgeMacroparticleWeights',
    'getDriverAndWitness',
    'writeBeam',
    'makeBeamActiveBeamFile',
    'centerBeam',
    'collimateBeam',
    'sliceBeam',
    'getSingleBeamSlice',
    
    # Statistical analysis
    'smallestInterval',
    'smallestIntervalImpliedSigma',
    'smallestIntervalImpliedEmittance',
    'smallestIntervalImpliedEmittanceModelFunction',
    'emittance',
    'getBeamSpecs',
    
    # Matrix and lattice operations
    'displayMatrix',
    'getMatrix',
    'getMatrixLEGACY',
    'setLatticeAndGetMatrix',
    
    # Beam modulation
    'addLHmodulation',
    'calcBMAG',
    
    # Configuration utilities
    'loadConfig',
    
    # Optimization and correction
    'launchTwissCorrection',
    'launchTwissCorrectionObjective',
    'generalizedEmittanceSolver',
    'generalizedEmittanceSolverObjective',
    
    # Lattice configuration
    'disableAutoQuadEnergyCompensation',
    'disableAutoMagnetEnergyCompensation',
    'applyOtherConfig',
    
    # Helper utilities
    'sortIndices',
    
    # QPAD visualization and analysis
    'plotInteractiveQPADFigure',
    'saveAllQPADFigures',
    'plotPlasmaProfile',
    
    # Plotting utilities
    'plotMod',
    'slicePlotMod',
    'floorplanPlot',
    
    # Linac configuration
    'getLinacMatchStrings',
    'setLinacPhase',
    'setLinacGradientAuto',
    
    # Lattice element control
    'setLattice',
    'getBendkG', 'getQuadkG', 'getSextkG',
    'setBendkG', 'setQuadkG', 'setSextkG',
    'setXOffset', 'setYOffset',
    'getKickerkG', 'setKickerkG',
    'getBendGeVc', 'setBendGeVc',
    
    # Final focus optimization
    'finalFocusSolver',
]

__version__ = '0.1.0'
