"""Deprecated location of UTILITY_QPAD_PICMI, kept so old imports keep working.

The code moved to:
    FACET2_S2E.simulators.qpad_picmi
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .simulators.qpad_picmi import (
    codename,
    to_scientific_notation,
    constants,
    Neutral,
    Species,
    MultiSpecies,
    GaussianBunchDistribution,
    OpenPMDFileDistribution,
    UniformDistribution,
    PiecewiseDistribution,
    AnalyticDistribution,
    ParticleListDistribution,
    ConstantAppliedField,
    AnalyticAppliedField,
    Mirror,
    ElectromagneticSolver,
    ElectrostaticSolver,
    Cartesian1DGrid,
    Cartesian2DGrid,
    Cartesian3DGrid,
    CylindricalGrid,
    FileLayout,
    PseudoRandomLayout,
    GriddedLayout,
    Simulation,
    FieldDiagnostic,
    ElectrostaticFieldDiagnostic,
    LabFrameParticleDiagnostic,
    LabFrameFieldDiagnostic,
    ParticleDiagnostic,
    GaussianLaser,
    LaserAntenna,
    BinomialSmoother,
    normalize_math_func,
    format_decimal,
    construct_bounds,
)

warnings.warn(
    "FACET2_S2E.UTILITY_QPAD_PICMI is deprecated and will be removed; "
    "import from FACET2_S2E.simulators.qpad_picmi or use FACET2_S2E.<name> "
    "instead.",
    FutureWarning,
    stacklevel=2,
)
