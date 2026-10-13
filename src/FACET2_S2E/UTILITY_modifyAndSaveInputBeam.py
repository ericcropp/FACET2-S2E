"""Deprecated location of UTILITY_modifyAndSaveInputBeam, kept so old imports keep working.

The code moved to:
    FACET2_S2E.beam.manipulation
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .beam.manipulation import (
    modifyAndSaveInputBeam,
)

warnings.warn(
    "FACET2_S2E.UTILITY_modifyAndSaveInputBeam is deprecated and will be "
    "removed; import from FACET2_S2E.beam.manipulation or use "
    "FACET2_S2E.<name> instead.",
    FutureWarning,
    stacklevel=2,
)
