"""Deprecated location of UTILITY_impact, kept so old imports keep working.

The code moved to:
    FACET2_S2E.simulators.impact
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .simulators.impact import (
    runImpact,
    update_impact,
    update_distgen,
)

warnings.warn(
    "FACET2_S2E.UTILITY_impact is deprecated and will be removed; import "
    "from FACET2_S2E.simulators.impact or use FACET2_S2E.<name> instead.",
    FutureWarning,
    stacklevel=2,
)
