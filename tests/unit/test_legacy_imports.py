"""
The old UTILITY_*.py import paths still work (with a FutureWarning) after the
move into subpackages, and give the same objects as the new locations.
"""

import importlib
import sys

import pytest

import FACET2_S2E


LEGACY_MODULES = [
    "UTILITY_quickstart",
    "UTILITY_setLattice",
    "UTILITY_linacPhaseAndAmplitude",
    "UTILITY_plotMod",
    "UTILITY_finalFocusSolver",
    "UTILITY_modifyAndSaveInputBeam",
    "UTILITY_impact",
    "UTILITY_QPAD",
    "UTILITY_QPAD_PICMI",
]


def import_fresh(name):
    sys.modules.pop(f"FACET2_S2E.{name}", None)
    return importlib.import_module(f"FACET2_S2E.{name}")


@pytest.mark.parametrize("name", LEGACY_MODULES)
def test_legacy_module_warns(name):
    with pytest.warns(FutureWarning, match=f"FACET2_S2E.{name} is deprecated"):
        import_fresh(name)


@pytest.mark.parametrize("name", LEGACY_MODULES)
def test_legacy_names_are_new_objects(name):
    """Every public name exported at the top level is the same object via the old path."""
    with pytest.warns(FutureWarning):
        legacy = import_fresh(name)
    shared = [n for n in vars(legacy) if not n.startswith("_") and n in FACET2_S2E.__all__]
    for n in shared:
        assert getattr(legacy, n) is getattr(FACET2_S2E, n), n


def test_legacy_plotmod_is_phase_space_plotmod():
    """UTILITY_plotMod.plotMod is the single (merged) plotMod."""
    with pytest.warns(FutureWarning):
        legacy = import_fresh("UTILITY_plotMod")
    assert legacy.plotMod is FACET2_S2E.plotMod


def test_legacy_quickstart_examples():
    """Imports used by the old tests and notebooks."""
    sys.modules.pop("FACET2_S2E.UTILITY_quickstart", None)
    with pytest.warns(FutureWarning):
        from FACET2_S2E.UTILITY_quickstart import (  # noqa: F401
            initializeTao, trackBeam, getBeamAtElement, centerBeam, getBeamSpecs,
            getMatrix, loadConfig, addLHmodulation, setLattice, finalFocusSolver,
            runImpact, run_QPAD, QPAD_sim, modifyAndSaveInputBeam, plotMod,
        )
