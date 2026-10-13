# FACET2-S2E Test Suite

Comprehensive test coverage for the FACET2-S2E package with unit, integration, and system tests.

## Test Organization

```
tests/
├── unit/
│   ├── test_quickstart.py
│   ├── test_initialize_tao.py
│   ├── test_legacy_imports.py
│   ├── test_beam_generation.py
│   ├── test_beam_editing.py
│   ├── test_microbunching.py
│   └── test_plotting.py
├── integration/
│   ├── test_examples_integration.py
│   └── test_bmad_workflow.py
├── smoke/                         (4 tests)
│   └── test_full_s2e.py
├── system/                        (4 tests)
│   └── test_system.py
├── conftest.py
├── __init__.py
└── README.md
```

### Unit Tests - `unit/`
- **`test_quickstart.py`**: Core utility, mathematical operations, data processing, and advanced functions requiring Tao/bmad integration
- **`test_initialize_tao.py`**: Tao initialization and configuration
- **`test_legacy_imports.py`**: The deprecated `UTILITY_*` module paths still import, with a `FutureWarning`
- **`test_beam_generation.py`**: `make_bunch` (6D covariance, Cholesky, zero and singular covariances, flat profile) and the `make_simple_bunch*` wrappers
- **`test_beam_editing.py`**: `edit_bunch_parameters(_from_PG)` (sizes, correlations, means, Twiss, no side effects on arguments), `cut_length`, `modifyInputBeamSimple`
- **`test_microbunching.py`**: Spectrum binning, `get_spectrum`, band analysis, microbunching gain, `make_modulated_bunch`
- **`test_plotting.py`**: `plotMod` (standalone, embedded in a figure, `z_from_t`), `print_result*`, `make_a_plot`, `enable_plt_styling`

### Integration Tests - `integration/`
- **`test_bmad_workflow.py`**: The Bmad/DAQ workflow with a **real Tao** (no mocks), in a temporary copy of the
  repository layout so that nothing is written into the repository:
  - Lattice queries: `get_element_array`, `get_rij`, `get_tijk` (checked against tracking), `make_comparison_dz_2nd_order`
  - Magnets and settings: sextupole offsets, `applyBMADCollectiveEffectSettings`
  - Energy: `tune_to_P0Cs`, dipole field saving/restoring, beam-based energy feedback (`edit_energy_based_on_beam_*`)
  - Runs: `set_beam`, `run_initialized_sim*`
  - DAQ database edits with a fake `DATASET`: cavities, quads (with boost PVs), sextupoles, movers, correctors
  - `get_tao_from_experiment` with a generated beam, a beam file, and the DAQ branch
  - Tests that take more than ~10 s are marked `slow`
- **`test_examples_integration.py`**: Real function calls from example notebooks with mocked external dependencies
  - Jitter study functions from `Example - Jitter study.py`
  - Beam operations: ballistic propagation, driver/witness separation, collimation
  - Utility functions: sorting, slicing, emittance calculation
  - Configuration loading and I/O operations

### Smoke Tests (4 tests) - `smoke/`
- **`test_full_s2e.py`**: Full start-to-end workflow validation
  - Real test: Runs complete S2E with IMPACT-T, Bmad, and QPAD (minimal particles)
  - Mocked tests: Verify simulator calls without running actual simulations
  - Validates ParticleGroup creation and beam propagation

### System Tests (4 tests) - `system/`
- **`test_system.py`**: End-to-end simulator validation (100+ particles)
  - IMPACT-T beam generation
  - Bmad tracking and Twiss parameters
  - QPAD plasma simulation

**Total: 210+ tests**

## Quick Start

```bash
# Run all tests
pytest tests/

# Run only fast tests (skip slow integration/system tests)
pytest tests/ -m "not slow"

# Run specific category
pytest tests/unit -v              # Unit tests
pytest tests/integration -v       # Integration tests
pytest tests/smoke -v             # Smoke tests
pytest tests/system -v            # System tests

# Run with coverage
pytest tests/ --cov=src/FACET2_S2E --cov-report=html
```

## Test Markers

Available markers for filtering tests:

- `slow`: Long-running tests (integration, system, smoke)
- `system`: Simulator validation tests (IMPACT-T, Bmad, QPAD)
- `smoke`: End-to-end workflow tests
- `integration`: Example code integration tests
- `qpad`: QPAD-specific tests

```bash
pytest tests/ -m slow           # Only slow tests
pytest tests/ -m system         # Only system tests
pytest tests/ -m "not slow"     # Skip slow tests
```

## Test Design

### Unit Tests
- ✓ Mock external dependencies (Tao, file I/O)
- ✓ Test mathematical correctness
- ✓ Fast execution (< 1 second each)
- ✓ No special environment required

### Integration Tests
- ✓ Call **real functions** from examples and FACET2_S2E
- ✓ Mock slow/external operations (Tao commands, file I/O)
- ✓ Use realistic data (ParticleGroup objects)
- ✓ Validate actual behavior, not just existence

### Smoke Tests
- ✓ Execute full S2E workflow with minimal particles
- ✓ Test complete pipeline: IMPACT-T → Bmad → QPAD
- ✓ Both real (actual execution) and mocked (call verification) variants
- ✓ Verify ParticleGroup creation and beam propagation
- ✓ Faster than system tests (~seconds vs minutes)
- ✓ Use realistic data (ParticleGroup objects)
- ✓ Validate actual behavior, not just existence

### System Tests
- ✓ Execute full simulators separately
- ✓ Use minimal parameters (100 particles) for speed
- ✓ Skip gracefully if simulators unavailable
- ✓ Verify integration between components

## Test Coverage

### Functions Tested

**Mathematical & Utility Functions**
- `ballisticPropagation`: Beam propagation over distance
- `sortIndices`: Index-based sorting
- `smallestInterval`: Statistical interval calculations
- `smallestIntervalImpliedSigma`: Gaussian fitting
- `smallestIntervalImpliedEmittance`: Emittance from statistics
- `calcBMAG`: Beam mismatch parameter

**Beam Manipulation**
- `centerBeam`: Shift to zero median/mean
- `collimateBeam`: Remove particles outside aperture
- `sliceBeam`: Divide into slices
- `getSingleBeamSlice`: Extract slice
- `getDriverAndWitness`: Split two-bunch beams
- `nudgeMacroparticleWeights`: Weight manipulation

**Beam Analysis**
- `getBeamSpecs`: Comprehensive statistics
- `emittance`: Emittance calculation
- `smallestIntervalImpliedEmittance`: Fitted emittance

**Configuration**
- `loadConfig`: YAML loading
- `applyOtherConfig`: Parameter application

**Tao Integration**
- `getBeamAtElement`: Extract beam at element
- `trackBeamHelper`: Tracking with error handling
- `getMatrix`: Transfer matrix calculation
- `setLatticeAndGetMatrix`: Combined setup

**Optimization**
- `launchTwissCorrection`: Twiss optimization
- `generalizedEmittanceSolver`: Emittance fitting

**File I/O**
- `writeBeam`: HDF5 output
- `makeBeamActiveBeamFile`: Active file setting

## Mocking Strategy

Tests use mocking to avoid dependencies on:
- Full Tao installation
- FACET2 lattice files
- IMPACT-T binary
- Display/GUI components

This allows tests to run in CI/CD and verify logic independently.

## Continuous Integration

Tests automatically run on GitHub:
- **Trigger**: Every push and pull request
- **Workflow**: `.github/workflows/tests.yml`
- **Runs**: All 210+ tests
- **Coverage**: Reports to PR
- **Fast option**: `pytest -m "not slow"` available

See `.github/workflows/tests.yml` for CI configuration.

## Adding Tests

When adding tests:

1. Choose appropriate file or create new one
2. Use fixtures from `conftest.py`
3. Add markers (`@pytest.mark.slow`, etc.) if needed
4. Mock external dependencies
5. For integration tests: call real functions, validate behavior
6. For unit tests: mock heavily, test logic
7. Update this README
