# API reference

The FACET2-S2E toolkit provides a high-level Python API for performing start-to-end simulations. Import the main utilities with:

```python
import FACET2_S2E as qs
```

### **Core Workflow Functions**

#### `initializeTao()`
Initialize a Bmad/PyTao simulation instance with optional beam generation.

```python
tao = qs.initializeTao(
    filePath='/path/to/FACET2-S2E',
    inputBeamFilePathSuffix='beams/mybeam.h5',  # Or None to use default
    numMacroParticles=50000,                     # Number of macroparticles
    runImpactTF=False,                           # True to run IMPACT-T beam generation
    runQPAD=False,                               # True to enable QPAD plasma simulation
    setQPADDefaultsFile='qpad/defaults.yml',     # QPAD configuration
    csrTF=True,                                  # Enable coherent synchrotron radiation
    transverseWakes=False,                       # Enable transverse wakefields
    scratchPath='/tmp',                          # Working directory for temporary files
    randomizeFileNames=True                      # Prevent file collisions in parallel runs
)
```

**Returns:** A PyTao object with custom attributes (`activeFilePath`, `patchFilePath`, `qpadSimPath`, etc.)

#### `setLattice()`
Configure the lattice using physical or control system parameters.

```python
qs.setLattice(
    tao,
    configFile='setLattice_configs/defaults.yml',  # Configuration file
    L0BFPHASE=-3.0,                                # L0B phase (degrees)
    L1PHASE=0.0,                                   # L1 phase (degrees)
    L2PHASE=-7.0,                                  # L2 phase (degrees)
    L3PHASE=-10.0,                                 # L3 phase (degrees)
    FBSOL1=0.5,                                    # Solenoid 1 field (T)
    FBSOL2=0.5,                                    # Solenoid 2 field (T)
    QFF1=-0.3,                                     # Final focus quad strength (T)
    # ... many more parameters available
)
```

**Configuration files** define default values and can be loaded with `loadConfig()`.

#### `trackBeam()`
Track the beam through the lattice with optional processing at checkpoints.

```python
qs.trackBeam(
    tao,
    filePath='/path/to/FACET2-S2E',
    trackStart='BEGBC20',                # Start element
    trackEnd='PENT',                     # End element
    centerDL10=True,                     # Center beam at DL10
    centerBC14=True,                     # Center beam at BC14
    centerBC20=True,                     # Center beam at BC20
    centerMFFF=False,                    # Center at final focus
    assertEnergyBC14=10.5e9,            # Assert energy at BC14 (eV)
    plasmaSIM=False                      # Enable QPAD plasma simulation
)
```

### Beam Analysis Functions

#### `getBeamAtElement()`
Extract beam data at a specific element.

```python
beam = qs.getBeamAtElement(tao, 'PENT')  # Returns ParticleGroup object
print(f"Beam sigma_x: {beam.sigma('x')} m")
print(f"Beam charge: {beam.charge} C")
```

#### `getBeamSpecs()`
Get comprehensive beam parameters and Twiss values at treaty points.

```python
specs = qs.getBeamSpecs(
    beam,
    twissTreatyPointString='PR10571',  # Or 'BEGBC20', 'MFFF', 'PENT'
    savedData={}                       # Optional dict to append results
)
# Returns dict with keys: sigmaX, sigmaY, sigmaZ, emitX, emitY, emitZ,
# betaX, betaY, alphaX, alphaY, charge, nPart, etc.
```

#### `getDriverAndWitness()`
Separate a two-bunch beam into driver and witness populations.

```python
driver, witness = qs.getDriverAndWitness(beam, threshold=0.02)
print(f"Driver charge: {driver.charge} C")
print(f"Witness charge: {witness.charge} C")
```

### Beam Manipulation Functions

#### `centerBeam()`
Center the beam distribution.

```python
centered_beam = qs.centerBeam(
    beam,
    centerType='median',        # 'median' or 'mean'
    assertEnergy=10.5e9         # Optional: assert mean energy (eV)
)
```

#### `collimateBeam()`
Apply collimator apertures to remove particles.

```python
collimated = qs.collimateBeam(
    beam,
    allCollimatorRules=[
        [-2e-3, 2e-3],  # x-aperture in meters
        [-1e-3, 1e-3]   # y-aperture in meters
    ]
)
```

#### `sliceBeam()`
Divide beam into longitudinal slices.

```python
slices = qs.sliceBeam(beam, nSlices=10, coordinate='z')
# Returns list of ParticleGroup objects
```

### Lattice Configuration

#### `loadConfig()`
Load a configuration file.

```python
config = qs.loadConfig(
    'setLattice_configs/my_config.yml',
    '/path/to/FACET2-S2E'
)
```

#### `getLinacMatchStrings()`
Get element match strings for linac sections.

```python
L1, L2, L3, markers = qs.getLinacMatchStrings(tao)
# Returns lists of element names for each linac section
```

### Optimization Functions

#### `launchTwissCorrection()`
Optimize quadrupole settings to achieve target Twiss parameters.

```python
result = qs.launchTwissCorrection(
    tao,
    targetTwiss={'betaX': 5.7, 'alphaX': -2.1, 'betaY': 2.6, 'alphaY': 0.0},
    knobList=['QFF1', 'QFF2', 'QFF3', 'QFF4'],
    bounds=[(-1, 1), (-1, 1), (-1, 1), (-1, 1)],
    location='PENT'
)
# Returns optimized [betaX0, alphaX0, betaY0, alphaY0]
```

#### `generalizedEmittanceSolver()`
Calculate generalized emittance from R-matrix measurements.

```python
beta, alpha, emit = qs.generalizedEmittanceSolver(
    dataList=[
        {'R11': 1.2, 'R12': 0.5, 'sigma': 100e-6},
        {'R11': 1.0, 'R12': 0.3, 'sigma': 80e-6},
        # ... more measurements
    ],
    energyGeV=10.5
)
```

### Visualization Functions

#### `plotMod()`
Create 2D histogram plots of beam phase space.

```python
fig = qs.plotMod(
    beam,
    'z', 'pz',                    # x and y coordinates
    bins=200,                     # Number of bins
    xlim=(-200e-6, 100e-6),      # x limits
    ylim=(9e9, 10.5e9)           # y limits
)
# z_from_t=True plots z = -c*delta_t, for beams recorded at a fixed s (e.g. Bmad output).
# qs.print_result(beam, couples=[['x','xp'], ['z','pz']]) draws several such plots side by side.
```

#### `plotInteractiveQPADFigure()`
Create interactive visualizations of QPAD simulation output.

```python
ui, update = qs.plotInteractiveQPADFigure(
    sim_fold=tao.qpadSimPath,
    quants=['rho', 'ez', 'raw'],
    plot_type=['imshow', 'slice, r, 0e-6', 'z,pz'],
    ylims=[[-100e-6, 100e-6], [None, None], [None, None]],
    xlims=[[None, None], [None, None], [None, None]],
    vlims=[[0, 2], [None, None], [None, None]],
    cmaps=['Blues', None, 'jet']
)
display(ui)
update()
```


### QPAD Integration
When `runQPAD=True`, plasma wakefield simulations are automatically performed during tracking:

```python
tao = qs.initializeTao(
    filePath=filepath,
    runQPAD=True,
    setQPADDefaultsFile='qpad/2025-08-20-QPAD_defaults.yml'
)
qs.trackBeam(tao, filepath, plasmaSIM=True)

# Access QPAD results
qs.saveAllQPADFigures(
    sim_fold=tao.qpadSimPath,
    save_fold=tao.qpadSimPath + '/figures'
)
```

### Parallel Jitter Studies
Use multiprocessing for parameter sensitivity studies:

```python
from multiprocessing import Pool

def worker(config):
    tao = qs.initializeTao(**config)
    qs.setLattice(tao, **config['lattice'])
    # Apply jitter, track, analyze
    return results

with Pool(8) as pool:
    results = pool.map(worker, config_list)
```

### Bmad workflow and beam tools

Functions used with `get_tao_from_experiment()` (see [bmad_simulations.md](bmad_simulations.md)). All are available as `qs.<name>`.

| Area | Functions |
|---|---|
| Beam creation and editing | `make_bunch`, `make_simple_bunch*`, `edit_bunch_parameters`, `edit_bunch_parameters_from_PG`, `modifyInputBeamSimple`, `cut_length` |
| Energy tuning | `tune_to_P0Cs`, `edit_energy_based_on_beam_inj` / `_L1` / `_L2` / `_L3` / `_all` |
| Collective effects | `applyBMADCollectiveEffectSettings` |
| Lattice information | `get_element_array`, `get_rij`, `get_tijk` |
| Sextupoles | `setAllWChicaneSextupoles`, `setAllWChicaneSextupolesXOffsets`, `setAllWChicaneSextupolesYOffsets` |
| Scans | `make_1d_scan`, `make_comparison_dz_2nd_order` |
| Plotting | `print_result`, `print_result_from_tao`, `print_result_from_file`, `plotMod`, `make_a_plot`, `enable_plt_styling` |
| Microbunching | `make_modulated_bunch`, `hist`, `get_spectrum`, `print_spec`, `analyze_spec`, `get_microbunching_gain*` |

### Package Modules

Everything in `FACET2_S2E.__all__` is available directly as `qs.<name>`. The code lives in:

| Module | Contents |
|---|---|
| `simulators.bmad.core` | `initializeTao`, `trackBeam`, `getBeamAtElement`, `writeBeam`, collective-effect settings |
| `simulators.bmad.config` | `loadConfig`, `applyOtherConfig`, auto energy-compensation switches |
| `simulators.bmad.runs` | `set_beam`, `run_initialized_sim*`, `edit_energy_based_on_beam_all` |
| `simulators.bmad.energy` | `tune_to_P0Cs`, dipole field handling, per-linac energy edits |
| `simulators.bmad.experiment` | `get_tao_from_experiment`, BMAD-to-EPICS-PV maps, DAQ-dataset lattice edits |
| `simulators.bmad.scans` | `make_1d_scan`, `make_comparison_dz_2nd_order` |
| `simulators.bmad.lattice.set_lattice` | `setLattice`, get/set helpers in control-system units (kG, GeV/c), offsets |
| `simulators.bmad.lattice.linac` | Linac phasing and gradient utilities |
| `simulators.bmad.lattice.optics` | Transfer matrices, `get_rij`/`get_tijk`, `launchTwissCorrection` |
| `simulators.bmad.lattice.final_focus` | `finalFocusSolver` |
| `beam.generation` | `make_bunch` (6D covariance, Cholesky) and the `make_simple_bunch*` wrappers |
| `beam.manipulation` | `modifyAndSaveInputBeam`, `edit_bunch_parameters*`, center/collimate/slice/cut |
| `beam.analysis` | Smallest-interval spot sizes and emittances, `calcBMAG`, `getBeamSpecs`, `generalizedEmittanceSolver` |
| `beam.microbunching` | `addLHmodulation`, `make_modulated_bunch`, spectra and microbunching gain |
| `plotting.phase_space` | `plotMod`, `slicePlotMod` |
| `plotting.bunch_summary` | `print_result*` |
| `plotting.floorplan` | `floorplanPlot` |
| `plotting.style` | `enable_plt_styling`, `make_a_plot` |
| `simulators.impact` | IMPACT-T interface |
| `simulators.qpad`, `simulators.qpad_picmi` | QPAD interface and visualization |

The old `UTILITY_*` module paths (e.g. `from FACET2_S2E.UTILITY_quickstart import trackBeam`) still work but emit a `FutureWarning`; they will be removed in a future version, so please switch to `qs.<name>` or the modules above.


For complete examples, see the notebooks in [`examples/`](../examples/).
