# FACET-II start-to-end (S2E) simulation toolkit

This repository contains utilities, Jupyter notebooks, and configuration files used to perform start-to-end (S2E) simulations of the [FACET-II](https://facet-ii.slac.stanford.edu/) particle accelerator beamline, a US Department of Energy National National User Facility which hosts hundreds of users a year.  The core workflow uses [IMPACT-T](https://github.com/impact-lbl/IMPACT-T) for beam generation and low energy transport, [Bmad, Tao, and PyTao](https://www.classe.cornell.edu/bmad/) for most of the beam transport through the kilometer-long linear accelerator, and, optionally, [QPAD](https://picksc.physics.ucla.edu/qpad.html) for particle-in-cell simulations of the beam in plasma, and [openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics) for handling beam files. It is intended to abstract away unnecessary detail so present or prospective facility users can quickly and easily run the most common types of simulations including parameter scans, constrained optimization of both Twiss and multiparticle tracking objectives, and jitter sensitivity analysis.


## Installation

### To use the package:
1. **Clone with Git LFS**
   ```bash
   git clone https://github.com/slaclab/FACET2-S2E.git
   cd FACET2-S2E
   git lfs pull
   ```
   Many beam files are tracked with Git LFS.  Without LFS the notebooks will fail to load example beams.

2. **Create the conda environment**
   ```bash
   conda env create -f bmadQPADCondaEnv.yml
   conda activate bmad-qpad
   ```
3. **Register the environment (Jupyter)**
   ```bash
   python -m ipykernel install --user --name bmad-qpad --display-name "bmad-qpad"
   ```
4. **pip install**
   ```bash
   pip install .
   ```
### To edit the package:
1. **Clone with Git LFS**
   ```bash
   git clone https://github.com/slaclab/FACET2-S2E.git
   cd FACET2-S2E
   git lfs pull
   ```
   Many beam files are tracked with Git LFS.  Without LFS the notebooks will fail to load example beams.

2. **Create the conda environment**
   ```bash
   conda env create -f bmadQPADCondaEnv_dev.yml
   conda activate bmad-qpad-dev
   ```
3. **Register the environment (Jupyter)**
   ```bash
   python -m ipykernel install --user --name bmad-qpad-dev --display-name "bmad-qpad-dev"
   ```
4. **pip install**
   ```bash
   pip install . -e
   ```

## Examples

The notebooks in the repository demonstrate typical workflows:

* **`Example - Basic introduction.ipynb`** – runs Bmad simulations with a reference lattice and beam file.  It also introduces some basic functionality like reading and setting magnets using control system units, phasing linacs, etc.
* **`Example - IMPACT-T beam generation.ipynb`** – performs a full S2E run that generates the input beam with IMPACT‑T before tracking it through the lattice to the end of the beamline
* **`Example - Final focus tuning.ipynb`** – demonstrates the final focus optics optimizer to pick magnet settings to achieve desired Twiss
* **`Example - Multiparticle tracking optimization.ipynb`** – demonstrates optimization constrained by real-world hardware limits of a multiparticle tracked beam
* **`Example - Solution postprocessing and analysis.ipynb`** – postprocessing and analysis of the beam throughout the lattice
* **`Example - Beam visualization.nb`** – Mathematica notebook for advanced beam visualization and analysis, including 3D animation generation
* **`Example - Optimization progress dashboard.nb`** – Mathematica companion notebook which visualizes optimization progress, e.g. parameter sensitivities and convergence
* **`Example - Jitter study.py`** – Parallel computation of many simulations with parameters subject to jitter, informed by real-world measurements
* **`Example - QPAD jitter simulation.py`** – performs a single S2E jitter simulation using QPAD to model plasma wakefield acceleration in a Lithium Oven plasma source (located at PENT+25 cm).

## Main Repo Features

- `initializeTao()` – set up a Bmad/PyTao instance. Optionally run IMPACT‑T to create a beam or import a reference beam
- `setLattice()` – apply lattice configuration to commonly changed knobs using a dictionary or reference file
- `trackBeam()` – track a beam between arbitrary points in the lattice, applying specialized functions like centering or energy correction at checkpoints

### Other features

- `UTILITY_setLattice` functions to translate between the language and units of the FACET-II EPICS control system and simulation
- `UTILITY_linacPhaseAndAmplitude` which conveniently phases and sets the gradients of the linacs
- Plotting tools for displaying beams and the beamline itself
- Twiss optimizers for the final focus and golden lattice matching
- Infrastructure for dealing with two-bunch operation
- Various options of calculating spot sizes and emittances
- Tools to model laser heater interactions
- Mathematica notebooks which track and analyze optimizer progress
- Mathematica notebooks which visualize beam files, including as 3D animations


## Tests

Automated unit, integration, smoke, and system tests are located in the tests/ folder and are run through Github CI on each commit. These cover the full scope of the examples. To run them, do the following:

```bash
# Run all tests
pytest tests/

# Run only fast tests (skip slow integration/smoke/system tests)
pytest tests/ -m "not slow"

# Run specific category
pytest tests/unit -v              # Unit tests
pytest tests/integration -v       # Integration tests
pytest tests/smoke -v             # Smoke tests (full S2E workflow)
pytest tests/system -v            # System tests

# Run with timing information
pytest tests/ -v --durations=10   # Show 10 slowest tests
```

More details can be found at [tests/README.md](tests/README.md).

## Repository layout

```
ARCHIVE/                 Historical studies, optimizations, and experimental notebooks
beams/                   Reference beams and scripts to generate them
bmad/                    Bmad 'golden lattice' (https://github.com/slaclab/facet2-lattice)
examples/                Example Jupyter notebooks and Python scripts
impact/                  IMPACT‑T configuration files
other_configs/           Atypical configurations including misalignment and steering solutions
qpad/                    QPAD configuration files
setLattice_configs/      Reference configurations
src/FACET2_S2E/          Main package source code with utility functions
tests/                   Automated test suite (unit, integration, system tests)
  ├── unit/              Unit tests for core functions
  ├── integration/       Integration tests calling real functions
  └── system/            End-to-end simulator tests
bmadCondaEnv.yml         Conda environment specification for Bmad and Impact-T
bmadQPADCondaEnv.yml     Conda environment specification for Bmad, Impact-T, and QPAD
bmadQPADCondaEnv_dev.yml Development environment with testing dependencies
pyproject.toml           Package configuration and dependencies
```

## API Reference

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

### Utility Modules

Additional functionality is available in specialized modules:

- **`UTILITY_setLattice.py`** – Control system unit conversions
- **`UTILITY_linacPhaseAndAmplitude.py`** – Linac phasing utilities
- **`UTILITY_modifyAndSaveInputBeam.py`** – Beam file manipulation
- **`UTILITY_QPAD.py`** – QPAD interface and visualization
- **`UTILITY_plotLattice.py`** – Lattice visualization tools


For complete examples, see the notebooks in `examples/`.

## Notes on large files

Git LFS is required because beam files can be tens of megabytes.  Without LFS you may see errors such as:

```
OSError: Unable to synchronously open file (file signature not found)
```

If you cannot use LFS, manually download the `.h5` beam files from another source and place them in the appropriate directories.

The target destination should have >2 GB of space available before cloning this repo.  

## Further documentation

Most development work happens inside the notebooks.  The notebooks in the `ARCHIVE` folder are previous investigations and may serve as additional examples.


## Support

For support, contact @majernik-slac-stanford-edu
