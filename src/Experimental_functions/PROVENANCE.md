# Provenance of `Experimental_functions`

This package is **not original work of the FACET2-S2E project**. It is a partial,
locally-modified copy of:

    Repository: https://github.com/rariniello/standaloneFACETScripts
    Author:     Robert Ariniello (rariniello)
    Commit:     2f5f3c4a82efd3cf8e6c508016aed86ec207661b  ("Fix issue with loading
                images from hdf5 DAQs", 2025-03-26), the tip of `main` at time of copy.

## License

Upstream is licensed **BSD-3-Clause** as of commit
`6b0e3166d6b7991446a36d61eb89a8d8e519cd44` ("Add BSD 3-Clause License file",
2026-09-07). That license text is reproduced verbatim in `LICENSE` beside this
file, as clause 1 requires, and must accompany any redistribution of these
sources.

The copy in this directory was taken at `2f5f3c4`, which predates the license
commit. Robert Ariniello additionally granted permission by email on 2026-09-07:

> I added a BSD-3-Clause license to the repo.
> I am happy to authorize the redistribution of those files as part of the
> FACET2-S2E project.

so the earlier snapshot is covered explicitly as well as by the repository
license. Retain that email with the project records.

Two ongoing obligations:

* **Clause 1** -- keep `LICENSE` shipped with these sources. `pyproject.toml`
  declares it under `[tool.setuptools.package-data]` so it is present in built
  wheels and sdists, not only in the git tree.
* **Clause 3** -- do not use the author's name to endorse or promote products
  derived from this software without prior written permission. Plain
  attribution, as in the README, is fine.

BSD-3-Clause also matches FACET2-S2E's own license, so there is no
compatibility question, and it is OSI-approved for the JOSS submission.

## Files

Copied from upstream:

| File          | Status                                             |
|---------------|----------------------------------------------------|
| `analysis.py` | byte-identical to upstream 2f5f3c4                 |
| `mplstyle.py` | byte-identical to upstream 2f5f3c4                 |
| `dataset.py`  | modified, see below                                |
| `image.py`    | modified, see below                                |

Not copied: `plot.py`, `spec.py`, `example.ipynb`, `README.md`.
Added by this project: `__init__.py` (original work of this project).

`DATASET` needs `image.py`, which needs `analysis.py`. `mplstyle.py` is not used by the
other modules and sets matplotlib `rcParams` globally when imported, so `__init__.py`
does not import it; it is available as an opt-in `import Experimental_functions.mplstyle`.

## Local modifications

`dataset.py`
1. `import image` -> `from . import image` (needed to import as a subpackage).
2. `DATASET.__init__`: added a `pathfull` argument, to point at a dataset
   directory directly instead of composing it from `dataPath`/`experiment`.
3. `_loadDataStruct`: record `self.length`; fall back to a synthesised
   `common_index` when the DAQ struct stores an empty one.
4. `_loadCameraCalibration`: tolerate metadata with no `RESOLUTION` key
   (upstream indexed it unconditionally); added a debug `print(cam)`.
5. `getImage`: same empty-`common_index` fallback; index computation moved
   into the non-HDF5 branch.
6. `getBackground`: tolerate metadata with no `X_ORIENT`/`Y_ORIENT`; return the
   un-oriented background instead of `None` when it is not 3-dimensional.
7. Added the `getScalarByStep()` method (original work of this project).

`image.py`
1. `import analysis as an` -> `from . import analysis as an` (as above).
2. `orientImage`: accept `XOrient`/`YOrient` of `None` and pass the data through
   unrotated, rather than raising.
3. `_orientImage`: guard on `X_ORIENT`/`Y_ORIENT` being present in the metadata.

Modifications 1-6 of `dataset.py` and 1-3 of `image.py` are workarounds for FACET-II
DAQ datasets whose scalar/metadata structs are incomplete; they are candidates to
be sent upstream as a pull request.

## Use within FACET2-S2E

Only the `DATASET` class is used, and only at a single site:
`src/FACET2_S2E/simulators/bmad/experiment.py`, in `get_tao_from_experiment()`. The
resulting object is consumed by the lattice-edit functions in the same module,
which read `dataset._data["scalars"][...]`.
