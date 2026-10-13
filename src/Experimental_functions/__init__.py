"""
FACET2: Functions to analyze DAQ data.
"""

from .analysis import (
    fwhm,
    fw_e2,
    fit_gaussian,
    fit_elliptical_gaussian,
    fit_superGaussian,
    fit_2D_superGaussian,
    find_roots,
    moving_average,
)
from .dataset import DATASET
from .image import IMAGE, Elog, DAQ, HDF5_DAQ, set_calibration, orientImage, specialFlips, ElogImage, parseDate
# mplstyle is not imported here: it sets matplotlib rcParams globally on import.
# Opt in with `import Experimental_functions.mplstyle`.

__all__ = [
    'fwhm',
    'fw_e2',
    'fit_gaussian',
    'fit_elliptical_gaussian',
    'fit_superGaussian',
    'fit_2D_superGaussian',
    'find_roots',
    'moving_average',
    'DATASET',
    'IMAGE',
    'Elog',
    'DAQ',
    'HDF5_DAQ',
    'set_calibration',
    'orientImage',
    'specialFlips',
    'ElogImage',
    'parseDate',
]

__version__ = "0.1.0"