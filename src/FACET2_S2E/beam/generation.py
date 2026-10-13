"""Create bunches (Gaussian and theory-matched).

Moved from beamFunctions.py.
"""

from scipy.stats import gennorm
from scipy.special import gamma
import numpy as np
from pmd_beamphysics import ParticleGroup

from .manipulation import cut_length, sqrtm_psd


## Bunch support functions

### Create a bunch

def _covariance_sqrt(sigma):
    """Return A with A @ A.T == sigma, for drawing correlated samples as A @ (unit normals).

    A is the Cholesky factor of the correlation matrix, scaled back by the rms values, so the
    mixed units (m, rad, eV/c) do not spoil the conditioning. Coordinates with zero spread are
    left out of the decomposition and stay at their mean (by Cauchy-Schwarz a zero-variance
    coordinate has zero covariance with everything). If the rest is still singular, e.g. two
    exactly proportional coordinates, the symmetric PSD square root is used instead.
    """
    sigma = np.asarray(sigma, dtype=float)
    std = np.sqrt(np.clip(np.diag(sigma), 0.0, None))
    keep = std > 0
    A = np.zeros_like(sigma)
    if keep.any():
        s = std[keep]
        corr = sigma[np.ix_(keep, keep)] / np.outer(s, s)
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            L, _ = sqrtm_psd(corr)
        A[np.ix_(keep, keep)] = s[:, None] * L
    return A


def make_bunch(N, sigma, means, charge=1e-9, t_profile='gaussian', save_path=None):
    """Create a bunch with a given 6x6 covariance matrix (core generator for make_simple_bunch*).

    Coordinates are (x, xp, y, yp, z, pz) in (m, rad, m, rad, m, eV/c), with xp = px/pz,
    yp = py/pz and z = -c*(t - <t>), so the head is at positive z and sigma[4, 5] > 0 means the
    head has more momentum. Particles are generated at a fixed z (z = 0) with a spread in t, as
    the Bmad tracking functions expect.

    Parameters:
        N: Number of macro particles.
        sigma: 6x6 covariance matrix of (x, xp, y, yp, z, pz). Correlations are generated with a
            Cholesky decomposition; coordinates with zero rms are set to their mean.
        means: Means of (x, xp, y, yp, c*t, pz). As in edit_bunch_parameters, the 5th entry is c*<t>.
        charge: Total bunch charge in C.
        t_profile: 'gaussian', or 'flat' for a generalized normal (order 4) longitudinal profile.
            The profile applies to the part of z that is uncorrelated with x, xp, y, yp.
        save_path: Output file path without extension. If None, nothing is written.

    Returns:
        ParticleGroup: Generated bunch.
    """
    N = int(N)
    u = np.random.standard_normal((N, 6))
    if t_profile == 'flat':
        u[:, 4] = gennorm.rvs(4, size=N) / np.sqrt(gamma(3/4) / gamma(1/4))  # unit variance
    elif t_profile != 'gaussian':
        raise ValueError(f"t_profile must be 'gaussian' or 'flat', not {t_profile!r}")
    X = u @ _covariance_sqrt(sigma).T

    pz = X[:, 5] + means[5]
    data = {'x': X[:, 0] + means[0], 'px': (X[:, 1] + means[1]) * pz,
            'y': X[:, 2] + means[2], 'py': (X[:, 3] + means[3]) * pz,
            'z': np.zeros(N), 'pz': pz, 't': (means[4] - X[:, 4]) / 3e8,
            'status': np.ones(N).astype(int), 'weight': np.ones(N)*charge/N, 'species': 'electron', 'id': np.arange(N).astype(int)}
    P1 = ParticleGroup(data = data)

    if save_path is not None:
        P1.write(save_path + '.h5')
    return P1


def make_simple_bunch(file, n = 0, save_path = '', correlations = False, t_profile = 'gaussian'):
    """Create a Gaussian bunch with statistics matched to an input beam.

    With correlations=False (default) every coordinate is drawn independently, from a normal
    distribution matched to the input beam rms values:
    the transverse emittance is increased (alpha = 0, rms are the same);
    the longitudinal emittance is kept approximately the same (alpha = 0, rms by pz is taken from a 0.1um slice).
    With correlations=True the full 6x6 covariance of the input beam is reproduced (Twiss,
    projected emittances, linear chirp, dispersion), see make_bunch.
    The bunch is centered transversely and in t; the mean pz of the input beam is kept.

    Parameters:
        file: Base path of input beam file without the .h5 extension.
        n: Number of macro particles to generate. If 0, the input beam size is used.
        save_path: Output file path without extension. If "", file+'_simple.h5' is used.
        correlations: If True, keep the linear correlations of the input beam.
        t_profile: 'gaussian', or 'flat' for a flatter longitudinal profile (see make_bunch).

    Returns:
        ParticleGroup: Generated bunch.
    """
    beam = ParticleGroup(file + '.h5')
    
    N = np.size(beam.x)
    N = N if n==0 else n

    if correlations:
        sigma = np.cov(np.vstack((beam.x, beam.xp, beam.y, beam.yp, -3e8*beam.t, beam.pz)), bias=True)
    else:
        Pslice = cut_length(beam, length = 1e-7)
        meanpz = np.mean(beam.pz)
        sigma = np.diag([np.std(beam.x), np.std(beam.px)/meanpz, np.std(beam.y), np.std(beam.py)/meanpz,
                         3e8*np.std(beam.t), np.std(Pslice.pz)])**2

    return make_bunch(N, sigma, [0, 0, 0, 0, 0, np.mean(beam.pz)], charge=beam.charge, t_profile=t_profile,
                      save_path=file + '_simple' if save_path=='' else save_path)


def make_simple_bunch_flatter(file, n = 0, save_path = '', correlations = False):
    """Create a flatter bunch using a generalized normal time distribution.

    Same as make_simple_bunch(..., t_profile='flat'): the longitudinal time coordinate is
    sampled from a generalized normal distribution to produce a flatter longitudinal profile.

    Parameters:
        file: Base path of input beam file without the .h5 extension.
        n: Number of macro particles to generate. If 0, the input beam size is used.
        save_path: Output file path without extension. If empty, file+'_simple.h5' is used.
        correlations: If True, keep the linear correlations of the input beam.
    """
    return make_simple_bunch(file, n=n, save_path=save_path, correlations=correlations, t_profile='flat')


def make_simple_bunch_standalone(N = 0, meanPzMeV = 125 , moments=[0.3e-3, 0.2e-3, 0.4e-3, 0.2e-3, 0.58e-3, 0], charge = 1e-9, save_path = '', means=[0,0,0,0,0,0]):
    """Create a Gaussian bunch from explicit statistical moments.

    Parameters:
        N: Number of macro particles.
        meanPzMeV: Mean longitudinal momentum in MeV/c.
        moments: RMS values in x, xp, y, yp, z, pz (m, rad, m, rad, m, eV/c). Zeros are allowed.
        charge: bunch charge.
        save_path: Output file path without extension. If empty, nothing is written.
        means: Mean values for x, xp, y, yp, c*t (the 6th entry is ignored; meanPzMeV sets <pz>).

    Returns:
        ParticleGroup: Generated bunch.
    """
    sigma = np.diag(np.asarray(moments, dtype=float)**2)
    return make_bunch(N, sigma, [*means[:5], meanPzMeV*1e6], charge=charge,
                      save_path=save_path if save_path else None)


def make_simple_bunch_theory_from_bunch_sims(bunch_file, mean_lattice_P0C_MeV, means_shift=[0,0,0,0,0,0]):
    """Convert a BMAD bunch into theory coordinates for map-based calculations.

    Parameters:
        bunch_file: Base path of the bunch file without extension.
        mean_lattice_P0C_MeV: Reference lattice momentum in MeV/c (the bunch will have this <pz>).
        means_shift: Shifts to apply to x, xp, y, yp, z, delta.

    Returns:
        np.ndarray: Nx6 array in the order [x, xp, y, yp, z, delta].
    """
    sim_bunch = ParticleGroup(bunch_file+".h5")
    return np.stack((sim_bunch.x+means_shift[0], sim_bunch.xp+means_shift[1], sim_bunch.y+means_shift[2], sim_bunch.yp+means_shift[3], -3e8*sim_bunch.t, (sim_bunch.pz*1e-6-mean_lattice_P0C_MeV)/mean_lattice_P0C_MeV), axis=1)
