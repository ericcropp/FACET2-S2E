"""Add density modulation to a beam and analyse its spectrum / microbunching gain.

Moved from microbunchingFunctions.py; addLHmodulation from UTILITY_quickstart.py.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn as besselj
from pmd_beamphysics import ParticleGroup
import math
from scipy.stats import moment
from scipy.ndimage import gaussian_filter1d

from ..plotting.style import make_a_plot


def addLHmodulation(
    inputBeam, 
    #Elaser, 
    showplots=False,
    laserHeater_laserEnergy = 0.5e-3,
    laserHeater_sigma_t =  (2 / 2.355) * 1e-12,
    laserHeater_offset = -0.5
):
    """ From C. Emma, 2024-08-23 """
    # Hardcode FACET-II laser and undulator parameters
    # Laser parameters
    Elaser = laserHeater_laserEnergy
    lambda_laser = 760e-9
    sigmar_laser = 200e-6
    sigmat_laser = laserHeater_sigma_t # (2 / 2.355) * 1e-12
    Plaser = Elaser / np.sqrt(2 * np.pi * sigmat_laser**2)
    offset = laserHeater_offset #-0.5  # laser to e-beam offset if you want you can add it
    # Undulator parameters
    K = 1.1699
    lambdaw = 0.054
    Nwig = 9
    # Electron beam
    outputBeam = inputBeam.copy()
    x = inputBeam.x - np.mean(inputBeam.x)
    y = inputBeam.y - np.mean(inputBeam.y)
    gamma = inputBeam.gamma
    gamma0 = np.mean(inputBeam.gamma)
    t = inputBeam.t-np.mean(inputBeam.t);
    # Calculated parameters
    lambda_r = lambdaw / 2 / gamma0**2 * (1 + K**2 / 2)  # Assumes planar undulator
    omega = 2 * np.pi * 299792458 / lambda_r  # Resonant frequency
    JJ = besselj(0, K**2 / (4 + 2 * K**2)) - besselj(1, K**2 / (4 + 2 * K**2))
    totalLaserEnergy = np.sqrt(2 * np.pi * sigmat_laser**2) * Plaser
    # Laser is assumed Gaussian with peak power Plaser
    # This formula from eq. 8 Huang PRSTAB 074401 2004
    mod_amplitude = np.sqrt(Plaser / 8.7e9) * K * lambdaw * Nwig / gamma0 / sigmar_laser * JJ
    #print(mod_amplitude / np.sqrt(Plaser))
    # offset = 1.0  # temporal offset between laser and e-beam in units of laser wavelengths
    # Calculate induced modulation deltagamma
    deltagamma = mod_amplitude * np.exp(-0.25 * (x**2 + y**2) / sigmar_laser**2) * \
                 np.sin(omega * t + offset * 2 * np.pi) * \
                 np.exp(-0.5 * ((t - offset * sigmat_laser) / sigmat_laser)**2)
    outputBeam.gamma = inputBeam.gamma + deltagamma
    return outputBeam, deltagamma, t


def make_modulated_bunch(beam, wavelength=30e-6, mod_amplitude=0.1, save_file=""):
    '''
    Add microbunching to the bunch
    '''
    indices_to_leave = []
    for (i,p) in enumerate(beam):
        if(np.random.rand()<1-0.5*mod_amplitude*(math.sin(2*math.pi*((p.t)[0])*3*10**8/wavelength)+1)):
            indices_to_leave.append(i)
            
    indices = np.array(indices_to_leave)
    P_local=beam[indices]
    P_local.charge = beam.charge
    if save_file!="":
        P_local.write(save_file+".h5")
    return P_local


## Display the bunch density distribution \rho(z)

def hist(data, label='Longitudinal Coordinate z (um)', num_bins=200, xlim=None):
    # Compute histogram (density=True normalizes area under curve to 1)
    hist_kwargs = dict(bins=num_bins, density=True)
    if xlim is not None:
        if len(xlim) != 2:
            raise ValueError('xlim must be a sequence of two values: (xmin, xmax)')
        hist_kwargs['range'] = xlim
    counts, bin_edges = np.histogram(data, **hist_kwargs)
    
    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot as line (envelope)
    #plt.figure(figsize=(8, 4))
    #plt.plot(bin_centers, counts, color='black', linewidth=2)
    
    # Optional: Smooth the curve
    smoothed_counts = gaussian_filter1d(counts, sigma=1.0)

    fig = plt.figure(figsize=(8, 3.25))
    plt.plot(bin_centers, smoothed_counts, color='black')

    plt.fill_between(bin_centers, smoothed_counts, color='grey', alpha=0.5)
    
    if xlim is not None:
        plt.xlim(xlim)
    
    # Styling
    plt.xlabel(label, fontsize=12)
    #plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()


## Spectrum functions

DEFAULT_NBINS = 5000


BINS_PER_PERIOD = 16


MAX_NBINS = int(2e6)


def get_nbins_for_wavelength(l_dist, min_wavelength, bins_per_period=BINS_PER_PERIOD, max_nbins=MAX_NBINS):
    """Number of histogram bins needed to resolve min_wavelength in the distribution l_dist.

    The histogram spans the full length of the bunch, so the bin width is span/nbins.
    Asking for bins_per_period bins across the shortest wavelength of interest keeps that
    wavelength well below the Nyquist limit (2 bins per period), where the spectrum is empty.

    Parameters:
        l_dist: Longitudinal coordinates of the macroparticles (m).
        min_wavelength: Shortest wavelength that has to be resolved (m).
        bins_per_period: Bins per min_wavelength period. The gain converges to within a few
        percent from 16 bins per period upwards; 4 is already ~15% off.
        max_nbins: Safety cap, in case a very short wavelength is asked of a long bunch.

    Returns:
        int: Number of bins.
    """
    span = np.ptp(l_dist)
    nbins = int(np.ceil(span/(min_wavelength/bins_per_period)))
    if nbins > max_nbins:
        print(f"Warning: resolving {min_wavelength:.3g} m over a {span:.3g} m long bunch needs "
              f"{nbins} bins; capping at {max_nbins}. The spectrum near {min_wavelength:.3g} m "
              f"will be undersampled.")
        nbins = max_nbins
    return int(np.maximum(nbins, 2))


def get_spectrum(beam, nbins=None, min_wavelength=None, bins_per_period=BINS_PER_PERIOD):
    """Longitudinal density spectrum of a bunch.

    Parameters:
        beam: ParticleGroup.
        nbins: Number of histogram bins. Overrides min_wavelength if given.
        min_wavelength: Shortest wavelength of interest (m). The binning is chosen to resolve it.
        If both nbins and min_wavelength are None, DEFAULT_NBINS bins are used.
        bins_per_period: Bins per min_wavelength period.

    Returns:
        list: [bin width (m), [normalized frequency axis, complex FFT of the histogram], rms bunch length (m)]
    """
    c = 2.99792458e8
    l_dist = (beam.t-np.mean(beam.t))*c
    sigz = moment(l_dist, moment=2) ** 0.5
    if nbins is None:
        nbins = DEFAULT_NBINS if min_wavelength is None else get_nbins_for_wavelength(l_dist, min_wavelength, bins_per_period=bins_per_period)
    nbins = int(nbins)
    [hist, bin_edges] = np.histogram(l_dist, bins=nbins)
    fourier = np.fft.fft(hist)
    x = np.arange(nbins)/nbins
    # Keep x (real) and fourier (complex) in a list, not a stacked np.array:
    # stacking would promote the frequency axis to complex128 and leak ComplexWarnings
    # into every downstream plot and float() conversion.
    return [bin_edges[1]-bin_edges[0],[x, fourier],sigz]


def print_spec(beam, maxlam, minlam, nbins=None, bins_per_period=BINS_PER_PERIOD):
    spec = get_spectrum(beam, nbins=nbins, min_wavelength=minlam, bins_per_period=bins_per_period)
    specx, specy, idx_wl, idx_wh = get_spec_band(spec, maxlam, minlam)
    make_a_plot(specx[idx_wl:idx_wh],specy[idx_wl:idx_wh], label="Spectrum of z", x_label="Frequency (normalized)", y_label="Power", cartesian_axes=[False, True], axes_location=[0, 0])


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def get_spec_band(spec, maxlam, minlam):
    """Select the [minlam, maxlam] wavelength band of a spectrum from get_spectrum().

    Returns:
        tuple: (frequency axis, power, first index of the band, last index of the band)
    """
    bsize = spec[2]
    maxlam = np.minimum(maxlam, bsize/2.5)
    minFreq = spec[0]/maxlam
    maxFreq = spec[0]/minlam
    maxFreq = np.minimum(maxFreq, 0.5)
    specx = spec[1][0]
    specy = np.power(np.abs(spec[1][1]),2)
    idx_wl = find_nearest(specx,minFreq)
    idx_wh = find_nearest(specx,maxFreq)
    if idx_wh <= idx_wl:
        raise ValueError(
            f"Empty spectral band: wavelengths {minlam:.3g}-{maxlam:.3g} m are not resolved by a "
            f"histogram with {len(specx)} bins of {spec[0]:.3g} m (Nyquist wavelength "
            f"{2*spec[0]:.3g} m). Pass min_wavelength={minlam:.3g} to get_spectrum(), or a larger nbins.")
    return specx, specy, idx_wl, idx_wh


def analyze_spec(spec, maxlam, minlam):
    specx, specy, idx_wl, idx_wh = get_spec_band(spec, maxlam, minlam)
    minFreq = specx[idx_wl]
    # idx_low/idx_high are computed and then overridden with the full band below
    idx_max = np.abs(specy[idx_wl:idx_wh]).argmax()+idx_wl

    idx_low = find_nearest(specx,np.maximum(minFreq, 0.8*specx[idx_max]))
    idx_high = find_nearest(specx, 1.2*specx[idx_max])

    idx_low = idx_wl
    idx_high = idx_wh
    
    l = idx_high-idx_low

    meanx = 0
    toty = 0
    for i in range(l):
        meanx += specx[idx_low+i]*specy[idx_low+i]
        toty += specy[idx_low+i]
    meanx = meanx/toty

    sigmax = 0
    for i in range(l):
        sigmax += np.power(specx[idx_low+i]-meanx,2)*specy[idx_low+i]
    sigmax = np.sqrt(sigmax/toty)

    idx_3sl = find_nearest(specx,meanx-3*sigmax)
    idx_3sh = find_nearest(specx,meanx+3*sigmax)
    l1 = idx_3sh-idx_3sl
    quadpower = 0
    for i in range(l1):
        quadpower += (specx[idx_3sl+i+1]-specx[idx_3sl+i])*specy[idx_3sl+i]
    return [meanx,sigmax,quadpower]


def get_microbunching_gain_from_beams(initial_beam, final_beam, lmax1, lmin1, lmax2, lmin2, file="", nbins=None, bins_per_period=BINS_PER_PERIOD):
    ipars = analyze_spec(get_spectrum(initial_beam, nbins=nbins, min_wavelength=lmin1, bins_per_period=bins_per_period),lmax1,lmin1)
    fpars = analyze_spec(get_spectrum(final_beam, nbins=nbins, min_wavelength=lmin2, bins_per_period=bins_per_period),lmax2,lmin2)
    if file!="":
        np.savetxt(file,np.real(np.array([fpars,ipars])))
    return fpars[2]/ipars[2]


def get_microbunching_gain(initial_beam_path, final_beam_path, lmax1, lmin1, lmax2, lmin2, file="", nbins=None, bins_per_period=BINS_PER_PERIOD):
    return get_microbunching_gain_from_beams(ParticleGroup(initial_beam_path), ParticleGroup(final_beam_path), lmax1, lmin1, lmax2, lmin2, file=file, nbins=nbins, bins_per_period=bins_per_period)
