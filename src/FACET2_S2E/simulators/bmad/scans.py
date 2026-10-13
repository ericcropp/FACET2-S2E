"""Parameter scans over the lattice.

Moved from simulationFunctions.py.
"""

import numpy as np
from pmd_beamphysics import ParticleGroup

from ...beam.generation import make_simple_bunch_theory_from_bunch_sims
from .lattice.optics import get_rij, get_tijk
from ...plotting.style import make_a_plot
from .core import getBeamAtElement
from .runs import set_beam, run_initialized_sim


## Scans

def make_1d_scan(tao, mean=0, nscan=21, scan_span=5e-3, function_to_change_tao_in_scan=None, function_to_get_results_in_scan=None, plot=True, label="y", xlabel="x axis", ylabel="y axis", **kwargs):
    """Perform a 1D parameter scan over Tao and optionally plot the results.
    """
    scan_values = scan_span*(np.arange(nscan)-(nscan-1)/2)/(nscan-1)+mean
    output = []
    for value_ind in range(nscan):
        tao = function_to_change_tao_in_scan(tao, scan_values[value_ind], **kwargs)
        output.append(function_to_get_results_in_scan(tao, **kwargs))
    output = np.array(output)
    if plot:
        make_a_plot(scan_values, output, label=label, x_label=xlabel, y_label=ylabel, cartesian_axes=[False, False], axes_location=[0, 0])
    return scan_values, output


def make_comparison_dz_2nd_order(tao, beam_file=None, start='L0AFEND', finish='PR11375', means_shift=[0,0,0,0,0,0], theory=True, second_order=True, **kwargs):
    """Compare simulation and second-order theory for longitudinal bunch size growth.
    beam_file: beam file without ".h5". Default: <repository>/temp_beam/temp."""
    if beam_file is None:
        beam_file = tao.filePathGlobal + "/temp_beam/temp"
    # sim
    set_beam(tao, beam_file)
    run_initialized_sim(tao, start, finish)
    
    dzsim = 3e8*(getBeamAtElement(tao, finish, tToZ=False).t - np.mean(getBeamAtElement(tao, finish, tToZ=False).t))
    size_sim = np.sqrt(np.mean(dzsim**2) - np.mean(dzsim)**2)

    # theory
    if theory:
        beam = make_simple_bunch_theory_from_bunch_sims(beam_file+"_e", np.mean(ParticleGroup(beam_file+"_e.h5").pz)*1e-6, means_shift=means_shift)
        
        r5i = np.array([float(get_rij(tao, start, finish, 5, j+1)) for j in range(6)])
        t5ij = np.zeros((6,6))
        for j in range(6):
            for k in range(6):
                t5ij[j][k] = float(get_tijk(tao, start, finish, 5, j+1, k+1))
        # get_tijk returns the Taylor-map coefficient of the monomial x_j*x_k (z_out = ... + T566*delta**2,
        # no 1/2), and (j, k), (k, j) are the same monomial, so each pair is counted once
        nonlinear_impact = np.zeros(len(beam))
        for j in range(6):
            for k in range(6):
                nonlinear_impact += t5ij[j][k]*(beam[:, j]*beam[:, k]) if j>=k else 0
        nonlinear_impact = nonlinear_impact if second_order else 0*nonlinear_impact
        dz = np.sum([r5i[j]*beam[:, j] for j in range(6)], axis=0) + nonlinear_impact
        dz = dz - np.mean(dz)
        size_theory = np.sqrt(np.mean(dz**2) - np.mean(dz)**2)
        
        return size_sim, size_theory
    else:
        return size_sim
