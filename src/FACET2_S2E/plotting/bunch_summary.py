"""Display a bunch: phase-space panels plus printed moments.

Moved from plottingFunctions.py. Its horizontal-layout plotMod was merged into
plotting.phase_space.plotMod (fig/outer/i and z_from_t arguments).
"""

from scipy.stats import moment
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from pmd_beamphysics import ParticleGroup

from ..beam.manipulation import cut_length
from ..simulators.bmad.core import getBeamAtElement
from .phase_space import plotMod


### Display a bunch

# Prints specified 2d spaces, plots are arranged horizontally. Uncomment a section for vertical plotting.
def print_result(particle_group, length = 0, couples = [['t','pz']], moments = True, sliceEnergyMoment=False, energyInDelta=False, drift_to_z_in_cutting = True, z_from_t=False):
    """Display 2D phase-space plots and print basic bunch moments for a ParticleGroup."""
    P = particle_group.copy()
    #P.drift_to_z()
    if length != 0:
        Pt = cut_length(P, length, drift_to_z = drift_to_z_in_cutting)
    else:
        Pt = P

    l = len(couples)
    fig = plt.figure(figsize=(7*l,6))
    
    # outer layout: 1 row, l columns
    outer = GridSpec(1, l, wspace=0.3)
    
    for (i,couple) in enumerate(couples):
        plotMod(Pt, couple[0], couple[1], bins=300, fig=fig, outer=outer, i=i, z_from_t=z_from_t, background='white')

    plt.show()

    # if column-arranging is needed
    # for (i,couple) in enumerate(couples):
    #     #display(plotMod(Pt, couple[0], couple[1],  bins=300))
    #     plt.subplot(1,len(couples[:,1]),i+1)
    #     plotMod(Pt, couple[0], couple[1],  bins=300)
    #     plt.show()
        
    if moments:
        deltas = (P.gamma-np.mean(P.gamma))/np.mean(P.gamma)
        energies = P.pz
        energies = P.energy
        if sliceEnergyMoment:
            Pslice = cut_length(P, length = 1e-7)
            deltas = (Pslice.gamma-np.mean(Pslice.gamma))/np.mean(Pslice.gamma)
            energies = Pslice.pz
        if energyInDelta:
            sigmapz = moment(deltas, moment=2) ** 0.5
        else:
            sigmapz = moment(energies, moment=2) ** 0.5
        print([float(moment(Pt.x, moment=2) ** 0.5), float(moment(Pt.xp, moment=2) ** 0.5), float(moment(Pt.y, moment=2) ** 0.5), float(moment(Pt.yp, moment=2) ** 0.5), float(moment(Pt.z, moment=2) ** 0.5), float(sigmapz), float(moment(Pt.t, moment=2) ** 0.5)])


def print_result_from_tao(tao_local, location, length = 0, couples = [['t','pz']], moments = True, sliceEnergyMoment=False, energyInDelta=False, z_from_t=False):
    """Display results (2d distribution plots) for a beam extracted from Tao at a given location."""
    print_result(getBeamAtElement(tao_local, location, tToZ=False), length = length, couples = couples, moments = moments, sliceEnergyMoment=sliceEnergyMoment, energyInDelta=energyInDelta, z_from_t=z_from_t)


def print_result_from_file(file, length = 0, couples = [['t','pz']], moments = True, sliceEnergyMoment=False, energyInDelta=False, z_from_t=False):
    """Display results (2d distribution plots) for a beam loaded from an HDF5 file."""
    print_result(ParticleGroup(file), length = length, couples = couples, moments = moments, sliceEnergyMoment=sliceEnergyMoment, energyInDelta=energyInDelta, z_from_t=z_from_t)
