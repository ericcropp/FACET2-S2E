from pmd_beamphysics import ParticleGroup
import numpy as np
import os
import random


def modifyAndSaveInputBeam(
    inputBeamFilePath,
    betaX = None,
    alphaX = None,
    betaY = None,
    alphaY = None,
    numMacroParticles = None,
    timeCenterTF = True,
    outputBeamFilePath = None
):
    """
    Modify and save a particle beam file with optional downsampling, time centering, and Twiss parameter matching.

    Parameters
    ----------
    inputBeamFilePath : str
        Path to the input beam file (HDF5 format) to be loaded as a ParticleGroup.
    betaX : float, optional
        Target beta function in the x-plane for Twiss matching. If None, no matching is performed.
    alphaX : float, optional
        Target alpha function in the x-plane for Twiss matching. If None, no matching is performed.
    betaY : float, optional
        Target beta function in the y-plane for Twiss matching. If None, no matching is performed.
    alphaY : float, optional
        Target alpha function in the y-plane for Twiss matching. If None, no matching is performed.
    numMacroParticles : int, optional
        If specified, randomly downsample the beam to this number of macroparticles, rescaling weights accordingly.
    timeCenterTF : bool, default True
        If True, center the time coordinate of the beam (subtract mean t from all particles).
    outputBeamFilePath : str, optional
        Path to save the modified beam file. If None, saves to './beams/activeBeamFile.h5' in the current working directory.

    Returns
    -------
    ParticleGroup
        The modified ParticleGroup object.

    Notes
    -----
    - Downsampling is performed by random selection and weight rescaling, preserving the distinction between driver/witness if present.
    - Time centering assumes all particle weights are equal or nearly equal.
    - Twiss matching is applied independently to x and y planes if the corresponding parameters are provided.
    - The function writes the modified beam to disk and also returns the ParticleGroup object for further use.
    """

    #Import
    P = ParticleGroup(inputBeamFilePath)

    #Downsample
    #if numMacroParticles:
    #    P.data.update(resample_particles(P, n=numMacroParticles))
    #PROBLEM! Built-in resampling smushes everything down to a single particle weight. No good for me since I'm using those to keep track of driver/witness
    #Instead, since the weights are ~equal, just pick a random subset then rescale their weights
    initialImportSize = np.size(P.x)
    if numMacroParticles:
        numMacroParticles = int(numMacroParticles)
        P = P[random.sample(range(initialImportSize), numMacroParticles)]
        P.weight = P.weight * (initialImportSize / numMacroParticles)
    

    #Time center
    if timeCenterTF:
        P.t=P.t-np.mean(P.t) #This is OK because present beam doesn't have different weights; np.unique(P.weight)

    #Apply linear matching
    if (betaX is not None) and (alphaX is not None):
        P.twiss_match(
              plane='x',
              beta = betaX,
              alpha = alphaX,
              inplace=True)

    if (betaY is not None) and (alphaY is not None):
        P.twiss_match(
              plane='y',
              beta = betaY,
              alpha = alphaY,
              inplace=True)

    filePath = os.getcwd()

    if not outputBeamFilePath:
        P.write(f'{filePath}/beams/activeBeamFile.h5')
    else:
        P.write(outputBeamFilePath)

    #Also return the beam object
    return P




