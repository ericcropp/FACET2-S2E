"""Modify an existing beam: rematch, resample, center, collimate, slice, cut, drift.

Moved from beamFunctions.py, UTILITY_modifyAndSaveInputBeam.py and
UTILITY_quickstart.py.
"""

import random
import numpy as np
from pmd_beamphysics import ParticleGroup
import os


## Modify the bunch as a whole (sizes, means, chirps, correlations)

def modifyInputBeamSimple(inputBeamFilePath, numMacroParticles = None, timeCenterTF = True):
    """Prepare an input beam for Tao by optionally downsampling and centering. Like modifyAndSaveInputBeam(), but without Twiss matching.

    The beam is drift_to_z(), set z=0, and optionally time-centered.
    If numMacroParticles is provided (and smaller than the beam), the beam is randomly subsampled and weights are adjusted.

    Parameters:
        inputBeamFilePath: Path to the input beam file, including extension.
        numMacroParticles: Target number of macroparticles.
        timeCenterTF: If True, subtract the mean time to center the bunch and avoid cavity phase mismatch.

    Returns:
        ParticleGroup: Modified beam ready for use with Tao.
    """
    P = ParticleGroup(inputBeamFilePath)

    if numMacroParticles:
        if 0 < numMacroParticles < np.size(P.x):  # asking for more than the file has keeps all particles
            initialImportSize = np.size(P.x)
            numMacroParticles = int(numMacroParticles)
            P = P[random.sample(range(initialImportSize), numMacroParticles)]
            P.weight = P.weight * (initialImportSize / numMacroParticles)

    P.drift_to_z()
    P.z = np.zeros(np.size(P.x))
    # Time center
    if timeCenterTF:
        P.t=P.t-np.mean(P.t) # This is OK because present beam doesn't have different weights; np.unique(P.weight)
        
    return P


def sqrtm_psd(M, tol=1e-14):
    """Compute the positive-semidefinite square root of a symmetric matrix.
    
    Returns:
        tuple: (matrix_sqrt, eigenvalues)
    """
    w, V = np.linalg.eigh(M)
    w_clipped = np.clip(w, 0.0, None)  # allow zero
    return V @ np.diag(np.sqrt(w_clipped)) @ V.T, w


def invsqrtm_psd(M, tol=1e-14):
    """Compute the inverse square root of a symmetric matrix, with small eigenvalues treated as zero.
    
    Returns:
        tuple: (matrix_sqrt, eigenvalues)
    """
    w, V = np.linalg.eigh(M)
    w_inv = np.zeros_like(w)
    mask = w > tol
    w_inv[mask] = 1.0 / np.sqrt(w[mask])
    # zero eigenvalues remain zero → projection
    return V @ np.diag(w_inv) @ V.T, w


def _correlation(a, b):
    """Pearson correlation coefficient of a and b; 0 if either has zero spread."""
    sa, sb = np.std(a), np.std(b)
    if sa == 0 or sb == 0:
        return 0
    return np.mean((a - np.mean(a))*(b - np.mean(b))) / (sa*sb)


def edit_bunch_parameters_from_PG(P_arg, pzMeV=None, moments=None, correlations=None, means=None,
                                  charge=-1, betaX=None, alphaX=None, emittanceX=None, betaY=None, alphaY=None, emittanceY=None, path_to_write='temp_beam/temp_e'):
    '''
    moments are for x, xp, y, yp, z, pz (default: all None, i.e. unchanged)
    correlations are the correlation coefficients r(x,xp), r(y,yp), r(z,pz) with z = -c*t; None keeps the beam's value
    means are for x, xp, y, yp, c*t (default: 0, 0, 0, 0, 0); None keeps the beam's value
    xp and yp are in radians
    moments are RMS sizes

    if betaX and alphaX are supplied, the X phase space will have the emittance corresponding to moments[0] (size x = sqrt{epsilon beta}), and moments[1] will be overwritten. Same for Y.
    '''
    P = P_arg.copy()
    # Copies, so neither the caller's lists nor the defaults are modified below
    moments = [None]*6 if moments is None else list(moments)
    correlations = [None]*3 if correlations is None else list(correlations)
    means = [0, 0, 0, 0, 0] if means is None else list(means)

    if correlations[0] is None:
        correlations[0] = _correlation(P.x, P.xp)
    if correlations[1] is None:
        correlations[1] = _correlation(P.y, P.yp)
    if correlations[2] is None:
        # same (z = -c*t, pz) basis as sigmaMatrixZ below
        correlations[2] = _correlation(-P.t*3e8, P.pz)
    
    sigmaMatrixX=np.array([[-1,-1],[-1,-1]], dtype=float)
    sigmaMatrixY=np.array([[-1,-1],[-1,-1]], dtype=float)
    sigmaMatrixZ=np.array([[-1,-1],[-1,-1]], dtype=float)
    sigmaMatrixX[0][0] = -1 if moments[0]==None else moments[0]**2
    sigmaMatrixX[1][1] = -1 if moments[1]==None else moments[1]**2
    sigmaMatrixY[0][0] = -1 if moments[2]==None else moments[2]**2
    sigmaMatrixY[1][1] = -1 if moments[3]==None else moments[3]**2
    sigmaMatrixZ[0][0] = -1 if moments[4]==None else moments[4]**2
    sigmaMatrixZ[1][1] = -1 if moments[5]==None else moments[5]**2
    
    # charge
    if charge!=-1:
        P.charge = charge
    N = len(P.x)
    
    # longitudinal momentum (pz), t and z
    if pzMeV is not None and pzMeV>0:
        P.pz = P.pz*1e6*pzMeV/np.mean(P.pz)

    meanpz = np.mean(P.pz)
    P.pz = P.pz - meanpz
    means[4] = means[4] if means[4]!=None else np.mean(P.t)*3e8
    P.t = P.t - np.mean(P.t)

    if not ((sigmaMatrixZ[1,1]==-1) and (sigmaMatrixZ[0,0]==-1)):
        if sigmaMatrixZ[0][0]==-1:
            sigmaMatrixZ[0][0] = np.std(P.t*3e8)**2
        if sigmaMatrixZ[1][1]==-1:
            sigmaMatrixZ[1][1] = np.std(P.pz)**2
        sigmaMatrixZ[0][1] = correlations[2]*np.sqrt(sigmaMatrixZ[0][0]*sigmaMatrixZ[1][1])
        sigmaMatrixZ[1][0] = sigmaMatrixZ[0][1]
        if np.std(P.t)==0 or np.std(P.pz)==0:
            P.t = np.random.normal(0, 1, N)/3e8
            P.pz = np.random.normal(0, 1, N)
        Z = np.vstack((-P.t*3e8, P.pz))
        Sigma_current = np.cov(Z, bias=True)
        S_target_sqrt, _ = sqrtm_psd(sigmaMatrixZ)
        S_current_invsqrt, _ = invsqrtm_psd(Sigma_current)
        T = S_target_sqrt @ S_current_invsqrt
        Z_new = T @ Z
        P.t = -Z_new[0]/3e8
        P.pz = Z_new[1]

    P.pz = P.pz + meanpz
    P.t = P.t + means[4]/3e8
            
    # means before
    means[0] = means[0] if means[0]!=None else np.mean(P.x)
    means[1] = means[1] if means[1]!=None else np.mean(P.xp)
    means[2] = means[2] if means[2]!=None else np.mean(P.y)
    means[3] = means[3] if means[3]!=None else np.mean(P.yp)
    P.x = P.x - np.mean(P.x)
    P.px = P.px - np.mean(P.px)
    P.y = P.y - np.mean(P.y)
    P.py = P.py - np.mean(P.py)

    #Apply linear matching X
    if (betaX is not None) and (alphaX is not None):
        current_sqrt_emmitance = np.sqrt(np.sqrt( np.mean(P.x**2)*np.mean(P.xp**2) - np.mean(P.x*P.xp)**2 ))
        if current_sqrt_emmitance==0:
            print("edit_bunch_parameters_from_PG: Zero emittance in X. Filling with Gaussian with the target emittance.")
            P.x = np.random.normal(0, np.sqrt(emittanceX), N)
            P.px = np.random.normal(0, np.mean(P.pz)*np.sqrt(emittanceX), N)
        P.twiss_match(plane='x', beta = betaX, alpha = alphaX, inplace=True)
        if current_sqrt_emmitance!=0:
            if emittanceX is not None:
                P.x = P.x*np.sqrt(emittanceX)/current_sqrt_emmitance
                P.px = P.px*np.sqrt(emittanceX)/current_sqrt_emmitance
            
    #if doing second moments matrix instead of beta alpha emittance
    else:
        if not ((sigmaMatrixX[1,1]==-1) and (sigmaMatrixX[0,0]==-1)):
            if sigmaMatrixX[0][0]==-1:
                sigmaMatrixX[0][0] = np.std(P.x)**2
            if sigmaMatrixX[1][1]==-1:
                sigmaMatrixX[1][1] = np.std(P.xp)**2
            sigmaMatrixX[0][1] = correlations[0]*np.sqrt(sigmaMatrixX[0][0]*sigmaMatrixX[1][1])
            sigmaMatrixX[1][0] = sigmaMatrixX[0][1]
            if np.std(P.x)==0 or np.std(P.xp)==0:
                P.x = np.random.normal(0, 1, N)
                P.px = np.random.normal(0, np.mean(P.pz)*1, N)
            X = np.vstack((P.x, P.xp))
            Sigma_current = np.cov(X, bias=True)
            S_target_sqrt, _ = sqrtm_psd(sigmaMatrixX)
            S_current_invsqrt, _ = invsqrtm_psd(Sigma_current)
            T = S_target_sqrt @ S_current_invsqrt
            X_new = T @ X
            P.x = X_new[0]
            P.px = X_new[1]*np.mean(P.pz)

    if (betaY is not None) and (alphaY is not None):
        current_sqrt_emmitance = np.sqrt(np.sqrt( np.mean(P.y**2)*np.mean(P.yp**2) - np.mean(P.y*P.yp)**2 ))
        if current_sqrt_emmitance==0:
            print("edit_bunch_parameters_from_PG: Zero emittance in Y. Filling with Gaussian with the target emittance.")
            P.y = np.random.normal(0, np.sqrt(emittanceY), N)
            P.py = np.random.normal(0, np.mean(P.pz)*np.sqrt(emittanceY), N)
        P.twiss_match(plane='y', beta = betaY, alpha = alphaY, inplace=True)
        if current_sqrt_emmitance!=0:
            if emittanceY is not None:
                P.y = P.y*np.sqrt(emittanceY)/current_sqrt_emmitance
                P.py = P.py*np.sqrt(emittanceY)/current_sqrt_emmitance
    else:
        if not ((sigmaMatrixY[1,1]==-1) and (sigmaMatrixY[0,0]==-1)):
            if sigmaMatrixY[0][0]==-1:
                sigmaMatrixY[0][0] = np.std(P.y)**2
            if sigmaMatrixY[1][1]==-1:
                sigmaMatrixY[1][1] = np.std(P.yp)**2
            sigmaMatrixY[0][1] = correlations[1]*np.sqrt(sigmaMatrixY[0][0]*sigmaMatrixY[1][1])
            sigmaMatrixY[1][0] = sigmaMatrixY[0][1]
            if np.std(P.y)==0 or np.std(P.yp)==0:
                P.y = np.random.normal(0, 1, N)
                P.py = np.random.normal(0, np.mean(P.pz)*1, N)
            Y = np.vstack((P.y, P.yp))
            Sigma_current = np.cov(Y, bias=True)
            S_target_sqrt, _ = sqrtm_psd(sigmaMatrixY)
            S_current_invsqrt, _ = invsqrtm_psd(Sigma_current)
            T = S_target_sqrt @ S_current_invsqrt
            Y_new = T @ Y
            P.y = Y_new[0]
            P.py = Y_new[1]*np.mean(P.pz)

    # means after
    P.x = P.x + means[0]
    P.px = P.px + means[1]*np.mean(P.pz)
    P.y = P.y + means[2]
    P.py = P.py + means[3]*np.mean(P.pz)

    P.write(path_to_write+".h5")
    return P


def edit_bunch_parameters(file_ext, pzMeV=None, moments=None, correlations=None, means=None, charge=-1,
                          betaX=None, alphaX=None, emittanceX=None, betaY=None, alphaY=None, emittanceY=None, path_to_write='temp_beam/temp_e'):
    '''Edit the bunch parameters.
    file_ext: Base file path without the .h5 extension.

    moments are for x, xp, y, yp, z, pz (default: all None, i.e. unchanged)
    correlations are the correlation coefficients r(x,xp), r(y,yp), r(z,pz) with z = -c*t; None keeps the beam's value
    means are for x, xp, y, yp, c*t (default: 0, 0, 0, 0, 0); None keeps the beam's value
    xp and yp are in radians
    moments are RMS sizes

    if betaX and alphaX are supplied, the X phase space will have the emittance corresponding to moments[0] (size x = sqrt{epsilon beta}), and moments[1] will be overwritten. Same for Y.
    '''
    return edit_bunch_parameters_from_PG(ParticleGroup(file_ext + ".h5"), pzMeV=pzMeV, moments=moments, correlations=correlations, means=means, charge=charge,
                                  betaX=betaX, alphaX=alphaX, emittanceX=emittanceX, betaY=betaY, alphaY=alphaY, emittanceY=emittanceY, path_to_write=path_to_write)


def cut_length(particle_group, length = 0, drift_to_z = True):
    """Return a slice of the beam around its mean arrival time.

    Parameters:
        particle_group: Input ParticleGroup.
        length: Half-width of the longitudinal window in meters (particles with |c*(t - <t>)| < length are kept).
        drift_to_z: If False, the beam will be drift_to_t to <t> after the slicing.

    Returns:
        ParticleGroup: Sliced beam.
    """

    P = particle_group.copy()
    P.drift_to_z()
    indices = np.flatnonzero(np.abs(P.t - np.mean(P.t)) < length/3e8)
    if len(indices) == 0:
        raise ValueError(f"cut_length: no particles within {length:.3g} m of the bunch center "
                         f"({len(P)} particles in the beam); use a longer window or more particles")
    Ptemp = P[indices]
    if not drift_to_z:
        Ptemp.drift_to_t()
    return Ptemp


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


    #For backwards compatibility, return to activeBeamFile. Might be unnecessary
    # tao.cmd(f'set beam_init position_file={filePathGlobal}/beams/activeBeamFile.h5')
    # tao.cmd('reinit beam')

# def trackBeamLEGACY(tao):
#     #This is the pre-2024-08-23 version of trackBeam(), retained for debugging purposes. Can be deleted
    
#     tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
#     tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation


def ballisticPropagation(P, distance):
    """ Propagates ParticleGroup P ballistically over some distance
    
    Parameters
    ----------
    P: OpenPMD ParticleGroup
    distance: propagation distance [m]

    """
    P.x = P.x + (P['px']/P['pz']) * distance
    P.y = P.y + (P['py']/P['pz']) * distance
    P.t = P.t + distance/299792458


def nudgeMacroparticleWeights(
    PInput,
    trailingBunchFraction = None,
    trailingBunchType = None
):
    """
    This is NOT a robust function. Don't trust it to do what you want
    Presently splits based on z and a user-specified charge ratio. Lots of things can go wrong if you aren't careful!
    
    Borrowing stuff from 2024-03-29_nudgeMacroparticleWeights.ipynb
    """

    P = PInput.copy()

    zVals = (P["delta_z"]).copy()
    zVals = np.sort(zVals)
    
    splitZ = zVals[int(trailingBunchFraction * len(zVals))] 
    
    
    
    startingWeight = P.weight[0]
    startingWeight
    
    witnessWeight = 0.999*startingWeight
    driverWeight = 1.001*startingWeight
    
    if trailingBunchType == "witness":
        trailingBunchWeight = witnessWeight
        leadingBunchWeight = driverWeight
    if trailingBunchType == "driver": 
        trailingBunchWeight = driverWeight
        leadingBunchWeight = witnessWeight
    
    newWeightArr = np.full(np.size(P.weight), -1.1)
    for i in range(np.size(newWeightArr)):
        if P["delta_z"][i] < splitZ:
            newWeightArr[i] = trailingBunchWeight
        else:
            newWeightArr[i] = leadingBunchWeight
    
    P.weight = newWeightArr

    return P


def getDriverAndWitness(P):
    """Splits a beam by unique weights into  drive and witness beam objects
    
    See, e.g. "2024-07-01 Nudge Macroparticle Weights.ipynb" for details
    """
    
    weights = np.sort(np.unique(P.weight))
    if len(weights) != 2:
        print("WARNING! Expected drive/witness structure not found")
        return
    PWitness = P[P.weight == weights[0]]
    PDrive = P[P.weight == weights[1]]
    return PDrive, PWitness


def centerBeam(
    P,
    centerType = "median",
    assertEnergy = None
):
    """
    Shifts x, y, xp, and yp of a beam to zero
    centerType is either "median" or "mean"
    """
    
    PMod = P
    if centerType == "median":
        PMod.x = P.x - np.median(P.x)
        PMod.y = P.y - np.median(P.y)
        PMod.px = P.px - np.median(P.px)
        PMod.py = P.py - np.median(P.py)
        if assertEnergy:
            PMod.pz = P.pz * assertEnergy / np.median(P.pz)
        
        return PMod
        
    if centerType == "mean":
        PMod.x = P.x - np.mean(P.x)
        PMod.y = P.y - np.mean(P.y)
        PMod.px = P.px - np.mean(P.px)
        PMod.py = P.py - np.mean(P.py)
        if assertEnergy:
            PMod.pz = P.pz * assertEnergy / np.mean(P.pz)
                    
        return PMod

    return


def collimateBeam(
    P,
    allCollimatorRules = None
):
    """
    allCollimatorRules is a list of lists. Each sublist should have exactly two elements for the lower and upper x position of a collimator
    Arbitrarily many collimators can be defined this way; therefore it works for notch and/or jaw collimators
    """
    PMod = P.copy()


    for collimatorRange in allCollimatorRules:

        print(collimatorRange)
        all_indices = np.arange(len(PMod.x))
        killedIndices = np.where(np.logical_and(PMod.x > collimatorRange[0], PMod.x < collimatorRange[1]))[0]
        survivingIndices = np.setdiff1d(all_indices, killedIndices)
        
        # OpenPMD checks the length so I can't just remove the "killed" particles
        # Also, for compatibility, I don't want to change either the weight or status of the killed particles
        filtered_data = {
            "x": PMod.x[survivingIndices],
            "y": PMod.y[survivingIndices],
            "z": PMod.z[survivingIndices],
            "px": PMod.px[survivingIndices],
            "py": PMod.py[survivingIndices],
            "pz": PMod.pz[survivingIndices],
            "t": PMod.t[survivingIndices], 
            "status": PMod.status[survivingIndices], 
            "weight": PMod.weight[survivingIndices], 
            "species": PMod.species
        }
        
        # Create a new ParticleGroup instance with the filtered data
        PMod = ParticleGroup(data=filtered_data)
        print(f"New particle count: {len(PMod.x)}")
        print(f"{len(PMod.x)}")

    return PMod


def sortIndices(lst):
    #Returns the indices of the sorted elements, e.g. [1, 3, 5, 2, 4] --> [0, 3, 1, 4, 2]
    return [i for i, _ in sorted(enumerate(lst), key=lambda x: x[1])]


def sliceBeam(
    P,
    sortKey = None,
    numBeamlets = None
):
    """Sort a beam by sortKey, then slice it into numBeamlets of equal count"""
    sortedIndices = sortIndices(P[sortKey])
    
    subsetIndices = np.array_split(sortedIndices, numBeamlets)
    
    resultBeamlets = []
    
    for activeSubsetIndices in subsetIndices:
        PMod = P.copy()
        
        # OpenPMD checks the length so I can't just remove the "killed" particles
        # Also, for compatibility, I don't want to change either the weight or status of the killed particles
        filtered_data = {
            "x": PMod.x[activeSubsetIndices],
            "y": PMod.y[activeSubsetIndices],
            "z": PMod.z[activeSubsetIndices],
            "px": PMod.px[activeSubsetIndices],
            "py": PMod.py[activeSubsetIndices],
            "pz": PMod.pz[activeSubsetIndices],
            "t": PMod.t[activeSubsetIndices], 
            "status": PMod.status[activeSubsetIndices], 
            "weight": PMod.weight[activeSubsetIndices], 
            "species": PMod.species
        }
        
        # Create a new ParticleGroup instance with the filtered data
        PMod = ParticleGroup(data=filtered_data)
        #print(f"New particle count: {len(PMod.x)}")
        #print(f"{len(PMod.x)}")
    
        resultBeamlets.append(PMod)

    return resultBeamlets


def getSingleBeamSlice(
    P,
    sortKey = None,
    minVal = None,
    maxVal = None
):
    """Return a beamlet of particles which satisfy the inequality """

    # Get indices where sortKey is within the given range
    mask = (P[sortKey] >= minVal) & (P[sortKey] <= maxVal)
    
    if not np.any(mask):
        raise ValueError("No particles found in the specified range.")
    
    # Filter data based on the mask
    filtered_data = {
        "x": P.x[mask],
        "y": P.y[mask],
        "z": P.z[mask],
        "px": P.px[mask],
        "py": P.py[mask],
        "pz": P.pz[mask],
        "t": P.t[mask],
        "status": P.status[mask],
        "weight": P.weight[mask],
        "species": P.species
    }


    PMod = ParticleGroup(data=filtered_data)

    return PMod
