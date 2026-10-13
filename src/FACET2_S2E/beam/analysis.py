"""Beam statistics: spot sizes, emittances, beam specs.

Moved from UTILITY_quickstart.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import curve_fit

from .manipulation import getDriverAndWitness, sliceBeam


def calcBMAG(b0, a0, b, a):
    #From Lucretia
    #For a bit more detail, see "BMAG from Lucretia.nb"
    #Not validated!!!

    # function [B,Bpsi]=bmag(b0,a0,b,a)
    # %
    # % [B,Bpsi]=bmag(b0,a0,b,a);
    # %
    # % Compute BMAG and its phase from Twiss parameters
    # %
    # % INPUTs:
    # %
    # %   b0 = matched beta
    # %   a0 = matched alpha
    # %   b  = mismatched beta
    # %   a  = mismatched alpha
    # %
    # % OUTPUTs:
    # %
    # %   B    = mismatch amplitude
    # %   Bpsi = mismatch phase (deg)

    g0 = (1 + a0 ** 2) / b0
    g  = (1 + a ** 2) / b
    B  = (b0 * g - 2.0 * a0 * a + g0 * b) / 2

    return B


def smallestInterval(nums, percentage=0.9):
    """Give the smallest interval containing a desired percentage of provided points"""
    nums = nums.copy()  # Create a copy to avoid modifying the original list
    nums.sort()
    n = len(nums)
    k = int(n * percentage)
    min_range = float('inf')
    interval = (None, None)
    
    for i in range(n - k + 1):
        current_range = nums[i + k - 1] - nums[i]
        if current_range < min_range:
            min_range = current_range
            interval = (nums[i], nums[i + k - 1])
    
    return interval[1]-interval[0]


def smallestIntervalImpliedSigma(nums, percentage=0.9):
    """Given a smallestInterval, infer the sigma assuming the distribution was Gaussian

    See "Discussion of alternative emittance and spot size calculations.ipynb"
    See also "2024-07-01 RMS vs FWHM at PENT.ipynb"
    """
    interval = smallestInterval(nums, percentage)
    intervalToSigmaFactor = scipy.special.erfinv(percentage) * (2 * np.sqrt(2))
    return interval/intervalToSigmaFactor


def smallestIntervalImpliedEmittanceModelFunction(z, sigmax, sigmaxp, rho):

    return np.sqrt(np.clip(
        sigmax**2 + 2 * z * rho * sigmax * sigmaxp + z**2 * sigmaxp**2,
        0, np.inf))


def smallestIntervalImpliedEmittance(P, plane = "x", percentage = 0.9, verbose = False):
    """Use a virtual quad scan and smallestInterval measurements to calculate a beam's emittance"""
    
    #zValues = np.arange(-20, 20, 0.1)
    zValues = np.array([-131.072, -65.536, -32.768, -16.384, -8.192, -4.096, -2.048, -1.024, 
               -0.512, -0.256, -0.128, -0.064, -0.032, -0.016, -0.008, -0.004, 
               -0.002, -0.001, 0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 
               0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096, 8.192, 
               16.384, 32.768, 65.536, 131.072]) #(-Reverse[#])~Join~{0}~Join~# &@PowerRange[0.001, 200, 2]
    
    if plane == "x":
        sigmaXResults = [ smallestIntervalImpliedSigma(P.x + z * P.xp, percentage = percentage) for z in zValues]
        sigmaXResultsExact = [ np.std(P.x + z * P.xp) for z in zValues]
    elif plane == "y":
        sigmaXResults = [ smallestIntervalImpliedSigma(P.y + z * P.yp, percentage = percentage) for z in zValues]
        sigmaXResultsExact = [ np.std(P.y + z * P.yp) for z in zValues]
    else:
        return

    #sigmaXResults = [ smallestIntervalImpliedSigma(P.x + z * P.xp, percentage = percentage) for z in zValues]
    #sigmaXResultsExact = [ np.std(P.x + z * P.xp) for z in zValues]
    
    
    # Fit the model to the data
    popt, pcov = curve_fit(smallestIntervalImpliedEmittanceModelFunction, zValues, sigmaXResults, p0=[1, 1, 0])
    
    # Extract optimal parameters
    sigmax_opt, sigmaxp_opt, rho_opt = popt


    
    emit_opt = np.sqrt(np.clip( 
        sigmax_opt**2 * sigmaxp_opt**2 - (rho_opt * sigmax_opt * sigmaxp_opt)**2 ,
        0, np.inf))


    if verbose:
        print(f"""True sigma_x, sigma_xp, rho: {P.std("x")}, {P.std("xp")}, {P["cov_x__xp"] / (P.std("x") * P.std("xp"))}""")
        print(f"Optimizer parameters: sigma_x = {sigmax_opt}, sigma_xp = {sigmaxp_opt}, rho = {rho_opt}")
    

        plt.scatter(zValues, sigmaXResultsExact, label='True rms')
        plt.scatter(zValues, sigmaXResults, label='Inferred rms')
        plt.plot(zValues, smallestIntervalImpliedEmittanceModelFunction(zValues, *popt), label='Fitted function', color='red')
        plt.xlabel('Drift [m]')
        plt.ylabel('Sigma_x [m]')
        plt.legend()
        plt.show()

        print(f"""Actual emittance: \t {P["norm_emit_x"]}""")
        print(f"""Fit emittance: \t\t {emit_opt * P["mean_gamma"]}""")

    return emit_opt * P["mean_gamma"]


def emittance(P, plane = "x", fraction = 0.9):
    """Just a wrapper for the OpenPMD functions which let you specify the fraction used in the emittance calculation"""
    return P.twiss(plane = plane, fraction = fraction)[f"norm_emit_{plane}"]


def getBeamSpecs(P, targetTwiss = None):
    """
    Returns a collection of convenient beam parameters as a dictionary
    Will automatically detect and add extra measurements for two-bunch beams
    
    targetTwiss can either be in the form [betaX, alphaX, betaY, alphaY] or
    for a very limited number of treaty point elements, can instead provide the element name. This will use the golden lattice targetTwiss
    Presently defined: "PR10571", "BEGBC20", "MFFF", "PENT"
    """
    
    savedData = {}

    
    #A silly trick; set() will give back unique values
    bunchCount = len(set(P.weight))
    
    if bunchCount == 1:
        PDrive = P.copy()
        beamsToEvaluate = ["PDrive"]
    elif bunchCount == 2:
        PDrive, PWitness = getDriverAndWitness(P)
        beamsToEvaluate = ["PDrive", "PWitness"]
    else:
        print("bunchCount doesn't make sense. Aborting")
        return



    if targetTwiss:
        if isinstance(targetTwiss, str): 
            if targetTwiss == "PR10571":
                #PR10571 lucretia live model lattice 2024-10-16
                targetBetaX = 5.7
                targetBetaY = 2.6
                targetAlphaX = -2.1
                targetAlphaY = 0.0
                
            elif targetTwiss == "BEGBC20":
                #BEGBC20 lucretia live model lattice 2024-10-16
                targetBetaX = 11.5
                targetBetaY = 27.3
                targetAlphaX = 0.7
                targetAlphaY = 1.2
            
            elif targetTwiss == "MFFF":
                #MFFF lucretia live model lattice 2024-10-16
                targetBetaX = 11.6
                targetAlphaX = -0.64
                targetBetaY = 25.2
                targetAlphaY = -1.6
        
            elif targetTwiss == "PENT":
                #PENT lucretia live model lattice 2024-10-16
                targetBetaX = 0.5
                targetAlphaX = 0.0
                targetBetaY = 0.5
                targetAlphaY = 0.0

            else:
                print("Not a valid treaty point. Aborting")
                #return

                
                #Invalid treaty point; setting to None to avoid BMAG evaluation
                targetTwiss = None


        else:
            targetBetaX, targetAlphaX, targetBetaY, targetAlphaY = targetTwiss

    
    
    for PActiveStr in beamsToEvaluate:
        PActive = locals()[PActiveStr]

        
        # for val in ["mean_x", "mean_y", "sigma_x", "sigma_y", "mean_xp", "mean_yp"]:
        #     savedData[f"{PActiveStr}_{val}"] = PActive[val]

        
        savedData[f"{PActiveStr}_median_x"] = np.median(PActive.x)
        savedData[f"{PActiveStr}_median_y"] = np.median(PActive.y)

        savedData[f"{PActiveStr}_median_xp"] = np.median(PActive.xp)
        savedData[f"{PActiveStr}_median_yp"] = np.median(PActive.yp)
        savedData[f"{PActiveStr}_median_energy"] = np.median(PActive.energy)
        
        savedData[f"{PActiveStr}_sigmaSI90_x"] = smallestIntervalImpliedSigma(PActive.x, percentage = 0.90)
        savedData[f"{PActiveStr}_sigmaSI90_y"] = smallestIntervalImpliedSigma(PActive.y, percentage = 0.90)
        savedData[f"{PActiveStr}_sigmaSI90_z"] = smallestIntervalImpliedSigma(PActive.t * 3e8, percentage=0.9)

        savedData[f"{PActiveStr}_sigmaSI90_xp"] = smallestIntervalImpliedSigma(PActive.xp, percentage = 0.90)
        savedData[f"{PActiveStr}_sigmaSI90_yp"] = smallestIntervalImpliedSigma(PActive.yp, percentage = 0.90)
        savedData[f"{PActiveStr}_sigmaSI90_energy"] = smallestIntervalImpliedSigma(PActive.energy, percentage = 0.90)

        savedData[f"{PActiveStr}_emitSI90_x"] = smallestIntervalImpliedEmittance(PActive, plane = "x", percentage = 0.90)
        savedData[f"{PActiveStr}_emitSI90_y"] = smallestIntervalImpliedEmittance(PActive, plane = "y", percentage = 0.90)

        savedData[f"{PActiveStr}_norm_emit_x"] = (PActive.twiss(plane = "x", fraction = 0.9))["norm_emit_x"]
        savedData[f"{PActiveStr}_norm_emit_y"] = (PActive.twiss(plane = "y", fraction = 0.9))["norm_emit_y"]

        if bunchCount == 2:
            savedData[f"{PActiveStr}_zCentroid"] = np.median(PActive.t * 3e8)

        savedData[f"{PActiveStr}_charge_nC"] = PActive.charge * 1e9


        PActiveTwiss = PActive.twiss(plane = "x", fraction = 0.9) | PActive.twiss(plane = "y", fraction = 0.9)

        if targetTwiss: 
            savedData[f"{PActiveStr}_BMAG_x"] = calcBMAG(targetBetaX, targetAlphaX, PActiveTwiss["beta_x"], PActiveTwiss["alpha_x"])
            savedData[f"{PActiveStr}_BMAG_y"] = calcBMAG(targetBetaY, targetAlphaY, PActiveTwiss["beta_y"], PActiveTwiss["alpha_y"])
    
            # Get BMAGs by energy slice
            slicedBeamlets = sliceBeam( PActive , sortKey = "pz", numBeamlets = 5 )
    
            slicedTwiss =  [ ( beamlet.twiss(plane = "x", fraction = 0.9) | beamlet.twiss(plane = "y", fraction = 0.9) ) for beamlet in slicedBeamlets ] 
            
            savedData[f"{PActiveStr}_sliced_BMAG_x"] = [ calcBMAG(targetBetaX, targetAlphaX, beamletTwiss["beta_x"], beamletTwiss["alpha_x"]) for beamletTwiss in slicedTwiss ]
            savedData[f"{PActiveStr}_sliced_BMAG_y"] = [ calcBMAG(targetBetaY, targetAlphaY, beamletTwiss["beta_y"], beamletTwiss["alpha_y"]) for beamletTwiss in slicedTwiss ]

    if bunchCount == 2:
        savedData["bunchSpacing"] = savedData["PWitness_zCentroid"] - savedData["PDrive_zCentroid"]

        savedData["transverseCentroidOffset"] = np.sqrt(
                (savedData["PDrive_median_x"] - savedData["PWitness_median_x"])**2 + 
                (savedData["PDrive_median_y"] - savedData["PWitness_median_y"])**2
            )

    #savedData["lostChargeFraction"] = 1 - (P.charge / PInit.charge)

    return savedData


def generalizedEmittanceSolverObjective(params, data):
    betaI, alphaI, emittanceGeo = params
    
    errorComponents = []

    for shot in data:
        # Twiss transfer matrix for beta
        # R11^2 \[Beta] - 2 R11 R12 \[Alpha] +  + R12^2 \[Gamma]
        term1 = betaI * shot["R11"] ** 2
        term2 = -2 * alphaI * shot["R11"] * shot["R12"]
        term3 =  ( (1 + alphaI ** 2) / betaI ) * shot["R12"] ** 2

        # (beta * emit_geo) == sigma^2
        errorComponent = ( term1 + term2 + term3 ) * emittanceGeo - shot["sigma"] ** 2

        #Add all error terms in quadrature
        errorComponents.append(errorComponent ** 2)
    
        
    
    return np.sum(errorComponents)


def generalizedEmittanceSolver(
    data,
    energyGeV = None,
    verbose = False,
    initialGuess = [0.5, 0.5, 1e-9],
    **kwargs
):
    """
    `data` should be a list of dictionaries, each of which should contain at least "R11", "R12", and "sigma" corresponding to the R-matrix terms for the transfer of interest
    and the beam sigma at the downstream screen.

    The the initial beta, alpha, and emittance are used as fit parameters to explain the observations
    The typical Twiss transfer is applied for each case and compared to the observed spot size

    If this isn't giving the values you expect, make sure the single-particle Twiss values are what you expect! (Consider running launchTwissCorrection())
    """
    
    from scipy.optimize import minimize


    # Perform optimization using Nelder-Mead
    result = minimize(
        generalizedEmittanceSolverObjective, 
        initialGuess, #Starting point
        method='Nelder-Mead',
        args = (data, ),
        **kwargs
    )


    if verbose:
        print("Optimization Results:")
        print(f"Optimal Parameters: {result.x}")
        print(f"Objective Function Value at Optimal Parameters: {result.fun}")
        print(f"Number of Iterations: {result.nit}")
        print(f"Converged: {result.success}")

    output = {"beta" : result.x[0], "alpha" : result.x[1], "emitGeo" : result.x[2]}

    if energyGeV:
        #Sloppy, ultrarel only
        output["emit"] = result.x[2] * energyGeV * 1000 / 0.511
    
    return output
