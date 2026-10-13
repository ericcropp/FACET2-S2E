"""Transfer matrices, Taylor-map elements, Twiss matching.

Moved from UTILITY_quickstart.py (matrices, BMAG, Twiss correction) and
simulationFunctions.py (get_element_array, get_rij, get_tijk).
"""

import numpy as np
import pandas as pd
from IPython.display import display

from .set_lattice import setLattice
from ..config import loadConfig


    
def displayMatrix(matrix):
    display(pd.DataFrame(matrix).style.hide(axis="index").hide(axis="columns"))


def getMatrix(tao, start, end, order = 1, startOffset = 0, endOffset = 0, print = False):
    """Return zero or first order transport matrix from start to end, with offsets. Optionally print in a human readable format"""
    
    startS = tao.ele_head(start)["s"] + startOffset
    endS = tao.ele_head(end)["s"] + endOffset

    #This disgusting workaround is required because PyTao implementation of taylor_map doesn't support -s mode
    if order == 0: 
        allStrings = tao.show(f"taylor_map -order 0 -s {startS} {endS}")
        transportMatrix = np.array([float(str) for str in allStrings[0].split()])
    
    elif order == 1:
        allStrings = tao.show(f"taylor_map -order 1 -s {startS} {endS}")
        splitStrings = [str.split() for str in allStrings]
        transportMatrix = np.array([ [float(str) for str in row[0:6]] for row in splitStrings[2:] ])

    else:
        print("Invalid matrix order requested")
        return
    

    if print:
        displayMatrix(transportMatrix)
        
    return transportMatrix


def getMatrixLEGACY(tao, start, end, order = 1, print = False):
    """Return zero or first order transport matrix from start to end. Optionally print in a human readable format"""

    if order == 0: 
        transportMatrix = (tao.matrix(start, end))["vec0"]
    elif order == 1:
        transportMatrix = (tao.matrix(start, end))["mat6"]
    else:
        print("Invalid matrix order requested")
        return
    
    
    if print:
        #display(pd.DataFrame(transportMatrix).style.hide(axis="index").hide(axis="columns"))
        displayMatrix(transportMatrix)
        
    return transportMatrix


def setLatticeAndGetMatrix(tao, start, end, startOffset = 0, endOffset = 0, defaultSettings = None, overrideSettings = {}):
    """
    Load lattice based on default and override settings. Then provide the transfer matrix between two elements, optionally with offsets
    If no defaultSettings are specified, the golden lattice from defaults.yml will be loaded
    """
    
    if not defaultSettings:
        defaultSettings = loadConfig(f"{tao.filePathGlobal}/setLattice_configs/defaults.yml")
        
    setLattice(tao, **( defaultSettings | overrideSettings ) )
   
    return getMatrix(tao, start, end, startOffset = startOffset, endOffset = endOffset)


#Here's a version that would work if the axes are coupled.... they really, really, really shouldn't ever be though
def launchTwissCorrectionObjective(params, tao, evalElement, targetBetaX, targetAlphaX, targetBetaY, targetAlphaY):
    betaSetX, alphaSetX, betaSetY, alphaSetY = params
    
    try:
        #Prevent recalculation until changes are made
        tao.cmd("set global lattice_calc_on = F")
        
        tao.cmd(f"set element beginning beta_a = {betaSetX}")
        tao.cmd(f"set element beginning alpha_a = {alphaSetX}")
        tao.cmd(f"set element beginning beta_b = {betaSetY}")
        tao.cmd(f"set element beginning alpha_b = {alphaSetY}")
        
        #Reenable lattice calculations
        tao.cmd("set global lattice_calc_on = T")
    
    except: #If Bmad doesn't like the proposed solution, don't crash, give a bad number
        return 1e20
    
    return (tao.ele_twiss(evalElement)[f"beta_a"] - targetBetaX) ** 2 + (tao.ele_twiss(evalElement)[f"alpha_a"] - targetAlphaX) ** 2 + (tao.ele_twiss(evalElement)[f"beta_b"] - targetBetaY) ** 2 + (tao.ele_twiss(evalElement)[f"alpha_b"] - targetAlphaY) ** 2


def launchTwissCorrection(tao, 
                          evalElement = None, 
                          targetBetaX = None, 
                          targetAlphaX = None, 
                          targetBetaY = None, 
                          targetAlphaY = None
                         ):
    """
    This function will update the BEGINNING twiss values (set in bmad/models/f2_elec/f2_elec.lat.bmad, e.g. BEGINNING[BETA_A] =  1.39449126865854395E-001) to achieve an arbitrary match at an arbitrary element.

    If no element is specified, the function will create the default golden lattice match at PR10571
    """
    
    from scipy.optimize import minimize

    if not evalElement:
        print("No evalElement provided. Assuming golden lattice PR10571")
        evalElement = "PR10571"
        targetBetaX = 5.73666431
        targetAlphaX = -2.14411559
        targetBetaY = 2.57530302
        targetAlphaY = 0.01016211

    # Perform optimization using Nelder-Mead
    result = minimize(
        launchTwissCorrectionObjective, 
        [0.137, 0.954, 0.406, 2.20], #Starting point
        #[101, -26, 101, -26],
        method='Nelder-Mead',
        bounds = [(1e-9, 1e3), (-100, 100), (1e-9, 1e3), (-100, 100)],
        args = (tao, evalElement, targetBetaX, targetAlphaX, targetBetaY, targetAlphaY)
    )


    #Apply best result to the lattice
    betaSetX, alphaSetX, betaSetY, alphaSetY = result.x
    
    #Prevent recalculation until changes are made
    tao.cmd("set global lattice_calc_on = F")
    
    tao.cmd(f"set element beginning beta_a = {betaSetX}")
    tao.cmd(f"set element beginning alpha_a = {alphaSetX}")
    tao.cmd(f"set element beginning beta_b = {betaSetY}")
    tao.cmd(f"set element beginning alpha_b = {alphaSetY}")
    
    #Reenable lattice calculations
    tao.cmd("set global lattice_calc_on = T")
                          
    print("Optimization Results:")
    print(f"Optimal Parameters: {result.x}")
    print(f"Objective Function Value at Optimal Parameters: {result.fun}")
    print(f"Number of Iterations: {result.nit}")
    print(f"Converged: {result.success}")

    return


    

## Simulation parameters functions

def get_element_array(tao, beg, end, values_to_show=[], values_to_remove=[], marginl=0, marginr=0):
    """Return a subset of lattice elements between two locations, filtered by element type."""
    keys = tao.lat_list("*", "ele.key")
    ss = tao.lat_list("*", "ele.s")
    names = tao.lat_list("*", "ele.name")
    
    elements = np.stack([ss, names, keys], axis=1)
    
    location1 = np.argwhere(elements==beg)[0,0]-marginl
    location2 = np.argwhere(elements==end)[0,0]+marginr
    
    trunc_array = elements[location1:location2+1]
    show_array = trunc_array[np.isin(trunc_array[:, 2], values_to_show)] if (len(values_to_show)>0) else trunc_array
    clean_array = show_array[~np.isin(show_array[:, 2], values_to_remove)] if (len(values_to_remove)>0) else show_array
    
    return clean_array


def get_tijk(tao, loc1, loc2, i_0, j_0, k_0):
    """Read a second-order Taylor map coefficient (T) from Tao between two locations."""
    n = 6
    t5ijterms = [
        [
            tuple(1 if k == i or k == j else 0 for k in range(n)) if i != j
            else tuple(2 if k == i else 0 for k in range(n))
            for j in range(n)
        ]
        for i in range(n)
    ]
    mapterms = tao.taylor_map(loc1, loc2, order='2', verbose=False, as_dict=True, raises=True)[i_0]
    return mapterms[t5ijterms[j_0-1][k_0-1]] if t5ijterms[j_0-1][k_0-1] in mapterms else 0


def get_rij(tao, loc1, loc2, i, j):
    """Read an R-matrix element from Tao between two locations."""
    r5i = np.zeros(6)
    s = tao.cmd("".join(["show matrix ", loc1, " ", loc2]))[i+1]
    numeric_part = s.split(':')[0]
    nums = [float(x) for x in numeric_part.split()]
    r5i = np.array(nums)
    return float(r5i[j-1])
