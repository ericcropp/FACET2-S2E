"""Run a tracked simulation on an initialized Tao object, saving beams at checkpoints.

Also holds edit_energy_based_on_beam_all, which iterates simulation runs and
energy retuning (simulation.energy).

Moved from simulationFunctions.py.
"""

import numpy as np

from ...beam.manipulation import modifyInputBeamSimple, edit_bunch_parameters_from_PG, edit_bunch_parameters
from .core import trackBeam, getBeamAtElement
from .energy import treat_dipoles, tune_to_P0Cs, default_bend_fields, edit_energy_based_on_beam_inj, edit_energy_based_on_beam_L1, edit_energy_based_on_beam_L2, edit_energy_based_on_beam_L3


def set_beam(tao, file, numMacroParticles = None, timeCenterTF=True):
    '''
    Sets the beam in the tao object. The beam is edited: drift to z, z=0. t=0 if time_centering.
    '''
    #filePath = os.getcwd()
    #file_e = f'{filePath}/beams/activeBeamFile.h5'
    file_e = file + "_e"
    #Write as the active file
    #modifyInputBeamSimple(folder + file + ".h5", numMacroParticles).write(file_e + ".h5")
    modifyInputBeamSimple(file + ".h5", numMacroParticles, timeCenterTF=timeCenterTF).write(file_e + ".h5")
    #tao.cmd(f'set beam_init position_file={folder + file_e + ".h5"}')
    tao.cmd(f'set beam_init position_file={file_e + ".h5"}')
    tao.cmd('reinit beam')


def run_initialized_sim(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], treat_dipoles_TF=False):
    '''
    If treat_dipoles_TF is False, this function just tracks from start to finish, saving the beam at the locations in "locations".
    If treat_dipoles_TF is True, before tracking it loads the constant fields from a "nominal" experiment lattice (default_bend_fields).

    '''
    if locations==[]:
        locations = [start, finish]
    if locations==None:
        locations=[]

    if treat_dipoles_TF:
        tao = treat_dipoles(tao)
    trackBeam(tao, filepath=tao.filePathGlobal, trackStart = start, trackEnd = finish, autoLoadActiveFile=False)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind], tToZ=False)
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


def run_initialized_sim_edit_lattice_energy_for_dipoles(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], desired_P0Cs_MeV=[None,None,None,None]):
    '''
    Old function to change the bend energies without DB_Field.
    If desired_P0Cs_MeV are None, this function just tracks from start to finish, saving the beam at the locations in "locations".
    If desired_P0Cs_MeV is something else, it:
    tunes the cavities to the desired_P0Cs_MeV
    tracks to the BX0FBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BX0FEND
    tunes them to the desired_P0Cs_MeV
    tracks to the BC11CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC11CEND
    tunes them to the desired_P0Cs_MeV
    tracks to the BC14CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC14CEND
    tunes them to the desired_P0Cs_MeV
    tracks to the BC20CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC20CEND
    tunes them to the desired_P0Cs_MeV
    tracks to the finish

    '''
    if locations==[]:
        locations = [start, finish]
    if locations==None:
        locations=[]

    current_start = start
    # injector
    if desired_P0Cs_MeV[0] is not None:
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV, change_only_L0B=True)
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BX0FBEG", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BX0FBEG", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000], change_only_L0B=True)        
        current_start = "BX0FBEG"
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BX0FEND", autoLoadActiveFile=False)
        #print(f'<x> inside of the dogleg: {tao.bunch_params("BPM10731")["centroid_vec_1"]}')
        getBeamAtElement(tao, "BX0FEND", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BX0FEND"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # L1
    if desired_P0Cs_MeV[1] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BC11CBEG", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BC11CBEG", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        current_start = "BC11CBEG"
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BC11CEND", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BC11CEND", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BC11CEND"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # L2
    if desired_P0Cs_MeV[2] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BEGBC14E", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BEGBC14E", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        current_start = "BEGBC14E"
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDBC14E", autoLoadActiveFile=False)
        getBeamAtElement(tao, "ENDBC14E", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDBC14E"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # L3
    if desired_P0Cs_MeV[3] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BEGBC20", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BEGBC20", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        current_start = "BEGBC20"
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDBC20", autoLoadActiveFile=False)
        getBeamAtElement(tao, "ENDBC20", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDBC20"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # Fin
    trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = finish, autoLoadActiveFile=False)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind], tToZ=False)
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


def run_initialized_sim_edit_bunch_energy(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], edited_bunch_energy_at_checkpoints_MeV=[None,None,None,None]):
    '''
    Tracks the bunch and changes the bunch energy at BX0FBEG, BC11CBEG, ENDL2F, ENDL3F_2 if any of edited_bunch_energy_at_checkpoints_MeV (list with 4 numbers) is not None.
    '''
    if locations==[]:
        locations = [start, finish]
    if locations==None:
        locations=[]
    current_start = start
    if edited_bunch_energy_at_checkpoints_MeV[0] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BX0FBEG", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "BX0FBEG", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[0], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BX0FBEG"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[1] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BC11CBEG", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "BC11CBEG", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[1], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BC11CBEG"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[2] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDL2F", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "ENDL2F", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[2], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDL2F"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[3] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDL3F_2", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "ENDL3F_2", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[3], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDL3F_2"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    
    trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = finish, autoLoadActiveFile=False)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind], tToZ=False)
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


def edit_energy_based_on_beam_all(tao, start, file, desired_beam_energies=None, change_only_L0B=False, change_file_pz=True, verbose=False, finalnumMacroParticles=5e4, tune_dipoles=True, dipole_fields=default_bend_fields):
    '''
    This function changes the cavity voltages so that the tracked beam has the "desired_P0Cs" <pz> between the cavities.
    desired_P0Cs: the desired <pz> in eV between the cavities. Must be None or a list of 4 numbers (dogleg, bc11, bc14, bc20).
    If None, the P0Cs from the lattice are used.
    start: start of the simulation. Need to be in the injector (use L0AFEND to avoid confusion).
    file: the bunch file to use for the tuning. Providing it is a must because the energy obviously depends on the charge and the size of the bunch.
    change_only_L0B: in the injector, scale only L0B. I think that at FACET we do exactly that.
    If changing both L0A and L0B, the input beam energy at L0A is adjusted accordingly later.
    change_file_pz: change the bunch energy at the "start" to match the "start" P0C.
    finalnumMacroParticles: the function uses "set_beam()" at the end. If you don't want to set the beam manually after the tuning,
    this option allows you to regulate the number of particles in that set beam (the one you provided with the "file").
    '''
    if desired_beam_energies is None:
        desired_beam_energies = [float(tao.ele_gen_attribs("BX0FBEG")["P0C"]), float(tao.ele_gen_attribs("BC11CBEG")["P0C"]), float(tao.ele_gen_attribs("ENDL2F")["P0C"]), float(tao.ele_gen_attribs("ENDL3F_2")["P0C"])]
    pre = 'temp_beam/'
    suf = 'temp'
    if verbose:
        print(f'P0C now: {desired_beam_energies}')
    
    # injector
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "BX0FBEG"]
    if tune_dipoles:
            tao = treat_dipoles(tao, dipole_fields)
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to BX0FBEG: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_inj(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[0]*1e-6, change_only_L0B=change_only_L0B)
    if verbose:
        print(f'edited P0C BX0FBEG: {tao.ele_gen_attribs("BX0FBEG")["P0C"]}')
    
    # L1
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "BC11CBEG"]

    if tune_dipoles:
            tao = treat_dipoles(tao, dipole_fields)
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to BC11CBEG: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_L1(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[1]*1e-6)
    if verbose:
        print(f'edited P0C BC11CBEG: {tao.ele_gen_attribs("BC11CBEG")["P0C"]}')
    
    # L2
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "ENDL2F"]
    
    if tune_dipoles:
            tao = treat_dipoles(tao, dipole_fields)
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to ENDL2F: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "ENDL2F", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_L2(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[2]*1e-6)
    if verbose:
        print(f'edited P0C ENDL2F: {tao.ele_gen_attribs("ENDL2F")["P0C"]}')
    
    # L3
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "ENDL3F_2"]
    
    if tune_dipoles:
            tao = treat_dipoles(tao, dipole_fields)
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to ENDL3F_2: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "ENDL2F", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "ENDL3F_2", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_L3(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[3]*1e-6)
    if verbose:
        print(f'edited P0C ENDL3F_2: {tao.ele_gen_attribs("ENDL3F_2")["P0C"]}')
    
    set_beam(tao, file, numMacroParticles=finalnumMacroParticles)
    return tao
