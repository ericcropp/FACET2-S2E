"""Loading and applying lattice/simulation configuration files.

Moved from UTILITY_quickstart.py.
"""

import yaml


def loadConfig(file, filepath, loaded_files=None):
    """Code to load nested config files... ChatGPT is the author, beware!"""
    #print(filepath)
    #print(file)
    if loaded_files is None:
        loaded_files = set()
    if file in loaded_files:
        return {}  # Avoid circular imports
    loaded_files.add(file)

    with open(f"{filepath}/{file}", 'r') as f:
        data = yaml.safe_load(f) or {}

    # Handle includes
    includes = data.pop('include', [])
    merged_data = {}
    for include_file in includes:
        # full_include_path = f"{filepath}/{include_file}"
        merged_data.update(loadConfig(include_file, filepath, loaded_files))

    #print(merged_data)
    
    merged_data.update(data)  # Later settings override earlier ones
    return merged_data


def applyOtherConfig(tao, configArr):
    """
    The format for other_configs is an array with rows
    [ elementName, attributeName, setValue ] 
    """
    
    #Prevent recalculation until changes are made
    tao.cmd("set global lattice_calc_on = F")

    try: 
        for row in configArr:
            tao.cmd(f"""set ele {row[0]} {row[1]} = {row[2]}""")

    except:
        print("WARNING! At least one assignment has failed!")

    #Prevent recalculation until changes are made
    tao.cmd("set global lattice_calc_on = T")


def disableAutoQuadEnergyCompensation(tao):
    """
    The golden lattice, by default, has the quads set according to the design K1 rather than a fixed gradient.
    For "typical" simulations, this is a good approach. It's basically assuming that we LEM the machine
    For some edge cases, like jitter simulations, we don't want the magnets to be changing though
    """ 
    
    tao.cmd("set ele QUAD::* FIELD_MASTER = T")

    return


def disableAutoMagnetEnergyCompensation(tao):
    """
    The golden lattice, by default, has the magnets set according to the design K# rather than a fixed field.
    For "typical" simulations, this is a good approach. It's basically assuming that we LEM the machine
    For some edge cases, like jitter simulations, we don't want the magnets to be changing though
    """ 
    
    tao.cmd("set ele Solenoid::* FIELD_MASTER = T")
    
    tao.cmd("set ele Sbend::* FIELD_MASTER = T")
    tao.cmd("set ele Quadrupole::* FIELD_MASTER = T")
    tao.cmd("set ele Sextupole::* FIELD_MASTER = T")
    tao.cmd("set ele Multipole::* FIELD_MASTER = T")

    tao.cmd("set ele Wiggler::* FIELD_MASTER = T")

    tao.cmd("set ele VKicker::* FIELD_MASTER = T")
    tao.cmd("set ele HKicker::* FIELD_MASTER = T")

    return
