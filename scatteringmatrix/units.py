import numpy as np
from scipy import constants

class Units:
    """Constants and units useful for optical calculations"""
    #Distance Units
    KILOMETERS = km = 10.0**3
    METERS = m = 1.0
    CENTIMETERS = cm = 10.0**-2
    MILLIMETERS = mm = 10.0**-3
    MICROMETERS = um = 10.0**-6
    NANOMETERS = nm = 10.0**-9
    PICOMETERS = pm = 10.0**-12
    FEMTOMETERS = fm = 10.0**-15

    #Frequency Units
    HERTZ = Hz = 1.0
    KILOHERTZ = kHz = 10.0**3
    MEGAHERTZ = MHz = 10.0**6
    GIGAHERTZ = GHz = 10.0**9
    TERAHERTZ = THz = 10.0**12
    PETAHERTZ = PHz = 10.0**15

    #Time Units
    SECONDS = s = 1.0
    MILLISECONDS = ms = 10.0**-3
    MICROSECONDS = us = 10.0**-6
    NANOSECONDS = ns = 10.0**-9
    PICOSECONDS = ps = 10.0**-12
    FEMTOSECONDS = fs = 10.0**-15
    
    #Power Units
    WATTS = W = 1.0
    
def convert_photon_unit(initial_unit, new_unit, initial_value):
    """convert from one unit to another"""
    convert = {}
    convert["eV"]={"wl": lambda x: (constants.h*constants.c)/(x*constants.e),
                   "freq": lambda x: (x*constants.e)/(constants.h)}
    convert["wl"]={"eV": lambda x: (constants.h*constants.c)/(x*constants.e),
                   "freq": lambda x: constants.c/x}
    convert["freq"]={"wl": lambda x: constants.c/x,
                     "eV": lambda x: (constants.h*x)/(constants.e)}

    return convert[initial_unit][new_unit](initial_value)
                 
def convert_index_unit(initial_unit, new_unit, initial_value): 
    """convert_nk[from][to](value)"""
    convert = {}
    convert["nk"]={"er_ei": lambda x: x**2.0}
    convert["er_ei"]={"nk": lambda x: np.sqrt(x)}
    return convert[initial_unit][new_unit](initial_value)