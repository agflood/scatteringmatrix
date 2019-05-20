# -*- coding: utf-8 -*-
"""This contains useful units and conversions for scattering matrix calculations

Because all the scattering matrix code is written with m.g.s SI units, constants
have been included that can be used as multipliers to improve the readability of
the code being written that uses this library. This will also hopefully cut down
on unit errors.

Imported External Modules: numpy, scipy.constants

Classes:
    Units: contains the various distance, frequency, time, and power multipliers

Functions:
    convert_photon_unit(string,string,numeric): converts between one unit of photon
        measurement (such as eV) to another (such as Hz).
    convert_index_unit(string,string,numeric): converts between complex nk values
        and complex relative permittivity.

Notes:
    Throughout the documentation, you will find type "numeric" indicated. Most
    functions are designed to work with floats, ints, complex, and numpy arrays.
    All of these are grouped under "numeric".

"""
import numpy as np
from scipy import constants

class Units:
    """Constants and units useful for optical calculations

    Attributes:
        KILOMETERS, km
        METERS, m
        CENTIMETERS, cm
        MILLIMETERS, mm
        MICROMETERS,  um
        NANOMETERS, nm
        PICOMETERS, pm
        FEMTOMETERS, fm
        HERTZ, Hz
        KILOHERTZ, kHz
        MEGAHERTZ, MHz
        GIGAHERTZ, GHz
        TERAHERTZ, THz
        PETAHERTZ, PHz
        SECONDS, s
        MILLISECONDS, ms
        MICROSECONDS, us
        NANOSECONDS, ns
        PICOSECONDS, ps
        FEMTOSECONDS, fs
        WATTS, W
    """
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
    """converts between one unit of photon measurement to another

    Args:
        initial_unit (string): Units of the current value. Must be one
            of 'eV', 'wl', or 'freq'.
        new_unit (string): Units to convert to. Must be one of 'eV',
            'wl', or 'freq'.
        initial_value (numeric): Initial value in initial_unit.

    Returns:
        numeric: value in terms of new_unit

    """
    convert = {}
    convert["eV"]={"wl": lambda x: (constants.h*constants.c)/(x*constants.e),
                   "freq": lambda x: (x*constants.e)/(constants.h)}
    convert["wl"]={"eV": lambda x: (constants.h*constants.c)/(x*constants.e),
                   "freq": lambda x: constants.c/x}
    convert["freq"]={"wl": lambda x: constants.c/x,
                     "eV": lambda x: (constants.h*x)/(constants.e)}

    return convert[initial_unit][new_unit](initial_value)

def convert_index_unit(initial_unit, new_unit, initial_value):
    """converts between complex nk values and complex relative permittivity.

    Args:
        initial_unit (string): Units of the current value. Must be one
            of 'nk', or 'er_ei'.
        new_unit (string): Units to convert to. Must be one of 'nk', or 'er_ei'.
        initial_value (numeric): Initial value in initial_unit.

    Returns:
        numeric: value in terms of new_unit

    """
    convert = {}
    convert["nk"]={"er_ei": lambda x: x**2.0}
    convert["er_ei"]={"nk": lambda x: np.sqrt(x)}
    return convert[initial_unit][new_unit](initial_value)
