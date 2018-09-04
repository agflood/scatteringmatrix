import math
import csv
import numpy as np
import scipy as sp
from scipy import constants
from scipy import interpolate
from .units import *

def single_index_function(index):
    """Creates a function that returns an index for every single wavelength
            
    Arg:
        index(complex): the index of refraction to be returned
        
    Returns:
        callable: returns the complex index for an argument of any float 
    """
    get_index = lambda wavelengths: index
    return get_index

def generate_mixed_medium_approximation(fraction_1, index_func_1, index_func_2,
                                        mix_type = "bruggeman"):
    """Creates a function that returns the mixed medium approximated index
    
    Arg:
        fraction_1(float): The volume fraction of index 1
        index_func_1(callable): the index returning function of medium 1
        index_func_2(callable): the index returning function of medium 2
        mix_type (string): the type of approximation formula used
                            - bruggeman: Bruggeman approximation used
                            - MG: Maxwell-Garnett apporximation used
        
    Returns:
        callable: returns the complex index given the mixture of those two
                  media
    """
    
    if(mix_type == "MG"):
        def porous_index(wavelength):
            epsilon_1 = index_func_1(wavelength)**2.0
            epsilon_2 = index_func_2(wavelength)**2.0
            numer = 3*epsilon_2-2*fraction_1*epsilon_2+2*fraction_1*epsilon_1
            denom = 3*epsilon_1+fraction_1*epsilon_2-fraction_1*epsilon_1
            new_epsilon = (numer/denom)*epsilon_1
            new_index = np.sqrt(new_epsilon)
            return new_index
        return porous_index    
    
    else:
        def porous_index(wavelength):
            a = -2.0
            b = ((3.0*fraction_1-1.0)*index_func_1(wavelength)
                +(2.0-3.0*fraction_1)*index_func_2(wavelength))
            c = index_func_1(wavelength)*index_func_2(wavelength)
            new_index = (-b-np.sqrt(b**2-4*a*c))/(2.0*a)
            return new_index
        return porous_index
