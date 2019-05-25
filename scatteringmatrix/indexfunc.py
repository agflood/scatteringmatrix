"""Scattering Matrix Index Functions

This contains various useful functions for creating refracive indexes

While any function that returns a complex number can be used as an index, some
things are done so frequently that it helps to have basic functions already
defined.

Imported External Modules: numpy
Imported Library Modules: units

Functions:
    single_index_function(complex):
        Creates a function that returns an index for every single wavelength
    generate_mixed_medium_approximation(float,callable,callable,string):
        Creates a function that returns the mixed medium approximated index
    create_index_Sellmeier_type2(float,float,float,float,float):
            Creates a function that returns the sellmeier calculated index
    create_index_Sellmeier_type3(float,float,float,float,float,float):
        Creates a function that returns the sellmeier calculated index
    create_index_drude(float,float,float):
        Creates a function that returns the drude-model calculated index
    create_index_lorentz_peak(float,float,float):
        Creates a function that returns the lorentz peak calculated index
    create_index_cauchy_law(float,float,float,float,float,float):
        Creates a function that returns the cauchy calculated index
    create_index_tauc_lorenz(float, float, float, float, float):
        Creates a function that returns the tauc-lorenz calculated index

Notes:
    Throughout the documentation, you will find type "numeric" indicated. Most
    functions are designed to work with floats, ints, complex, and numpy arrays.
    All of these are grouped under "numeric".

"""

import numpy as np
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

def create_index_Sellmeier_type2(A,B,C,D,E):
    """Creates a function that returns the sellmeier calculated index

    The sellmeier index function is defined as:
    epsilon_r = 1.0+(A**2.0-1.0)*wavelength**2.0/(wavelength**2.0-B)
    epsilon_i = C/wavelength+D/wavelength**2.0+E/wavelength**5.0

    Arg:
        A(float): A parameter of the sellmeier function
        B(float): B parameter of the sellmeier function
        C(float): C parameter of the sellmeier function
        D(float): D parameter of the sellmeier function
        E(float): E parameter of the sellmeier function

    Returns:
        callable: returns the complex index given the wavelength

    Note: as per the rest of this library, meters are used for unit wavelength
    """
    def return_complex_index(wavelength):
        e_r = 1.0+(A**2.0-1.0)*wavelength**2.0/(wavelength**2.0-B)
        e_i = C/wavelength+D/wavelength**2.0+E/wavelength**5.0
        return convert_index_unit("er_ei", "nk", e_r-e_i*1.0j)
    return return_complex_index

def create_index_Sellmeier_type3(A,B,C,D,E,F):
    """Creates a function that returns the sellmeier calculated index

    The sellmeier index function is defined as:
    epsilon_r = A+B*wavelength**2.0/(wavelength**2.0-C)
    epsilon_i = D/wavelength+E/wavelength**2.0+F/wavelength**5.0

    Arg:
        A(float): A parameter of the sellmeier function
        B(float): B parameter of the sellmeier function
        C(float): C parameter of the sellmeier function
        D(float): D parameter of the sellmeier function
        E(float): E parameter of the sellmeier function
        F(float): F parameter of the sellmeier function

    Returns:
        callable: returns the complex index given the wavelength

    Note: as per the rest of this library, meters are used for unit wavelength
    """
    def return_complex_index(wavelength):
        e_r = A+B*wavelength**2.0/(wavelength**2.0-C)
        e_i = D/wavelength+E/wavelength**2.0+F/wavelength**5.0
        return convert_index_unit("er_ei", "nk", e_r-e_i*1.0j)
    return return_complex_index

def create_index_drude(P,E0,Tau):
    """Creates a function that returns the drude-model calculated index

    Arg:
        P(float): Real polarization
        E0(float): Plasma frequency (in eV)
        Tau(float): Mean free lifetime (1/Tau is the mean free pass)

    Returns:
        callable: returns the complex index given the wavelength

    Note: as per the rest of this library, meters are used for unit wavelength
    """
    def return_complex_index(wavelength):
        Lambda0 = convert_photon_unit("eV","wl", E0)
        tau_inv = (1.0/Tau)
        common = (wavelength/Lambda0)**2.0/(1.0+(wavelength*tau_inv)**2.0)
        e_r = P-common
        e_i = common*tau_inv*wavelength
        return convert_index_unit("er_ei", "nk", e_r-e_i*1.0j)
    return return_complex_index

def create_index_lorentz_peak(A,E0,gamma):
    """Creates a function that returns the lorentz peak calculated index

    Arg:
        A(float): Peak intensity
        E0(float): Center peak frequency (in eV)
        gamma(float): Peak width (in wavelength [meters])

    Returns:
        callable: returns the complex index given the wavelength

    Note: as per the rest of this library, meters are used for unit wavelength
    """
    def return_complex_index(wavelength):
        Lambda0 = convert_photon_unit("eV","wl", E0)
        common = wavelength**2.0-Lambda0**2
        common2 = A*wavelength**2.0/(common**2.0+gamma**2.0*wavelength**2.0)
        e_r = common*common2
        e_i = wavelength*gamma*common2
        return convert_index_unit("er_ei", "nk", e_r-e_i*1.0j)
    return return_complex_index

def create_index_cauchy_law(A,B,C,D,E,F):
    """Creates a function that returns the cauchy calculated index

    The cauchy index function is defined as:
    n_complex = A+B/x**2.0+C/x**4.0-1.0j*(D/x+E/x**3.0+F/x**5.0)

    Arg:
        A(float): A parameter of the cauchy function
        B(float): B parameter of the cauchy function
        C(float): C parameter of the cauchy function
        D(float): D parameter of the cauchy function
        E(float): E parameter of the cauchy function
        F(float): F parameter of the cauchy function

    Returns:
        callable: returns the complex index given the wavelength

    Note: as per the rest of this library, meters are used for unit wavelength
    """
    return lambda x: A+B/x**2.0+C/x**4.0-1.0j*(D/x+E/x**3.0+F/x**5.0)

def create_index_tauc_lorenz(er_inf, E_g, A, C, E_0):
    """Creates a function that returns the tauc-lorenz calculated index

    Arg:
        er_inf(float): Real permittivity at infinity
        E_g(float): Badgap of the material (in eV)
        A(float): Peak intensity
        C(float): Peak broadening term
        E_0(float): Center peak frequency (in eV)

    Returns:
        callable: returns the complex index given the wavelength

    Note: as per the rest of this library, meters are used for unit wavelength
    """
    def return_complex_index(wavelength):
        E = convert_photon_unit("wl","eV", wavelength)
        e_i = np.where(E >= E_g, (A*E_0*C*(E-E_g)**2)/(E*((E**2-E_0**2)**2+C**2*E**2)), 0.0)
        #the real permittivity has a VERY long formula
        a_ln = (E_g**2-E_0**2)*E**2+E_g**2*C**2-E_0**2*(E_0**2+3*E_g**2)
        a_atan = (E**2-E_0**2)*(E_0**2+E_g**2)+E_g**2*C**2
        alpha = np.sqrt(4*E_0**2-C**2)
        gamma = np.sqrt(E_0**2-(C**2/2))
        zeta_4 = (E**2-gamma**2)**2+alpha**2*C**2/4
        er_1 = A*C*a_ln/(2*np.pi*zeta_4*alpha*E_0)*np.log((E_0**2+E_g**2+alpha*E_g)/(E_0**2+E_g**2-alpha*E_g))
        er_2 = -A/np.pi*a_atan/(zeta_4*E_0)*(np.pi-np.arctan((2*E_g+alpha)/C)+np.arctan((alpha-2*E_g)/C))
        er_3 = 4*A*E_0*E_g*(E**2-gamma**2)/(np.pi*zeta_4*alpha)
        er_4 = np.arctan((2*E_g+alpha)/C)+np.arctan((alpha-2*E_g)/C)
        er_5 = -A*E_0*C*(E**2+E_g**2)/(np.pi*zeta_4*E)*np.log(np.absolute(E-E_g)/(E+E_g))
        er_6 = 2*A*E_0*C/(np.pi*zeta_4)*E_g*np.log((np.absolute(E-E_g)*(E+E_g))/np.sqrt((E_0**2-E_g**2)**2+E_g**2*C**2))
        e_r = er_inf+er_1+er_2+er_3*er_4+er_5+er_6
        return convert_index_unit("er_ei", "nk", e_r-e_i*1.0j)
    return return_complex_index
