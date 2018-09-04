import math
import csv
import numpy as np
import scipy as sp
from scipy import constants
from scipy import interpolate
from .units import *

def get_csv_columns_as_2d_nparray(filename, columns_to_retrieve, 
                                        delimiter=",", lines_to_skip = 0):
    all_column_data = list()
    data_as_np_arrays = list()
    for col in columns_to_retrieve:
        all_column_data.append(list())
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, skipinitialspace = True,delimiter=delimiter)             
        for i in range(lines_to_skip):
            next(reader)
        for row in reader:
            for i in range(len(columns_to_retrieve)):
                all_column_data[i].append(float(row[columns_to_retrieve[i]]))
    for col in all_column_data:
        data_as_np_arrays.append(sp.array(col))
    data_as_np_arrays=sp.array(data_as_np_arrays)
    return data_as_np_arrays

def create_spectrum_from_csv(filename, independent_col=0, dependent_col=1,
                             independent_unit=Units.m, dependent_unit=Units.W,
                             delimiter=",", lines_to_skip = 0):
    """Creates a funtion that returns the power at a wavelength based on a csv
    
    Arg:
        filename(string): The csv file to be used
        type(string): the type of file to be opened. This can be either:
                        - "nm_W": stored as nm,Watts with headings
        
    Returns:
        callable: returns the power at a given wavelength (per unit wavelength) 
                  given the data in the csv
    """
    
    cols_to_get = np.array([independent_col,dependent_col])
    data = get_csv_columns_as_2d_nparray(filename, cols_to_get, 
                                               delimiter, lines_to_skip)
    independent_data = data[0]*independent_unit
    dependent_data = data[1]*dependent_unit
    fills=(dependent_data[np.argmin(independent_data)],
           dependent_data[np.argmax(independent_data)])
    get_power = interpolate.interp1d(independent_data, dependent_data, 
                                     bounds_error=False, fill_value=fills)
    return lambda x: 1.0*get_power(x)


def create_nk_from_csv(filename, independent_col=0, dependent_col=np.array([1,2]),
                        independent_unit_str="nm", dependent_unit_str="nk",
                        delimiter=",", lines_to_skip = 0):

    if(len(dependent_col) != 2):
        raise ValueError("dependent_col - An iterable of length two expected")
    
    cols_to_get = np.array([independent_col,dependent_col[0],dependent_col[1]])
    data = get_csv_columns_as_2d_nparray(filename, cols_to_get, 
                                               delimiter, lines_to_skip)
    independent_data = data[0]
    dependent_data = data[1]+data[2]*1.0j
    if(independent_unit_str == "nm"):
        independent_data = independent_data*Units.nm
    elif(independent_unit_str == "um"):
        independent_data = independent_data*Units.um
    elif(independent_unit_str == "eV"):
        independent_data = convert_photon_unit("eV","wl", independent_data)
    elif(independent_unit_str == "m"):
        pass
    else:
        raise ValueError(independent_unit_str,"is not a valid selection")
    
    if(dependent_unit_str == "er_ei"):
        dependent_data = convert_index_unit("er_ei","nk", dependent_data)
    elif(dependent_unit_str == "nk"):
        pass
    else:
        raise ValueError(dependent_unit_str,"is not a valid selection")
        
    fills=(dependent_data[np.argmin(independent_data)],
           dependent_data[np.argmax(independent_data)])
    get_index = interpolate.interp1d(independent_data, dependent_data, 
                                     bounds_error=False, fill_value=fills)
    return lambda x: 1.0*get_index(x)

def generate_index_from_2csv(filename1, filename2, type_str="nm_nk", 
                             delimiter = ",", skip_first_line = True):
    """Creates a function that returns the index based on a csv file
    
    Arg:
        filename(string): The csv file to be used
        type(string): the type of file to be opened. This can be either:
                        - "eV_Epsilon": stored as eV,er,ei with headings
                        - "um_nk": stored as um,n,k, with headings
        
    Returns:
        callable: returns the complex index given the data in the csv
    """
    if(type_str=="nm_nk"):
        nm_list, n_list = list(), list()
        with open(filename1) as csv_file:
            reader = csv.reader(csv_file, delimiter = delimiter, 
                                skipinitialspace = True)             
            if(skip_first_line):
                next(reader)
            for row in reader:
                nm_list.append(float(row[0]))
                n_list.append(float(row[1]))
        nm_array = sp.array(nm_list)
        wavelengths = nm_array*Units.nm
        index = sp.array(n_list)
        get_index_n = interpolate.interp1d(wavelengths, index, bounds_error=False,
                                         fill_value=(index[0],index[-1]))
        nm_list, k_list = list(), list()
        with open(filename2) as csv_file:
            reader = csv.reader(csv_file, delimiter = delimiter, 
                                skipinitialspace = True)             
            if(skip_first_line):
                next(reader)
            for row in reader:
                nm_list.append(float(row[0]))
                k_list.append(float(row[1]))
        nm_array = sp.array(nm_list)
        wavelengths = nm_array*Units.nm
        index = - 1.0j*sp.array(k_list)
        get_index_k = interpolate.interp1d(wavelengths, index, bounds_error=False,
                                         fill_value=(index[0],index[-1]))
        
        return lambda x: 1.0*(get_index_n(x)+get_index_k(x))   

def generate_index_from_SOPRA_nk(filename):
    """creates a function that returns an index based on SOPRA nk file data
    
    Arg:
        filename(string): The nk file to be used
                
    Returns:
        callable: returns the complex index given the data in the nk file
    """    
    unit_list, n_list, k_list = list(), list(), list()
    unit_type = 0
    start_unit, end_unit, num_units, inc_units = 0.0,0.0,0.0,0.0
    with open(filename) as csv_file:
        current_row = 0             
        for line in csv_file:
            row = line.split()
            if(current_row == 0):
                unit_type = int(row[0])
                start_unit = float(row[1])
                end_unit = float(row[2])
                num_units = int(row[3])
                inc_units = (end_unit-start_unit)/(num_units-1)
            else:                        
                if(len(row) == 3):
                    unit_list.append(float(row[0]))
                    n_list.append(float(row[1]))
                    k_list.append(float(row[2]))
                else:
                    unit_list.append(start_unit+current_row*inc_units)
                    n_list.append(float(row[0]))
                    k_list.append(float(row[1]))
            current_row += 1
    unit_array = sp.array(unit_list)
    wavelengths = sp.zeros(len(unit_array))
    if(unit_type == 0):
        wavelengths = unit_array*Units.nm
    elif(unit_type == 1):
        wavelengths = convert_photon_unit("eV","wl",unit_array)
    elif(unit_type == 2):
        wavelengths = unit_array*Units.um
    index = sp.array(n_list) - 1.0j*sp.array(k_list)
    fill_indices = (index[0],index[-1])
    if(wavelengths[-1]<wavelengths[0]):
        fill_indices = (index[-1],index[0])        
    get_index = interpolate.interp1d(wavelengths, index, bounds_error=False,
                                     fill_value=fill_indices)
    return get_index 