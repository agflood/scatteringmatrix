"""
This file implements finding the transmission and reflection for various
wavelengths for a single index lyaer on a glass substrate.
"""

import scatteringmatrix as sm
import numpy as np

'''
Declare Structure
'''

#Defining index functions. Any function that returns a complex value given
#a specific wavelength (in meters) can be used. Here, we declare functions that
#always return the same index, regardless of wavelength
index_air = sm.single_index_function(1.0)
index_glass = sm.single_index_function(1.5)
index_Si3N4 = sm.single_index_function(2.0)

#A 1D optical structure is a list of layers. We declare two of the layers here
film = sm.SingleLayer(index_Si3N4, 60.0*sm.Units.NANOMETERS)
substrate = sm.SingleLayer(index_glass, 1.1*sm.Units.MILLIMETERS)

#Here we declare the actual list and append the layers.
structure = sm.LayerList()
structure.append(sm.SingleLayer(index_air, sm.Units.MICROMETERS))
structure.append(film)
structure.append(substrate)
structure.append(sm.SingleLayer(index_air, sm.Units.MICROMETERS))

'''
Calculate transmission and reflection at 0 degrees
'''

wavelengths = np.linspace(300.0, 1200.0, 91)*sm.Units.NANOMETERS
#Data is returned in an array of arrays.
results = sm.scattering_matrix_calculation(structure, wavelengths)
print("Selected Wavelength Data")
print("----------------------------")
print("Incident Angle:", results[4])
print("Mode:", results[5])
print("Wavelengths (nm):", '{0:.1f}'.format(results[3][30]/sm.Units.NANOMETERS),
      ",", '{0:.1f}'.format(results[3][60]/sm.Units.NANOMETERS))
print("Transmission @ 600 nm, 900 nm:", '{0:.6f}'.format(results[0][30]),
      ",", '{0:.6f}'.format(results[0][60]))
print("Reflection @ 600 nm, 900 nm:", '{0:.6f}'.format(results[1][30]),
      ",", '{0:.6f}'.format(results[1][60]))
print("Absorption @ 600 nm, 900 nm:", '{0:.6f}'.format(results[2][30]),
      ",", '{0:.6f}'.format(results[2][60]))
