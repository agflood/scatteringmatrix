# Optical Scattering Matrix Library
This implements a simple 1D scattering matrix algorithm for the calculation of reflection, transmission, and absorption of optical films and filters. It is an alternative to the transfer matrix algorithm and is known for being less prone to issues with highly absorbing films.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

This library makes heavy use of the numpy and scipy libraries. You should have:

* python version >3.x
* numpy version >x.y
* scipy version >x.y

(testing is ongoing for version compatibility)

### Installing

The best way to install is via pip. This is a python3 library, so the recommend install is either

```
pip install scatteringmatrix
```
or
```
pip3 install scatteringmatrix
```

## Examples
Right now there is a single example file in the examples directory.  It is exactly the same as the code below.

```python
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

```

## Running unit tests

Go into the directory where you have the library installed to run the basic unit tests. This can be done via the command line with

```
python3 -m unittest
```
which should result in an output similar to:
```
Ran 80 tests in 0.026s

OK
```

## Deployment

This is enitrely a python module, so no special deployment instructions are required.

## Contributing

Please contact the lead author andrew.flood@mail.utoronto.ca if you wish to join the project.

## Versions

We have only had development versions up until now, so it will still stay at 0.1 until the beta release.

## Authors

* **Andrew Flood** - *Initial work and current project lead*


## License

This project is licensed under the MIT License - see the LICENSE.txt file for details
