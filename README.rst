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

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
