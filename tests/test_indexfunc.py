import scatteringmatrix as sm
import numpy as np
import scipy as sp
import unittest
import numpy.testing as npt
import os

class TestSingleIndexFunction(unittest.TestCase):

    def test_single_index_real(self):
        index_fcn = sm.single_index_function(1.5)
        self.assertAlmostEqual(index_fcn(sm.Units.um), 1.5)

class TestMixedMediumFunction(unittest.TestCase):
    def test_bruggemann_mixed_medium(self):
        index1_fcn = sm.single_index_function(1.5)
        index2_fcn = sm.single_index_function(1.0)
        brug_fcn = sm.generate_mixed_medium_approximation(0.5, index1_fcn,
                                                          index2_fcn,
                                                          mix_type = "bruggeman")
        self.assertAlmostEqual(brug_fcn(sm.Units.um), 1.233182491)

    def test_MG_mixed_medium(self):
        index1_fcn = sm.single_index_function(1.5)
        index2_fcn = sm.single_index_function(1.0)
        brug_fcn = sm.generate_mixed_medium_approximation(0.5, index1_fcn,
                                                          index2_fcn,
                                                          mix_type = "MG")
        self.assertAlmostEqual(brug_fcn(sm.Units.um), 1.249489691)

if __name__ == '__main__':
    unittest.main()
