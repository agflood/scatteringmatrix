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

class TestSellmeierIndexFunctions(unittest.TestCase):
    def test_sm2_function_returns_noerror(self):
        index_fcn = sm.create_index_Sellmeier_type2(1.0,1.0,1.0,1.0,1.0)
        some_index = index_fcn(600.0*sm.Units.nm)
    def test_sm3_function_returns_noerror(self):
        index_fcn = sm.create_index_Sellmeier_type3(1.0,1.0,1.0,1.0,1.0,1.0)
        some_index = index_fcn(600.0*sm.Units.nm)

class TestCauchyIndexFunctions(unittest.TestCase):
    def test_cauchy_function_returns_noerror(self):
        index_fcn = sm.create_index_cauchy_law(1.0,1.0,1.0,1.0,1.0,1.0)
        some_index = index_fcn(600.0*sm.Units.nm)

class TestDrudeIndexFunction(unittest.TestCase):
    def test_drude_function_returns_noerror(self):
        index_fcn = sm.create_index_drude(1.0,1.0,1.0)
        some_index = index_fcn(600.0*sm.Units.nm)

class TestLorentzPeakIndexFunction(unittest.TestCase):
    def test_lp_function_returns_noerror(self):
        index_fcn = sm.create_index_lorentz_peak(1.0,1.0,1.0)
        some_index = index_fcn(600.0*sm.Units.nm)

class TestTauLorenzIndexFunction(unittest.TestCase):
    def test_tlf_function_returns_noerror(self):
        index_fcn = sm.create_index_tauc_lorenz(1.0,1.0,1.0,1.0,1.0)
        some_index = index_fcn(600.0*sm.Units.nm)

if __name__ == '__main__':
    unittest.main()
