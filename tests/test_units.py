import scatteringmatrix as sm
import numpy as np
import scipy as sp
import unittest
import numpy.testing as npt
import os

class TestConvertUnits(unittest.TestCase):
    
    def test_convert_eV_wl_1eV_is_1dot24um(self):
        wavelength_calc = sm.convert_photon_unit("eV","wl",1.0)
        wavelength_predict = 1.23984*sm.Units.um
        ratio = wavelength_calc/wavelength_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
        
    def test_convert_eV_wl_nparray(self):
        wavelength_calc = sm.convert_photon_unit("eV","wl",np.array([1.0,1.5]))
        wavelength_predict = np.array([1.23984, 0.82656108])*sm.Units.um
        ratio = wavelength_calc/wavelength_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)
    
    def test_convert_eV_wl_is_reversable(self):
        eV_start = 1.5
        wavelength_first = sm.convert_photon_unit("eV","wl",eV_start)
        eV_next = sm.convert_photon_unit("wl","eV",wavelength_first)
        ratio = eV_next/eV_start
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_convert_wl_freq_1um_is_300THz(self):
        freq_calc = sm.convert_photon_unit("wl","freq",1.0*sm.Units.um)
        freq_predict = 299.793*sm.Units.THz
        ratio = freq_calc/freq_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_convert_wl_freq_nparray(self):
        freq_calc = sm.convert_photon_unit("wl","freq",np.array([1.0,0.5])*sm.Units.um)
        freq_predict = np.array([299.793,599.586])*sm.Units.THz
        ratio = freq_calc/freq_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)
    
    def test_convert_wl_freq_is_reversable(self):
        wavelength_start = 1.5*sm.Units.um
        freq_first = sm.convert_photon_unit("wl","freq",wavelength_start)
        wavelength_next = sm.convert_photon_unit("freq","wl",freq_first)
        ratio = wavelength_next/wavelength_start
        self.assertAlmostEqual(ratio,1.0,places=5)
        
    def test_convert_freq_eV_100THz_is_0dot413(self):
        eV_calc = sm.convert_photon_unit("freq","eV",100.0*sm.Units.THz)
        eV_predict = 0.4135659
        ratio = eV_calc/eV_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_convert_freq_eV_nparray(self):
        eV_calc = sm.convert_photon_unit("freq","eV",np.array([100.0,2000.0])*sm.Units.THz)
        eV_predict = np.array([0.4135659,8.271318])
        ratio = eV_calc/eV_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)
    
    def test_convert_freq_eV_is_reversable(self):
        freq_start = 500.0*sm.Units.THz
        eV_first = sm.convert_photon_unit("freq","eV",freq_start)
        freq_next = sm.convert_photon_unit("eV","freq",eV_first)
        ratio = freq_next/freq_start
        self.assertAlmostEqual(ratio,1.0,places=5)

class TestConvertNk(unittest.TestCase):
        
    def test_convert_nk_erei_1dot5_is_2dot25(self):
        erei_calc = sm.convert_index_unit("nk","er_ei",1.5+1.0j)
        erei_predict = 1.25+3.0j
        ratio = erei_calc/erei_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_convert_nk_erei_is_reversable(self):
        nk_start = 1.5+1.0j
        erei_first = sm.convert_index_unit("nk","er_ei",nk_start)
        nk_next = sm.convert_index_unit("er_ei","nk",erei_first)
        ratio = nk_next/nk_start
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_convert_nk_nparray(self):
        nk_init = np.array([1.5+1.0j, 2.0])
        erei_calc = sm.convert_index_unit("nk","er_ei",nk_init)
        erei_predict = np.array([1.25+3.0j, 4.0])
        ratio = erei_calc/erei_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)

        
if __name__ == '__main__':    
    unittest.main()