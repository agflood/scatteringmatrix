# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 20:11:00 2017

@author: AndrewFlood
"""

import scatteringmatrix as sm
import numpy as np
import scipy as sp
import unittest
import numpy.testing as npt
import os

class TestTransmissionAngle(unittest.TestCase):
    
    def test_transmission_angle_zero(self):
        self.assertAlmostEqual(sm.transmission_angle(1.0, 1.5, 0.0), 0.0)

    def test_transmission_angle_1to1dot5_at_angle(self):
        angle_calc = sm.transmission_angle(1.0, 1.5, np.pi/3.0)
        angle_predict = 0.61547970867
        ratio = angle_calc/angle_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_angle_total_internal_reflection(self):
        angle_calc = sm.transmission_angle(1.5, 1.0, np.pi/3.0)
        angle_predict = 1.570796327+0.7552738756j
        #Because both +/- the imaginary are valid solutions
        ratio = (angle_calc.real+abs(angle_calc.imag)*1.0j)/angle_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_angle_complex(self):
        angle_calc = sm.transmission_angle(1.2+1.0j, 1.5+1.5j, np.pi/4.0)
        angle_predict = 0.544230-0.0550733j
        ratio=angle_calc/angle_predict
        self.assertAlmostEqual(ratio,1.0,places=5)        
    
    def test_transmission_angle_reversable(self):
        angle_init = np.pi/5.0
        angle_fwd = sm.transmission_angle(1.0+1.5j, 2.0+2.5j, angle_init)
        angle_reverse = sm.transmission_angle(2.0+2.5j, 1.0+1.5j, angle_fwd)
        ratio = angle_reverse/angle_init
        self.assertAlmostEqual(ratio,1.0,places=5)        
        
    def test_transmission_angle_nparray(self):
        angle_init = np.array([np.pi/3.0, np.pi/4.0])
        index1 = np.array([1.0,1.0])
        index2 = np.array([1.5,1.75])
        angle_calc = sm.transmission_angle(index1,index2,angle_init)
        angle_predict = np.array([0.61547970867, 0.415952087])
        ratio = angle_calc/angle_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)  

class TestGetKz(unittest.TestCase):
    
    def test_get_kz_angle_0(self):
        kz_calc = sm.get_kz(1.5,0.0,1.0*sm.Units.um)
        kz_predict = 9.42477796*10**6
        ratio = kz_calc/kz_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_get_kz_with_angle(self):
        kz_calc = sm.get_kz(1.5,np.pi/3.0,1.0*sm.Units.um)
        kz_predict = 4.712388898*10**6
        ratio = kz_calc/kz_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_get_kz_complex(self):
        kz_calc = sm.get_kz(1.2+1.0j,0.5-0.1j,1.0*sm.Units.um)
        kz_predict = (6.34819404+5.903688016j)*10**6
        ratio = kz_calc/kz_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_get_kz_nparray(self):
        index_init = np.array([1.5,1.0])
        angle_init = np.array([0.0,np.pi/4.0])
        wavelengths = np.array([0.5,1.0])*sm.Units.um
        kz_calc = sm.get_kz(index_init,angle_init,wavelengths)
        kz_predict = np.array([18.849555,4.4428829])*10**6
        ratio = kz_calc/kz_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)
        
class TestReflectionTE(unittest.TestCase):
    
    def test_reflection_TE_0_angle(self):
        reflected_calc = sm.reflection_TE(1.0,1.5,0.0)
        reflected_predict = -0.2
        ratio = reflected_calc/reflected_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_reflection_TE_with_angle(self):
        reflected_calc = sm.reflection_TE(1.0,1.5,np.pi/3.0)
        reflected_predict = -0.420204103
        ratio = reflected_calc/reflected_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_reflection_TE_complex_index(self):
        reflected_calc = sm.reflection_TE(1.0,3.5+1.0j,np.pi/3.0)
        reflected_predict = -0.76028867-0.063273675j
        ratio = reflected_calc/reflected_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_reflection_TE_complex_angle(self):
        reflected_calc = sm.reflection_TE(1.5+1.5j, 1.2+1.0j, 
                                          0.544230-0.0550733j)
        reflected_predict = 0.2451657156+0.058335218j
        ratio = reflected_calc/reflected_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_reflection_TE_nparray(self):
        init_index1 = np.array([1.0,1.0])
        init_index2 = np.array([1.5,3.5+1.0j])
        init_angle = np.array([0.0,np.pi/3.0])
        reflected_calc = sm.reflection_TE(init_index1, init_index2, init_angle)
        reflected_predict = np.array([-0.2,-0.76028867-0.063273675j])
        ratio = reflected_calc/reflected_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)

class TestTransmissionTE(unittest.TestCase):
    
    def test_transmission_TE_0_angle(self):
        transmitted_calc = sm.transmission_TE(1.0,1.5,0.0)
        transmitted_predict = 0.8
        ratio = transmitted_calc/transmitted_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_TE_with_angle(self):
        transmitted_calc = sm.transmission_TE(1.0,1.5,np.pi/3.0)
        transmitted_predict = 0.5797958971
        ratio = transmitted_calc/transmitted_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_TE_complex_index(self):
        transmitted_calc = sm.transmission_TE(1.0,3.5+1.0j,np.pi/3.0)
        transmitted_predict = 0.239711424-0.063273707j
        ratio = transmitted_calc/transmitted_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_TE_complex_angle(self):
        transmitted_calc = sm.transmission_TE(1.5+1.5j, 1.2+1.0j, 
                                          0.544230-0.0550733j)
        transmitted_predict = 1.245165639+0.058335193866j
        ratio = transmitted_calc/transmitted_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_TE_nparray(self):
        init_index1 = np.array([1.0,1.0])
        init_index2 = np.array([1.5,3.5+1.0j])
        init_angle = np.array([0.0,np.pi/3.0])
        transmitted_calc = sm.transmission_TE(init_index1, init_index2, init_angle)
        transmitted_predict = np.array([0.8,0.239711424-0.063273707j])
        ratio = transmitted_calc/transmitted_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)

class TestPowerConservation_TE(unittest.TestCase):
    
    def is_power_conserved_TE(self, index1, index2, angle):
        reflected_power = sm.reflected_power_TE(index1, index2, angle)
        transmitted_power = sm.transmitted_power_TE(index1, index2, angle)
        total_power = reflected_power+transmitted_power
        if np.isclose(total_power, 1.0):
            return True
        else:            
            print(total_power)
            return False
        
    def test_power_conservation_TE_0_angle(self):
        self.assertTrue(self.is_power_conserved_TE(1.0, 1.5, 0.0))
    
    def test_power_conservation_TE_with_angle(self):
        self.assertTrue(self.is_power_conserved_TE(1.0,1.5,np.pi/3.0))        
    
    def test_power_conservation_TE_complex_index(self):
        self.assertTrue(self.is_power_conserved_TE(1.0,3.5+1.0j,np.pi/3.0))
            
    def test_power_conservation_TE_complex_angle(self):
        self.assertTrue(self.is_power_conserved_TE(1.5+1.5j, 1.2+1.0j, 
                                          0.544230-0.0550733j))

class TestReflectionTM(unittest.TestCase):
    
    def test_reflection_TM_0_angle(self):
        reflected_calc = sm.reflection_TM(1.0,1.5,0.0)
        reflected_predict = 0.2
        ratio = reflected_calc/reflected_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_reflection_TM_with_angle(self):
        reflected_calc = sm.reflection_TM(1.0,1.5,np.pi/3.0)
        reflected_predict = -0.0424492346
        ratio = reflected_calc/reflected_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_reflection_TM_complex_index(self):
        reflected_calc = sm.reflection_TM(1.0,3.5+1.0j,np.pi/3.0)
        reflected_predict = 0.306683918+0.119831459j
        ratio = reflected_calc/reflected_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_reflection_TM_complex_angle(self):
        reflected_calc = sm.reflection_TM(1.5+1.5j, 1.2+1.0j, 
                                          0.544230-0.0550733j)
        reflected_predict = -0.0567032644-0.028603612j
        ratio = reflected_calc/reflected_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_reflection_TM_nparray(self):
        init_index1 = np.array([1.0,1.0])
        init_index2 = np.array([1.5,3.5+1.0j])
        init_angle = np.array([0.0,np.pi/3.0])
        reflected_calc = sm.reflection_TM(init_index1, init_index2, init_angle)
        reflected_predict = np.array([0.2,0.306683918+0.119831459j])
        ratio = reflected_calc/reflected_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)

class TestTransmissionTM(unittest.TestCase):
    
    def test_transmission_TM_0_angle(self):
        transmitted_calc = sm.transmission_TM(1.0,1.5,0.0)
        transmitted_predict = 0.8
        ratio = transmitted_calc/transmitted_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_TM_with_angle(self):
        transmitted_calc = sm.transmission_TM(1.0,1.5,np.pi/3.0)
        transmitted_predict = 0.6383671769
        ratio = transmitted_calc/transmitted_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_TM_complex_index(self):
        transmitted_calc = sm.transmission_TM(1.0,3.5+1.0j,np.pi/3.0)
        transmitted_predict = 0.354205673-0.0669640612j
        ratio = transmitted_calc/transmitted_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_TM_complex_angle(self):
        transmitted_calc = sm.transmission_TM(1.5+1.5j, 1.2+1.0j, 
                                          0.544230-0.0550733j)
        transmitted_predict = 1.279287116+0.0772939415j
        ratio = transmitted_calc/transmitted_predict
        self.assertAlmostEqual(ratio,1.0,places=5)
    
    def test_transmission_TM_nparray(self):
        init_index1 = np.array([1.0,1.0])
        init_index2 = np.array([1.5,3.5+1.0j])
        init_angle = np.array([0.0,np.pi/3.0])
        transmitted_calc = sm.transmission_TM(init_index1, init_index2, init_angle)
        transmitted_predict = np.array([0.8,0.354205673-0.0669640612j])
        ratio = transmitted_calc/transmitted_predict
        npt.assert_array_almost_equal(ratio, np.array([1.0,1.0]),decimal=5)

class TestPowerConservation_TM(unittest.TestCase):
    
    def is_power_conserved_TM(self, index1, index2, angle):
        reflected_power = sm.reflected_power_TM(index1, index2, angle)
        transmitted_power = sm.transmitted_power_TM(index1, index2, angle)
        total_power = reflected_power+transmitted_power
        if np.isclose(total_power, 1.0):
            return True
        else:            
            print(total_power)
            return False
        
    def test_power_conservation_TM_0_angle(self):
        self.assertTrue(self.is_power_conserved_TM(1.0, 1.5, 0.0))
    
    def test_power_conservation_TM_with_angle(self):
        self.assertTrue(self.is_power_conserved_TM(1.0,1.5,np.pi/3.0))        
    
    def test_power_conservation_TM_complex_index(self):
        self.assertTrue(self.is_power_conserved_TM(1.0,3.5+1.0j,np.pi/3.0))
            
    def test_power_conservation_TM_complex_angle(self):
        self.assertTrue(self.is_power_conserved_TM(1.5+1.5j, 1.2+1.0j, 
                                          0.544230-0.0550733j))

class TestScatteringMatrixCalc(unittest.TestCase):
    
    def test_scattering_matrix_calculation_singleinterface(self):
        index_air = sm.single_index_function(1.0)
        index_glass = sm.single_index_function(1.5)
        structure = sm.LayerList()
        structure.append(sm.SingleLayer(index_air, 1.0*sm.Units.um))
        structure.append(sm.SingleLayer(index_glass, 1.0*sm.Units.um))
        calculate_powers = sm.scattering_matrix_calculation(structure, 1.0*sm.Units.um)
        calc_transmission = calculate_powers[0]
        calc_reflection = calculate_powers[1]
        calc_absorption = calculate_powers[2]
        expected_transmission = sm.transmitted_power_TE(1.0, 1.5, 0.0)
        expected_reflection = sm.reflected_power_TE(1.0, 1.5, 0.0)
        ratio_t = calc_transmission/expected_transmission
        ratio_r = calc_reflection/expected_reflection        
        npt.assert_array_almost_equal(ratio_t,np.array([1.0]),decimal=5)
        npt.assert_array_almost_equal(ratio_r,np.array([1.0]),decimal=5)
        npt.assert_array_almost_equal(calc_absorption,np.array([0.0]),decimal=5)
    
    def test_scattering_matrix_calculation_singleinterface_complex(self):
        index_air = sm.single_index_function(1.0)
        index_complex = sm.single_index_function(1.5-2.0j)
        structure = sm.LayerList()
        structure.append(sm.SingleLayer(index_air, 1.0*sm.Units.um))
        structure.append(sm.SingleLayer(index_complex, 0.0*sm.Units.um))
        calculate_powers = sm.scattering_matrix_calculation(structure, 1.0*sm.Units.um)
        calc_transmission = calculate_powers[0]
        calc_reflection = calculate_powers[1]
        calc_absorption = calculate_powers[2]
        expected_transmission = sm.transmitted_power_TE(1.0, 1.5-2.0j, 0.0)
        expected_reflection = sm.reflected_power_TE(1.0, 1.5-2.0j, 0.0)
        ratio_t = calc_transmission/expected_transmission
        ratio_r = calc_reflection/expected_reflection        
        npt.assert_array_almost_equal(ratio_t,np.array([1.0]),decimal=5)
        npt.assert_array_almost_equal(ratio_r,np.array([1.0]),decimal=5)
        npt.assert_array_almost_equal(calc_absorption,np.array([0.0]),decimal=5)
        
    def test_scattering_matrix_calculation_singlethickness_complex(self):
        index_air = sm.single_index_function(1.0)
        index_complex = sm.single_index_function(1.5-2.0j)
        structure = sm.LayerList()
        structure.append(sm.SingleLayer(index_air, 1.0*sm.Units.um))
        structure.append(sm.SingleLayer(index_complex, 1.0*sm.Units.um))
        calculate_powers = sm.scattering_matrix_calculation(structure, 1.0*sm.Units.um)
        calc_transmission = calculate_powers[0]
        calc_reflection = calculate_powers[1]
        expected_transmission = (sm.transmitted_power_TE(1.0, 1.5-2.0j, 0.0)
                                 *np.exp((-4.0*np.pi)*2))
        expected_reflection = sm.reflected_power_TE(1.0, 1.5-2.0j, 0.0)
        ratio_t = calc_transmission/expected_transmission
        ratio_r = calc_reflection/expected_reflection        
        sum_t_r = calc_transmission+calc_reflection
        npt.assert_array_almost_equal(ratio_t,np.array([1.0]),decimal=5)
        npt.assert_array_almost_equal(ratio_r,np.array([1.0]),decimal=5)
        self.assertTrue((sum_t_r<1.0).all())
    
    def test_scattering_matrix_calculation_singlethickness_complex_TE(self):
        i1 = 1.0
        i2 = 1.5-2.0j
        angle = np.pi/3
        
        #Calculated
        index_air = sm.single_index_function(i1)
        index_complex = sm.single_index_function(i2)
        structure = sm.LayerList()
        structure.append(sm.SingleLayer(index_air, 1.0*sm.Units.um))
        structure.append(sm.SingleLayer(index_complex, 1.0*sm.Units.um))
        calculate_powers = sm.scattering_matrix_calculation(structure, 
                                                            1.0*sm.Units.um,
                                                            incident_angle=angle)
        calc_transmission = calculate_powers[0]
        calc_reflection = calculate_powers[1]
        
        #Expected
        t_angle = sm.transmission_angle(i1, i2, angle)
        kz = sm.get_kz(i2, t_angle, 1.0*sm.Units.um)
        expected_transmission = (sm.transmitted_power_TE(1.0, 1.5-2.0j, angle)
                                 *np.exp((-2.0j*kz*1.0*sm.Units.um).real))
        expected_reflection = sm.reflected_power_TE(1.0, 1.5-2.0j, angle)
        
        
        #Check
        ratio_t = calc_transmission/expected_transmission
        ratio_r = calc_reflection/expected_reflection        
        sum_t_r = calc_transmission+calc_reflection
        npt.assert_array_almost_equal(ratio_t,np.array([1.0]),decimal=5)
        npt.assert_array_almost_equal(ratio_r,np.array([1.0]),decimal=5)
        self.assertTrue((sum_t_r<1.0).all())
    
    def test_scattering_matrix_calculation_singlethickness_complex_TM(self):
        i1 = 1.0
        i2 = 1.5-2.0j
        angle = np.pi/3
        
        #Calculated
        index_air = sm.single_index_function(i1)
        index_complex = sm.single_index_function(i2)
        structure = sm.LayerList()
        structure.append(sm.SingleLayer(index_air, 1.0*sm.Units.um))
        structure.append(sm.SingleLayer(index_complex, 1.0*sm.Units.um))
        calculate_powers = sm.scattering_matrix_calculation(structure, 
                                                            1.0*sm.Units.um,
                                                            incident_angle=angle,
                                                            mode="TM")
        calc_transmission = calculate_powers[0]
        calc_reflection = calculate_powers[1]
        
        #Expected
        t_angle = sm.transmission_angle(i1, i2, angle)
        kz = sm.get_kz(i2, t_angle, 1.0*sm.Units.um)
        expected_transmission = (sm.transmitted_power_TM(1.0, 1.5-2.0j, angle)
                                 *np.exp((-2.0j*kz*1.0*sm.Units.um).real))
        expected_reflection = sm.reflected_power_TM(1.0, 1.5-2.0j, angle)
        
        
        #Check
        ratio_t = calc_transmission/expected_transmission
        ratio_r = calc_reflection/expected_reflection        
        sum_t_r = calc_transmission+calc_reflection
        npt.assert_array_almost_equal(ratio_t,np.array([1.0]),decimal=5)
        npt.assert_array_almost_equal(ratio_r,np.array([1.0]),decimal=5)
        self.assertTrue((sum_t_r<1.0).all())
     
    def test_scattering_matrix_calculation_fabryperot(self):
        index_air = sm.single_index_function(1.0)
        index_glass = sm.single_index_function(1.5)
        wavelengths = np.array([1.5*sm.Units.um,0.75*sm.Units.um, 
                                6.0/5.0*sm.Units.um])
        structure = sm.LayerList()
        structure.append(sm.SingleLayer(index_air, 1.0*sm.Units.um))
        structure.append(sm.SingleLayer(index_glass, 1.0*sm.Units.um))
        structure.append(sm.SingleLayer(index_air, 1.0*sm.Units.um))
        calculate_powers = sm.scattering_matrix_calculation(structure, wavelengths)
        calc_transmission = calculate_powers[0]
        calc_reflection = calculate_powers[1]
        calc_absorption = calculate_powers[2]
        reflection_face = sm.reflected_power_TE(1.0,1.5,0.0)
        expected_transmission = np.array([1.0,1.0,
                                          1.0-(4*reflection_face/((1+reflection_face)**2.0))])
        #expected_reflection = 1.0-expected_transmission
        ratio_t = calc_transmission/expected_transmission
        #ratio_r = calc_reflection/expected_reflection        
        npt.assert_array_almost_equal(ratio_t,np.array([1.0,1.0,1.0]),decimal=5)
        #npt.assert_array_almost_equal(ratio_r,np.array([1.0]),decimal=5)
        npt.assert_array_almost_equal(calc_absorption,np.array([0.0,0.0,0.0]),decimal=5)

    def test_scattering_matrix_calculation_complex_slab(self):
        pass
    
        
if __name__ == '__main__':    
    unittest.main()