import scatteringmatrix as sm
import numpy as np
import scipy as sp
import unittest
import numpy.testing as npt
import os

class TestGetCSVColumns(unittest.TestCase):

    def load_tab_delimited(self, filename):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,filename)
        data = sm.get_csv_columns_as_2d_nparray(filename_full, np.array([0,2]),
                                                delimiter='\t', lines_to_skip = 2)
        return data

    def load_comma_delimited(self, filename):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,filename)
        data = sm.get_csv_columns_as_2d_nparray(filename_full, np.array([1,3]),
                                                delimiter=',', lines_to_skip = 1)
        return data

    def test_get_csv_columns_tabdelim_num_cols(self):
        data = self.load_tab_delimited("TestCSVFile_tab.csv")
        self.assertTrue(len(data)==2)

    def test_get_csv_columns_tabdelim_type_cols(self):
        data = self.load_tab_delimited("TestCSVFile_tab.csv")
        self.assertTrue(isinstance(data,np.ndarray))

    def test_get_csv_columns_tabdelim_type_element(self):
        data = self.load_tab_delimited("TestCSVFile_tab.csv")
        self.assertTrue(all((isinstance(x, np.float64) for x in col) for col in data))

    def test_get_csv_columns_tabdelim_values(self):
        data = self.load_tab_delimited("TestCSVFile_tab.csv")
        expected_data = np.array([[200.0,300.0,400.0,500.0,600.0,700.0,800.0,900.0,1000.0,1100.0,1200.0,1300.0,1400.0],
                                  [2.0,3.0,4.0,3.0,2.0,1.0,2.0,3.0,4.0,3.0,2.0,1.0,2.0]])
        self.assertTrue((data==expected_data).all())

    def test_get_csv_columns_commadelim_num_cols(self):
        data = self.load_comma_delimited("TestCSVFile_comma.csv")
        self.assertTrue(len(data)==2)

    def test_get_csv_columns_commadelim_type_cols(self):
        data = self.load_comma_delimited("TestCSVFile_comma.csv")
        self.assertTrue(isinstance(data,np.ndarray))

    def test_get_csv_columns_commadelim_type_element(self):
        data = self.load_comma_delimited("TestCSVFile_comma.csv")
        self.assertTrue(all((isinstance(x, np.float64) for x in col) for col in data))

    def test_get_csv_columns_commadelim_values(self):
        data = self.load_comma_delimited("TestCSVFile_comma.csv")
        expected_data = np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13],
                                  [0.0,0.1,0.2,0.3,0.4,0.3,0.2,0.1,0.0,0.1,0.2,0.3,0.4]])
        self.assertTrue((data==expected_data).all())

class TestCreateNKFromCSV(unittest.TestCase):

    def test_create_nk_from_csv_expected_nk(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma.csv")
        index_function = sm.create_nk_from_csv(filename_full,
                                               dependent_col=np.array([2,3]),
                                               independent_unit_str="um",
                                               lines_to_skip = 1)
        ratio = index_function(0.8*sm.Units.um)/(2.0+0.2j)
        self.assertAlmostEqual(ratio,1.0,places=5)

    def test_create_nk_from_csv_expected_nk_interpolate(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma.csv")
        index_function = sm.create_nk_from_csv(filename_full,
                                               dependent_col=np.array([2,3]),
                                               independent_unit_str="um",
                                               lines_to_skip = 1)
        ratio = index_function(0.95*sm.Units.um)/(3.5+0.05j)
        self.assertAlmostEqual(ratio,1.0,places=5)

    def test_create_nk_from_csv_expected_nk_upperbound(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma.csv")
        index_function = sm.create_nk_from_csv(filename_full,
                                               dependent_col=np.array([2,3]),
                                               independent_unit_str="um",
                                               lines_to_skip = 1)
        ratio = index_function(1.5*sm.Units.um)/(2.0+0.4j)
        self.assertAlmostEqual(ratio,1.0,places=5)

    def test_create_nk_from_csv_expected_nk_lowerbound(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma.csv")
        index_function = sm.create_nk_from_csv(filename_full,
                                               dependent_col=np.array([2,3]),
                                               independent_unit_str="um",
                                               lines_to_skip = 1)
        ratio = index_function(0.1*sm.Units.um)/(2.0+0.0j)
        self.assertAlmostEqual(ratio,1.0,places=5)

    def test_create_nk_from_csv_expected_nk_nm_er_ei(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_tab.csv")
        index_function = sm.create_nk_from_csv(filename_full,
                                               delimiter="\t",
                                               dependent_col=np.array([2,3]),
                                               dependent_unit_str="er_ei",
                                               lines_to_skip = 2)
        ratio = index_function(1.3*sm.Units.um)/(1.010947736+0.1483756228j)
        self.assertAlmostEqual(ratio,1.0,places=5)

    def test_create_nk_from_csv_expected_nk_eV(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma_eV.csv")
        index_function = sm.create_nk_from_csv(filename_full,
                                               independent_unit_str="eV",
                                               lines_to_skip = 1)
        ratio = index_function(1.23984*sm.Units.um)/(2.0+0.4j)
        self.assertAlmostEqual(ratio,1.0,places=5)

    def test_create_nk_from_csv_expected_nk_eV_lowerbound(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma_eV.csv")
        index_function = sm.create_nk_from_csv(filename_full,
                                               independent_unit_str="eV",
                                               lines_to_skip = 1)
        ratio =index_function(sm.convert_photon_unit("eV","wl",0.05))/(2.0+0.2j)
        self.assertAlmostEqual(ratio,1.0,places=5)

    def test_create_nk_from_csv_expected_nk_eV_upperbound(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma_eV.csv")
        index_function = sm.create_nk_from_csv(filename_full,
                                               independent_unit_str="eV",
                                               lines_to_skip = 1)
        ratio =index_function(sm.convert_photon_unit("eV","wl",6.0))/(2.0+0.0j)
        self.assertAlmostEqual(ratio,1.0,places=5)

    def test_create_nk_from_csv_indep_unit_except(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma_eV.csv")
        self.assertRaises(ValueError, sm.create_nk_from_csv, filename_full,
                          independent_unit_str="hi", lines_to_skip = 1)

    def test_create_nk_from_csv_dep_unit_except(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma_eV.csv")
        self.assertRaises(ValueError, sm.create_nk_from_csv, filename_full,
                          dependent_unit_str="hi", lines_to_skip = 1)

    def test_create_nk_from_csv_dep_columns_except(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename_full = os.path.join(dir_path,"TestCSVFile_comma_eV.csv")
        self.assertRaises(ValueError, sm.create_nk_from_csv, filename_full,
                          dependent_col=np.array([1,2,3]), lines_to_skip = 1)


if __name__ == '__main__':
    unittest.main()
