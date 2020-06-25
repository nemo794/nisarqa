from quality.GCOVFile import GCOVFile
from quality.SLCFile import SLCFile
from quality.SLCImage import SLCImage
from quality import errors_base, errors_derived

import h5py
import numpy

import os, os.path
import unittest

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"

    def setUp(self):
        pass
        
    def test_partial_nan(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "nan1.h5"), mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.slc_file.create_images()

        self.assertRaisesRegex(errors_base.WarningError, "LSAR A_HH has 8256 NaN's=49.6%", \
                               self.slc_file.check_nans)
        
    def test_full_nan(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "nan2.h5"), mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.slc_file.create_images()

        self.assertRaisesRegex(errors_base.FatalError, "LSAR A_HH is entirely NaN", \
                               self.slc_file.check_nans)
        

    def test_partial_zero(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "zeros1.h5"), mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.slc_file.create_images()

        self.assertRaisesRegex(errors_base.WarningError, "LSAR A_HH has 4134 Zeros=24.8%", \
                               self.slc_file.check_nans)

    def test_full_zero(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "zeros2.h5"), mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.slc_file.create_images()

        self.assertRaisesRegex(errors_base.FatalError, "LSAR A_HH is entirely Zeros", \
                               self.slc_file.check_nans)

    def test_full_nan_or_zero(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "nan_zeros.h5"), mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.slc_file.create_images()

        self.assertRaisesRegex(errors_base.FatalError, "LSAR A_HH is entirely NaNs or Zeros", \
                               self.slc_file.check_nans)

        
    def test_nan_partial_gcov(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_nan1.h5"), mode="r")

        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()
        self.gcov_file.create_images()

        self.assertRaisesRegex(errors_base.WarningError, "LSAR A_HHHH has 40734 NaN's=50.0%", \
                               self.gcov_file.check_nans)
        
    def test_nan_all_gcov(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_nan2.h5"), mode="r")

        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()
        self.gcov_file.create_images()

        self.assertRaisesRegex(errors_base.FatalError, "LSAR A_HHHH is entirely NaN", \
                               self.gcov_file.check_nans)
        
if __name__ == "__main__":
    unittest.main()

        
        

