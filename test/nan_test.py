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
        
    def test_partial(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "nan1.h5"), "r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.slc_file.create_images()

        self.assertRaisesRegex(errors_base.WarningError, "1 images had NaN's: (LSAR A HH, 3148200=50%)*", \
                               self.slc_file.check_nans)
        
    def test_full(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "nan2.h5"), "r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.slc_file.create_images()

        self.assertRaisesRegex(errors_base.FatalError, "2 images were empty: (LSAR A HH)*", \
                               self.slc_file.check_nans)
        


        
if __name__ == "__main__":
    unittest.main()

        
        

