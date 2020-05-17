from quality.SLCFile import SLCFile
from quality import errors_base, errors_derived

import h5py
import numpy

import os, os.path
import unittest

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"

    def setUp(self):
        pass

    #def tearDown(self):
    #    self.slc_file.close()
    
    def test_different_orbit(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "lsar_vs_ssar.h5"), "r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "Values of absoluteOrbitNumber differ between bands", \
                               self.slc_file.check_identification)

    
        
if __name__ == "__main__":
    unittest.main()

        
        

