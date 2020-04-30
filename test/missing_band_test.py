from quality.SLCFile import SLCFile
from quality import errors_derived

import h5py
import numpy

import os, os.path
import unittest

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"

    def setUp(self):
        pass

    def test_dummy(self):

        self.assertTrue(True)
    
    def test_missing_band(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "missing_band.h5"), "r")
        self.assertRaises(errors_derived.IdentificationFatal)


if __name__ == "__main__":
    unittest.main()

        
        

