from quality.SLCFile import SLCFile
from quality import errors_base, errors_derived

import h5py
import numpy

import os, os.path
import unittest
import xml.etree.ElementTree as ET

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR="xml"

    def setUp(self):
        self.xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L1_SLC.xml"))

    def test_different_orbit(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "lsar_vs_ssar.h5"), xml_tree=self.xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_freq_pol("SSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])

        self.assertRaisesRegex(errors_base.WarningError, "Values of absoluteOrbitNumber differ between bands", \
                               self.slc_file.check_identification)

    
        
if __name__ == "__main__":
    unittest.main()

        
        

