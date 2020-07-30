from quality.SLCFile import SLCFile
from quality import errors_base, errors_derived

import h5py
import numpy

import os, os.path
import unittest
import xml.etree.ElementTree as ET

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR = "xml"

    def setUp(self):
        self.xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L1_SLC.xml"))


    #def tearDown(self):
    #    self.slc_file.close()

    def test_missing_subswath(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "missing_subswath.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.assertRaisesRegex(errors_base.WarningError, "LSAR FrequencyA had missing SubSwath1 bounds", \
                               self.slc_file.check_subswaths_bounds)
    
    def test_subswath_bounds(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "subswath_bounds.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.assertRaisesRegex(errors_base.WarningError, "LSAR FrequencyA with nSlantRange 129 had invalid SubSwath bounds", \
                               self.slc_file.check_subswaths_bounds)

        
        
if __name__ == "__main__":
    unittest.main()

        
        

