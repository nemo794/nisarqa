from quality.SLCFile import SLCFile
from quality import errors_base, errors_derived, logging_base

import h5py
import numpy
from testfixtures import LogCapture

import os, os.path
import unittest
import xml.etree.ElementTree as ET

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR="xml"

    def setUp(self):
        self.rslc_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L1_SLC.xml"))
        self.lcapture = LogCapture()
        self.logger = logging_base.NISARLogger("junk.log")

    def tearDown(self):
        self.logger.close()
        self.lcapture.uninstall()
        
    def test_different_orbit(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "lsar_vs_ssar.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_freq_pol("SSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()

        message = "Values of absoluteOrbitNumber differ between bands"
        self.lcapture.check_present(("NISAR", "WARNING", message), order_matters=False)

        
if __name__ == "__main__":
    unittest.main()

        
        

