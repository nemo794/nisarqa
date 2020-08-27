from quality.SLCFile import SLCFile
from quality import errors_base, errors_derived, logging_base

import h5py
import numpy
from testfixtures import LogCapture

import logging
import os, os.path
import unittest
import xml.etree.ElementTree as ET

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR = "xml"

    def setUp(self):
        self.rslc_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L1_SLC.xml"))
        self.logger = logging_base.NISARLogger("junk.log")
        self.lcapture = LogCapture(level=logging.WARNING)

    def tearDown(self):
        self.logger.close()
        self.lcapture.uninstall()        

    def test_missing_subswath(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_missing_subswath.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_subswaths_bounds()

        self.lcapture.check_present(("NISAR", "WARNING", "LSAR FrequencyA had missing SubSwath1 bounds"), \
                                    order_matters=False)
        
    def test_subswath_bounds(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_subswath_bounds.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_subswaths_bounds()

        self.lcapture.check_present(("NISAR", "WARNING", "LSAR FrequencyA with nSlantRange 129 " \
                                                         + "had invalid SubSwath bounds"), \
                                    order_matters=False)        
        
if __name__ == "__main__":
    unittest.main()

        
        

