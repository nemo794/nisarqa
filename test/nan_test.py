from quality.GCOVFile import GCOVFile
from quality.SLCFile import SLCFile
from quality.SLCImage import SLCImage
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
        self.gslc_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L2_GSLC.xml"))
        self.gunw_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L2_GUNW.xml"))
        self.gcov_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L2_GCOV.xml"))
        self.logger = logging_base.NISARLogger("junk.log")
        self.lcapture = LogCapture(level=logging.WARNING)

    def tearDown(self):
        self.logger.close()
        self.lcapture.uninstall()
        
    def test_partial_nan(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_nan1.h5"), self.logger, \
                                xml_tree=self.rslc_xml_tree, mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        self.slc_file.check_nans()

        message = "LSAR A_HH has 8256 NaN's=49.6%:LSAR A_HH has 6 Zeros=0.0%"
        self.lcapture.check_present(("NISAR", "WARNING", message), \
                                    order_matters=False)

    def test_full_nan(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_nan2.h5"), self.logger, \
                                xml_tree=self.rslc_xml_tree, mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        self.slc_file.check_nans()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR A_HH is entirely NaN"), \
                                    order_matters=False)

    def test_partial_zero(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_zeros1.h5"), self.logger, \
                                xml_tree=self.rslc_xml_tree, mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        self.slc_file.check_nans()

        self.lcapture.check_present(("NISAR", "WARNING", "LSAR A_HH has 4134 Zeros=24.8%"),
                                    order_matters=False)

    def test_full_zero(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_zeros2.h5"), self.logger, \
                                xml_tree=self.rslc_xml_tree, mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        self.slc_file.check_nans()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR A_HH is entirely Zeros"), \
                                    order_matters=False)        

    def test_full_nan_or_zero(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_nan_zeros.h5"), self.logger, \
                                xml_tree=self.rslc_xml_tree, mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        self.slc_file.check_nans()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR A_HH is entirely NaNs or Zeros"), \
                                    order_matters=False)            

    def test_nan_partial_gcov(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_nan1.h5"), self.logger, \
                                  xml_tree=self.gcov_xml_tree, mode="r")

        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()
        self.gcov_file.create_images()
        self.gcov_file.check_nans()

        self.lcapture.check_present(("NISAR", "WARNING", "LSAR A_HHHH has 40734 NaN's=50.0%"), \
                                    order_matters=False)

    def test_nan_all_gcov(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_nan2.h5"), self.logger, \
                                  xml_tree=self.gcov_xml_tree, mode="r")

        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()
        self.gcov_file.create_images()
        self.gcov_file.check_nans()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR A_HHHH is entirely NaN"), \
                                    order_matters=False)
        
if __name__ == "__main__":
    unittest.main()

        
        

