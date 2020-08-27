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
    XML_DIR = "xml"

    def setUp(self):
        self.rslc_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L1_SLC.xml"))
        self.lcapture = LogCapture()
        self.logger = logging_base.NISARLogger("junk.log")

    def tearDown(self):
        self.logger.close()
        self.lcapture.uninstall()

    def test_start_end_time(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"1.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()

        message = "Start Time b'2018-07-30T16:15:47.000000' not less than End Time b'2018-07-30T16:14:39.000000'"
        self.lcapture.check_present(("NISAR", "WARNING", message), order_matters=False)
                                    
    def test_orbit(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"2.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()

        self.lcapture.check_present(("NISAR", "WARNING", "Invalid Orbit Number: 0"), \
                                    order_matters=False)
         
    def test_track(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"3.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()

        self.lcapture.check_present(("NISAR", "WARNING", "Invalid Track Number: 255"), \
                                    order_matters=False)
        
    def test_frame(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"4.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()

        self.lcapture.check_present(("NISAR", "WARNING", "Invalid Frame Number: 0"), \
                                    order_matters=False)
        
    def test_cycle(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"5.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()

        self.lcapture.check_present(("NISAR", "WARNING", "Invalid Cycle Number: -4"), \
                                    order_matters=False)
        
    def test_product_type(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"6.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()

        self.lcapture.check_present(("NISAR", "WARNING", "Invalid Product Type: ABCD"), \
                                    order_matters=False)        
        
    def test_look_direction(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"7.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()

        self.lcapture.check_present(("NISAR", "WARNING", "Invalid Look Direction: leftr"), \
                                    order_matters=False)          
         
    def test_acquired_frequencies(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"8.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()
        self.slc_file.check_frequencies("LSAR", self.slc_file.FREQUENCIES["LSAR"])

        message = "acquiredCenterFrequency A=1243000000.000000 not less than B=1233000000.000000"
        self.lcapture.check_present(("NISAR", "WARNING", message), order_matters=False)
        
    def test_processed_frequencies(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"9.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()
        self.slc_file.check_frequencies("LSAR", self.slc_file.FREQUENCIES["LSAR"])

        message = "processedCenterFrequency A=1243000000.000000 not less than B=1233000000.000000"
        self.lcapture.check_present(("NISAR", "WARNING", message), order_matters=False)        
        
    def test_orbit_track(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_identification"2b.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.check_identification()

        self.lcapture.check_present(("NISAR", "WARNING", "Invalid Orbit Number: 0"), \
                                    ("NISAR", "WARNING", "Invalid Track Number: 0"), \
                                     order_matters=False)
                                    
        
if __name__ == "__main__":
    unittest.main()

        
        

