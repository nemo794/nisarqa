from quality.GUNWFile import GUNWFile
from quality.SLCFile import SLCFile
from quality.SLCImage import SLCImage
from quality import errors_base, errors_derived, logging_base, utility

import h5py
import numpy
from testfixtures import LogCapture

import logging
import os, os.path
import unittest
import xml.etree.ElementTree as ET

class NISARFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR = "xml"

    def setUp(self):
        self.rslc_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L1_SLC.xml"))
        self.gunw_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L2_GUNW.xml"))
        self.lcapture = LogCapture(level=logging.WARNING)
        self.logger = logging_base.NISARLogger("junk.log")

    def tearDown(self):

        self.logger.close()
        self.lcapture.uninstall()
        
    def test_negative_spacing(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_time_spacing1.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()

        time = self.slc_file["/science/LSAR/RSLC/swaths/zeroDopplerTime"][...]
        spacing = self.slc_file["/science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing"][...]
        start_time = self.slc_file["/science/LSAR/identification/zeroDopplerStartTime"][...]

        error_string = utility.check_spacing(self.slc_file.flname, start_time, time, spacing, \
                                             "LSAR zeroDopplerStartTime")

        for e in error_string:
            self.logger.log_message(logging_base.LogFilterWarning, e)
        
        message1 = "LSAR zeroDopplerStartTime: Found 1 elements with negative spacing: [12003.461104] " \
                 + "at locations (array([0]),)"
        message2 = "LSAR zeroDopplerStartTime: Found 2 elements with unexpected steps: [0.00090906 0.00090906] " \
                 + "at locations (array([0, 1]),)"
        self.lcapture.check_present(("NISAR", "WARNING", message1), \
                                    ("NISAR", "WARNING", message2), order_matters=False)
        
    def test_unexpected_spacing(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_time_spacing2.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()

        time = self.slc_file["/science/LSAR/RSLC/swaths/zeroDopplerTime"][...]
        spacing = self.slc_file["/science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing"][...]
        start_time = self.slc_file["/science/LSAR/identification/zeroDopplerStartTime"][...]
        error_string = utility.check_spacing(self.slc_file.flname, start_time, time, spacing, \
                                             "LSAR zeroDopplerStartTime")

        for e in error_string:
            self.logger.log_message(logging_base.LogFilterWarning, e)

        message1 = "LSAR zeroDopplerStartTime: Found 2 elements with unexpected steps: [0.00030302 0.00030302] " \
                 + "at locations (array([0, 1]),)"
        self.lcapture.check_present(("NISAR", "WARNING", message1), order_matters=False)
        
    def test_gunw_coordinate_warning(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_spacing_uneven.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.find_missing_datasets()
        self.gunw_file.create_images()
        self.gunw_file.get_coordinates()

        message = "Grid: (LSAR A HH) xCoordinates: Found 1 elements with unexpected steps: " \
                + "[1.] at locations (array([510]),)"
        self.lcapture.check_present(("NISAR", "WARNING", message), order_matters=False)        
        
if __name__ == "__main__":
    unittest.main()

        
        

