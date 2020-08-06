from quality.GUNWFile import GUNWFile
from quality.SLCFile import SLCFile
from quality.SLCImage import SLCImage
from quality import errors_base, errors_derived, utility

import h5py
import numpy

import os, os.path
import unittest
import xml.etree.ElementTree as ET

class NISARFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR = "xml"

    def setUp(self):
        self.rslc_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L1_SLC.xml"))
        self.gunw_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L2_GUNW.xml"))
        
    def test_negative_spacing(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "time_spacing1.h5"), mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()

        time = self.slc_file["/science/LSAR/RSLC/swaths/zeroDopplerTime"][...]
        spacing = self.slc_file["/science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing"][...]
        start_time = self.slc_file["/science/LSAR/identification/zeroDopplerStartTime"][...]
        
        self.assertRaisesRegex(errors_base.FatalError, "LSAR zeroDopplerStartTime: Found 1 elements with negative spacing*", \
                               utility.check_spacing, self.slc_file.flname, start_time, time, spacing, \
                               "LSAR zeroDopplerStartTime", errors_derived.TimeSpacingWarning, \
                               errors_derived.TimeSpacingFatal)
        
    def test_unexpected_spacing(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "time_spacing2.h5"), mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()

        time = self.slc_file["/science/LSAR/RSLC/swaths/zeroDopplerTime"][...]
        spacing = self.slc_file["/science/LSAR/RSLC/swaths/zeroDopplerTimeSpacing"][...]
        start_time = self.slc_file["/science/LSAR/identification/zeroDopplerStartTime"][...]
        
        self.assertRaisesRegex(errors_base.WarningError, "LSAR zeroDopplerStartTime: Found 2 elements with unexpected steps*", \
                               utility.check_spacing, self.slc_file.flname, start_time, time, spacing, \
                               "LSAR zeroDopplerStartTime", errors_derived.TimeSpacingWarning, \
                               errors_derived.TimeSpacingFatal)
 
    def test_gunw_coordinate_warning(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_spacing_uneven.h5"), \
                                  xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.find_missing_datasets()
        self.gunw_file.create_images()
        self.assertRaisesRegex(errors_base.WarningError, "Grid.*LSAR A.*xCoordinates: Found 1 elements with unexpected steps*", \
                               self.gunw_file.get_coordinates)
        
    def test_gunw_coordinate_fatal(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_spacing_negative.h5"), \
                                  xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.find_missing_datasets()
        self.gunw_file.create_images()
        self.assertRaisesRegex(errors_base.FatalError, "Grid.*LSAR A.*yCoordinates: Found 1 elements with negative spacing*", \
                               self.gunw_file.get_coordinates)
        
if __name__ == "__main__":
    unittest.main()

        
        

