from quality.GSLCFile import GSLCFile
from quality.GUNWFile import GUNWFile
from quality.SLCFile import SLCFile
from quality.SLCImage import SLCImage
from quality import errors_base, errors_derived

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

        
    def test_slc_wrong_size(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "slc_arraysize.h5"), mode="r")

        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()

        self.assertRaisesRegex(errors_base.FatalError, "Dataset LSAR A HH has.*(129, 129).*(129, 119).*", \
                               self.slc_file.check_slant_range)

    def test_gslc_wrong_size(self):

        self.gslc_file = GSLCFile(os.path.join(self.TEST_DIR, "gslc_arraysize.h5"), mode="r")
        
        self.gslc_file.get_bands()
        self.gslc_file.get_freq_pol()
        self.gslc_file.check_freq_pol("LSAR", [self.gslc_file.SWATHS], [self.gslc_file.FREQUENCIES], [""])
        self.gslc_file.create_images()

        self.assertRaisesRegex(errors_base.FatalError, "Dataset LSAR A HH has.*(500, 275).*(600, 275).*", \
                               self.gslc_file.check_slant_range)

    def test_gunw_inconsistent_size1(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_arraysize1.h5"), \
                                  xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.find_missing_datasets()
        self.assertRaisesRegex(errors_base.FatalError, "Found 1 array-size mismatches for images: .*Grid: LSAR A HH.*inconsistent.*", \
                               self.gunw_file.create_images)
          
    def test_gunw_inconsistent_size2(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_arraysize2.h5"), \
                                  xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.find_missing_datasets()
        self.assertRaisesRegex(errors_base.FatalError, "Found 2 array-size mismatches for images: .*Grid.*coordinates.*", \
                               self.gunw_file.create_images)

    def test_gunw_missing_coords1(self):
        
        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_nocoords1.h5"), \
                                  xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        #self.gunw_file.find_missing_datasets()
        self.assertRaisesRegex(errors_base.FatalError, "Found 2 array-size mismatches for images: .*Offset.*doesn't have.*coordinates.*")
          
    def test_gunw_missing_coords2(self):
        
        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_nocoords2.h5"), \
                                  xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.find_missing_datasets()
        self.assertRaisesRegex(errors_base.FatalError, "Found 2 array-size mismatches for images: .*Offset.*doesn't have.*coordinates.*")
          

        
if __name__ == "__main__":
    unittest.main()

        
        

