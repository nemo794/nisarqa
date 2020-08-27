from quality.GSLCFile import GSLCFile
from quality.GUNWFile import GUNWFile
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

class NISARFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR = "xml"

    def setUp(self):
       self.rslc_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L1_SLC.xml"))
       self.gslc_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L2_GSLC.xml"))
       self.gunw_xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L2_GUNW.xml"))
       self.lcapture = LogCapture(level=logging.WARNING)
       self.logger = logging_base.NISARLogger("junk.log")

    def tearDown(self):

        self.logger.close()
        self.lcapture.uninstall()
        
    def test_slc_wrong_size(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_arraysize.h5"), self.logger, \
                                xml_tree=self.rslc_xml_tree, mode="r")

        self.slc_file.get_start_time()
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        self.slc_file.check_slant_range()

        self.lcapture.check_present(("NISAR", "ERROR", "Dataset LSAR A HH has shape (129, 129), expected (129, 119)"), \
                                    order_matters=False)

    def test_gslc_wrong_size(self):

        self.gslc_file = GSLCFile(os.path.join(self.TEST_DIR, "gslc_arraysize.h5"), self.logger, \
                                  xml_tree=self.gslc_xml_tree, mode="r")
        
        self.gslc_file.get_bands()
        self.gslc_file.get_freq_pol()
        self.gslc_file.check_freq_pol("LSAR", [self.gslc_file.SWATHS], [self.gslc_file.FREQUENCIES], [""])
        self.gslc_file.create_images()
        self.gslc_file.check_slant_range()

        self.lcapture.check_present(("NISAR", "ERROR", "Dataset LSAR A HH has shape (600, 275), expected (500, 275)"), \
                                    order_matters=False)

    def test_gunw_inconsistent_size1(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_arraysize1.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()

        print("Grid keys: %s" % self.gunw_file.GRIDS.keys())
        
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.find_missing_datasets()
        self.gunw_file.create_images()

        self.lcapture.check_present(("NISAR", "WARNING", "Grid: (LSAR A HH) inconsistent size in params " \
                                     + "['phaseSigmaCoherence', 'unwrappedPhase', 'connectedComponents']"), \
                                    order_matters=False)
        
    def test_gunw_inconsistent_size2(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_arraysize2.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.find_missing_datasets()
        self.gunw_file.create_images()

        errors = []
        for p in ("HH", "VV"):
            errors.append("Grid: (LSAR A %s) size doesn't match coordinates" % p)
        for p in ("HH", "VV"):
            errors.append("Offset: (LSAR A %s) size doesn't match coordinates" % p)

        for e in errors:
            self.lcapture.check_present(("NISAR", "WARNING", e), order_matters=False)

    def test_gunw_missing_coords1(self):
        
        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_nocoords1.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.create_images()

        for handle in ("Grid", "Offset"):
            for p in ("HH", "VV"):
                error = "%s: (LSAR A %s) doesn't have x/y coordinates" % (handle, p)
                self.lcapture.check_present(("NISAR", "ERROR", error), order_matters=False)
        
    def test_gunw_missing_coords2(self):
        
        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_nocoords2.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.create_images()

        for handle in ("Grid", "Offset"):
            for p in ("HH", "VV"):
                error = "%s: (LSAR A %s) size doesn't match coordinates" % (handle, p)
                self.lcapture.check_present(("NISAR", "WARNING", error), order_matters=False)
        
if __name__ == "__main__":
    unittest.main()

        
        

