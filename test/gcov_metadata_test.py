from quality.GCOVFile import GCOVFile
from quality import errors_base, errors_derived, logging_base

import h5py
import numpy
from testfixtures import LogCapture

import logging
import os, os.path
import unittest
import xml.etree.ElementTree as ET

class GCOVFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR = "xml"

    def setUp(self):
        self.xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L2_GCOV.xml"))
        self.logger = logging_base.NISARLogger("junk.log")
        self.lcapture = LogCapture(level=logging.DEBUG)

    def tearDown(self):
        self.logger.close()
        self.lcapture.uninstall()
 
    def xtest_missing_metadata(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_missing_metadata.h5"), \
                                  self.logger, xml_tree=self.xml_tree, mode="r")
        self.gcov_file.get_bands()

        self.lcapture.check_present(("NISAR", "ERROR", "/science/LSAR/GCOV/metadata does not exist"), \
                                    order_matters=False)
        #self.assertRaisesRegex(errors_base.FatalError, "/science/LSAR/GCOV/metadata does not exist", \
        #                       self.gcov_file.get_bands)

        
if __name__ == "__main__":
    unittest.main()

        
        

