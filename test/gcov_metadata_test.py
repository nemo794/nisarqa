from quality.GCOVFile import GCOVFile
from quality import errors_base, errors_derived

import h5py
import numpy

import os, os.path
import unittest
import xml.etree.ElementTree as ET

class GCOVFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR = "xml"

    def setUp(self):
        self.xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L2_GCOV.xml"))


    #def tearDown(self):
    #    self.gcov_file.close()
    
    def test_missing_band(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_missing_metadata.h5"), mode="r")
        self.assertRaisesRegex(errors_base.FatalError, "/science/LSAR/GCOV/metadata does not exist", \
                               self.gcov_file.get_bands)

    def test_missing_polarization(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_missing_polarizations.h5"), mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] missing polarization [VVVV]", \
                               self.gcov_file.check_freq_pol)

        
    def xtest_incorrect_frequency(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "wrong_frequencies1.h5"), mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Band has invalid frequency list", \
                               self.gcov_file.check_freq_pol)

    def xtest_missing_frequency(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "wrong_frequencies2.h5"), mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Band missing Frequency[A,B]", \
                               self.gcov_file.check_freq_pol)

    def xtest_extra_frequency(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "wrong_frequencies3.h5"), mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Band frequency list missing [A,B]", \
                               self.gcov_file.check_freq_pol)

    def xtest_incorrect_polarizations(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "wrong_polarizations1.h5"), mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] has invalid polarization list", \
                               self.gcov_file.check_freq_pol)
        
    def xtest_missing_polarizations(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "wrong_polarizations2.h5"), mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] missing polarization [HH,VV,HV,VH]", \
                               self.gcov_file.check_freq_pol)
        
    def xtest_extra_polarizations(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "wrong_polarizations3.h5"), mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] has extra polarization [HH,VV,HV,VH]", \
                               self.gcov_file.check_freq_pol)
        

    def xtest_missing_none(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "missing_none.h5"), xml_tree=self.xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.find_missing_datasets()
        self.gcov_file.close()

    def xtest_missing_one(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "missing_one.h5"), xml_tree=self.xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "LSAR Identification missing 1 fields: .*absoluteOrbitNumber", \
                               self.gcov_file.find_missing_datasets)
        

        
        
if __name__ == "__main__":
    unittest.main()

        
        

