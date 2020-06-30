from quality.GCOVFile import GCOVFile
from quality.GSLCFile import GSLCFile
from quality.SLCFile import SLCFile
from quality import errors_base, errors_derived

import h5py
import numpy

import os, os.path
import unittest
import xml.etree.ElementTree as ET

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"
    XML_DIR = "xml"

    def setUp(self):
        self.xml_tree = ET.parse(os.path.join(self.XML_DIR, "nisar_L1_SLC.xml"))


    #def tearDown(self):
    #    self.slc_file.close()
    
    def test_missing_band(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "missing_band.h5"), mode="r")
        self.assertRaisesRegex(errors_base.FatalError, "/science/LSAR/.*metadata does not exist", \
                               self.slc_file.get_bands)

    def test_slc_inconsistent_bands(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "inconsistent_bands.h5"), mode="r")
        self.assertRaisesRegex(errors_base.FatalError, "Metadata and swath have inconsistent naming", \
                               self.slc_file.get_bands)
                                
    def test_gslc_incorrect_frequency(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "wrong_frequencies1.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Band has invalid frequency list", \
                               self.slc_file.check_freq_pol)

    def test_gslc_missing_frequency(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "wrong_frequencies2.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Band missing Frequency[A,B]", \
                               self.slc_file.check_freq_pol)

    def test_gslc_extra_frequency(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "wrong_frequencies3.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Band frequency list missing [A,B]", \
                               self.slc_file.check_freq_pol)

    def test_gcov_incorrect_frequency(self):

        self.slc_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_frequencies1.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Band has invalid frequency list", \
                               self.slc_file.check_freq_pol)

    def test_gcov_missing_frequency(self):

        self.slc_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_frequencies2.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Band missing Frequency[A,B]", \
                               self.slc_file.check_freq_pol)

    def test_gcov_extra_frequency(self):

        self.slc_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_frequencies3.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Band frequency list missing [A,B]", \
                               self.slc_file.check_freq_pol)

    def test_gslc_incorrect_polarizations(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "wrong_polarizations1.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] has invalid polarization list", \
                               self.slc_file.check_freq_pol)
        
    def test_gslc_missing_polarizations(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "wrong_polarizations2.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] missing polarization [HH,VV,HV,VH]", \
                               self.slc_file.check_freq_pol)
        
    def test_gslc_extra_polarizations(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "wrong_polarizations3.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] has extra polarization [HH,VV,HV,VH]", \
                               self.slc_file.check_freq_pol)

    def test_gcov_no_polarizations(self):

        self.slc_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations1.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] is missing polarization list", \
                               self.slc_file.check_freq_pol)
        
    def test_gcov_missing_polarizations(self):

        self.slc_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations2.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] has extra polarization .*HV.*", \
                               self.slc_file.check_freq_pol)
        
    def test_gcov_extra_polarizations(self):

        self.slc_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations3.h5"), mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] is missing polarization .*VH.*", \
                               self.slc_file.check_freq_pol)

    def test_gcov_badname_polarizations1(self):

        self.slc_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations4.h5"), mode="r")
        self.slc_file.get_bands()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] has invalid polarization: HHVVH", \
                               self.slc_file.get_freq_pol)

    def test_gcov_badname_polarizations2(self):

        self.slc_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations5.h5"), mode="r")
        self.slc_file.get_bands()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] has invalid polarization: HHRV", \
                               self.slc_file.get_freq_pol)

    def test_gcov_badname_polarizations3(self):

        self.slc_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations6.h5"), mode="r")
        self.slc_file.get_bands()
        self.assertRaisesRegex(errors_base.FatalError, "[L,S]SAR Frequency[A,B] has invalid polarization: RHRZ", \
                               self.slc_file.get_freq_pol)

    def test_missing_none(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "missing_none.h5"), xml_tree=self.xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.find_missing_datasets()
        self.slc_file.close()

    def test_missing_one(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "missing_one.h5"), xml_tree=self.xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.assertRaisesRegex(errors_base.WarningError, "LSAR Identification missing 1 fields: .*absoluteOrbitNumber", \
                               self.slc_file.find_missing_datasets)
        

        
        
if __name__ == "__main__":
    unittest.main()

        
        

