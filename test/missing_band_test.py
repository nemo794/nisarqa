from quality.GCOVFile import GCOVFile
from quality.GSLCFile import GSLCFile
from quality.GUNWFile import GUNWFile
from quality.SLCFile import SLCFile
from quality import errors_base, errors_derived, logging_base

import h5py
from matplotlib.backends.backend_pdf import PdfPages
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
    
    def test_missing_band(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "gslc_missing_band.h5"), \
                                 self.logger, xml_tree=self.gslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.lcapture.check_present(("NISAR", "ERROR", "/science/LSAR/GSLC/metadata does not exist"), \
                                    order_matters=False)

    def test_slc_inconsistent_bands(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_inconsistent_bands.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.lcapture.check_present(("NISAR", "ERROR", "Metadata and swath have inconsistent naming"), \
                                    order_matters=False)
                                
    def test_gslc_incorrect_frequency(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "gslc_wrong_frequencies1.h5"), \
                                 self.logger, xml_tree=self.gslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR has invalid frequency list"), \
                                    order_matters=False)

    def test_gslc_missing_frequency(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "gslc_wrong_frequencies2.h5"), \
                                 self.logger, xml_tree=self.gslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR missing FrequencyB"), \
                                    order_matters=False)        

    def test_gslc_extra_frequency(self):

        self.slc_file = GSLCFile(os.path.join(self.TEST_DIR, "gslc_wrong_frequencies3.h5"), \
                                 self.logger, xml_tree=self.gslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR frequency list missing A"), \
                                    order_matters=False)        
        
    def test_gcov_missing_metadata(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_missing_metadata.h5"), \
                                  self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()

        self.lcapture.check_present(("NISAR", "ERROR", "/science/LSAR/GCOV/metadata does not exist"), \
                                    order_matters=False)

        
    def test_gcov_incorrect_frequency(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_frequencies1.h5"), \
                                 self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR has invalid frequency list"), \
                                    order_matters=False)

    def test_gcov_missing_frequency(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_frequencies2.h5"), \
                                 self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR missing FrequencyA"), \
                                    order_matters=False)

    def test_gcov_extra_frequency(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_frequencies3.h5"), \
                                 self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR frequency list missing B"), \
                                    order_matters=False)
        
    def test_gslc_incorrect_polarizations(self):

        self.gslc_file = GSLCFile(os.path.join(self.TEST_DIR, "gslc_wrong_polarizations1.h5"), \
                                 self.logger, xml_tree=self.gslc_xml_tree, mode="r")
        self.gslc_file.get_bands()
        self.gslc_file.get_freq_pol()
        self.gslc_file.check_freq_pol("LSAR", [self.gslc_file.SWATHS], [self.gslc_file.FREQUENCIES], [""])

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR FrequencyA has invalid polarization list: ['HH', 'XY']"), \
                                    order_matters=False)
        
    def test_gslc_missing_polarizations(self):

        self.gslc_file = GSLCFile(os.path.join(self.TEST_DIR, "gslc_wrong_polarizations2.h5"), \
                                 self.logger, xml_tree=self.gslc_xml_tree, mode="r")
        self.gslc_file.get_bands()
        self.gslc_file.get_freq_pol()
        self.gslc_file.check_freq_pol("LSAR", [self.gslc_file.SWATHS], [self.gslc_file.FREQUENCIES], [""])

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR FrequencyA missing polarization HV"), \
                                    order_matters=False)
        
    def test_gslc_extra_polarizations(self):

        self.gslc_file = GSLCFile(os.path.join(self.TEST_DIR, "gslc_wrong_polarizations3.h5"), \
                                 self.logger, xml_tree=self.gslc_xml_tree, mode="r")
        self.gslc_file.get_bands()
        self.gslc_file.get_freq_pol()
        self.gslc_file.check_freq_pol("LSAR", [self.gslc_file.SWATHS], [self.gslc_file.FREQUENCIES], [""])

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR FrequencyB has extra polarization VH"), \
                                    order_matters=False)

    def test_gcov_no_polarizations(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations1.h5"), \
                                 self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR FrequencyA is missing polarization list"), \
                                    order_matters=False)
        
    def test_gcov_missing_polarizations(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations2.h5"), \
                                 self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR FrequencyA has extra polarization ['HV']"), \
                                    order_matters=False)        
        
    def test_gcov_extra_polarizations(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations3.h5"), \
                                 self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()
        self.gcov_file.check_freq_pol()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR FrequencyA is missing polarization ['VH']"), \
                                    order_matters=False)             

    def test_gcov_badname_polarizations1(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations4.h5"), \
                                 self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR FrequencyA has invalid polarization: HHVVH"), \
                                    order_matters=False)
        
    def test_gcov_badname_polarizations2(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations5.h5"), \
                                 self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR FrequencyA has invalid polarization: HHRV"), \
                                    order_matters=False)       

    def test_gcov_badname_polarizations3(self):

        self.gcov_file = GCOVFile(os.path.join(self.TEST_DIR, "gcov_wrong_polarizations6.h5"), \
                                 self.logger, xml_tree=self.gcov_xml_tree, mode="r")
        self.gcov_file.get_bands()
        self.gcov_file.get_freq_pol()

        self.lcapture.check_present(("NISAR", "ERROR", "LSAR FrequencyA has invalid polarization: RHRZ"), \
                                    order_matters=False)            

    def test_missing_none(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_missing_none.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.find_missing_datasets([self.slc_file.SWATHS], [self.slc_file.FREQUENCIES])
        self.slc_file.close()

        logs_captured = str(LogCapture())
        self.assertTrue("No logging captured" in logs_captured)

    def test_missing_one(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "rslc_missing_one.h5"), \
                                self.logger, xml_tree=self.rslc_xml_tree, mode="r")
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.find_missing_datasets([self.slc_file.SWATHS], [self.slc_file.FREQUENCIES])

        message = "LSAR Identification missing 1 fields: /science/LSAR/identification/absoluteOrbitNumber"
        self.lcapture.check_present(("NISAR", "WARNING", message), order_matters=False)

    def test_gunw_missing_swath(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_noswath.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.assertFalse(self.gunw_file.has_swath["LSAR"])
        
    def test_gunw_missing_metadata(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_nometa.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.lcapture.check_present(("NISAR", "ERROR", "/science/LSAR/GUNW/metadata does not exist"), \
                                    order_matters=False)

    def test_gunw_missing_none(self):

        fpdf = PdfPages("junk.pdf")
        fhdf = h5py.File("junk.h5", "w")
        
        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_missing_none.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.find_missing_datasets()
        self.gunw_file.create_images()
        self.gunw_file.check_images(fpdf, fhdf)
        self.gunw_file.close()

        fpdf.close()
        fhdf.close()

        logs_captured = str(LogCapture())
        self.assertTrue("No logging captured" in logs_captured)

    def test_gunw_missing_images(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_nooffset.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.create_images()

        message = "File is missing 1 images: ['Offset: (LSAR, A, VV)']"
        self.lcapture.check_present(("NISAR", "ERROR", message), order_matters=False)
         
    def test_gunw_missing_image_component(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_nophase.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.create_images()

        message = "Grid: (LSAR A HH) is missing datasets: ['unwrappedPhase']"
        self.lcapture.check_present(("NISAR", "ERROR", message), order_matters=False)        
        
    def test_gunw_missing_all_component(self):

        self.gunw_file = GUNWFile(os.path.join(self.TEST_DIR, "gunw_missing_all.h5"), \
                                  self.logger, xml_tree=self.gunw_xml_tree, mode="r")
        self.gunw_file.get_bands()
        self.gunw_file.get_freq_pol()
        self.gunw_file.check_freq_pol("LSAR", [self.gunw_file.GRIDS], [self.gunw_file.FREQUENCIES_GRID], ["Grids"])
        self.gunw_file.create_images()

        message = "Grid: (LSAR A VV) is missing all datasets"
        self.lcapture.check_present(("NISAR", "ERROR", message), order_matters=False)        
        
if __name__ == "__main__":
    unittest.main()

        
        

