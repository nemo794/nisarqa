from quality.SLCFile import SLCFile
from quality.SLCImage import SLCImage
from quality import errors_base, errors_derived, utility

import h5py
import numpy

import os, os.path
import unittest

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"

    def setUp(self):
        pass
        
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
 

        
if __name__ == "__main__":
    unittest.main()

        
        

