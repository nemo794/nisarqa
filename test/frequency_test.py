from quality.SLCFile import SLCFile
from quality.SLCImage import SLCImage
from quality import errors_base, errors_derived, utility

import h5py
import numpy
from scipy import constants
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pyplot
from matplotlib.backends.backend_pdf import PdfPages

import os, os.path
import sys
import unittest

class SLCFile_test(unittest.TestCase):

    TEST_DIR = "test_data"

    def setUp(self):
        pass
        
    def test_frequency(self):

        # Re-open file and calculate power vs. frequency

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "frequency1.h5"), mode="r")
        fhdf = h5py.File(os.path.join(self.TEST_DIR, "frequency1_out.h5"), "w")
        fpdf = PdfPages(os.path.join(self.TEST_DIR, "frequency1_out.pdf"))
        
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        try:
            self.slc_file.check_nans()
        except errors_base.WarningError:
            pass
        self.slc_file.check_images(fpdf, fhdf)

        self.slc_file.close()
        fhdf.close()
        fpdf.close()
        
        # Open hdf summary file and verify frequencies

        summary_file = h5py.File(os.path.join(self.TEST_DIR, "frequency1_out.h5"), "r")
        power = summary_file["/frequency1/LSAR/ImageAttributes/LSAR A HH/Average Power"][...]
        frequency = summary_file["/frequency1/LSAR/ImageAttributes/LSAR A HH/FFT Spacing"][...] 
        summary_file.close()
        
        max_power = power.max()
        idx = numpy.where(power > 0.5*max_power)
        found_frequency = numpy.array(frequency[idx])
        expect_frequency = numpy.array([-0.5, -0.1, 0.1, 0.5])
        print("found frequencies %s" % found_frequency)

        self.assertTrue(len(idx[0]) == 4)
        self.assertTrue(numpy.all(numpy.fabs(found_frequency - expect_frequency) <= 0.01))

    def test_power_phase(self):

        # Re-open file and calculate power and phase

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "power_phase1.h5"), mode="r")
        fhdf = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out1.h5"), "w")
        fpdf = PdfPages(os.path.join(self.TEST_DIR, "power_phase_out1.pdf"))
        
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        try:
            self.slc_file.check_nans()
        except errors_base.WarningError:
            pass
        self.slc_file.check_images(fpdf, fhdf)

        self.slc_file.close()
        fhdf.close()
        fpdf.close()
        
        summary_file = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out1.h5"), "r")
        mean_power = summary_file["/power_phase1/LSAR/ImageAttributes/LSAR A HH/MeanPower"][...]
        sdev_power = summary_file["/power_phase1/LSAR/ImageAttributes/LSAR A HH/SDevPower"][...] 
        mean_phase = summary_file["/power_phase1/LSAR/ImageAttributes/LSAR A HH/MeanPhase"][...]
        sdev_phase = summary_file["/power_phase1/LSAR/ImageAttributes/LSAR A HH/SDevPhase"][...] 
        summary_file.close()
        
        array_power = numpy.array((5.0, 20.0, 11.25, 20.0))
        array_power = 10.0*numpy.log10(array_power)
        array_phase = numpy.array((numpy.angle(2.0+1.0j, deg=True), numpy.angle(-1.0-2.0j, deg=True)))

        self.assertAlmostEqual(mean_power, array_power.mean(), places=2)
        self.assertAlmostEqual(sdev_power, array_power.std(), places=2)
        self.assertAlmostEqual(mean_phase, array_phase.mean(), places=2)
        self.assertAlmostEqual(sdev_phase, array_phase.std(), places=2)

    def test_power_phase_nan(self):

        # Re-open file and calculate power and phase

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "power_phase2.h5"), mode="r")
        fhdf = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out2.h5"), "w")
        fpdf = PdfPages(os.path.join(self.TEST_DIR, "power_phase_out2.pdf"))
        
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        try:
            self.slc_file.check_nans()
        except errors_base.WarningError:
            pass
        self.slc_file.check_images(fpdf, fhdf)

        self.slc_file.close()
        fhdf.close()
        fpdf.close()
        
        summary_file = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out2.h5"), "r")
        mean_power = summary_file["/power_phase2/LSAR/ImageAttributes/LSAR A HH/MeanPower"][...]
        sdev_power = summary_file["/power_phase2/LSAR/ImageAttributes/LSAR A HH/SDevPower"][...] 
        mean_phase = summary_file["/power_phase2/LSAR/ImageAttributes/LSAR A HH/MeanPhase"][...]
        sdev_phase = summary_file["/power_phase2/LSAR/ImageAttributes/LSAR A HH/SDevPhase"][...] 
        summary_file.close()
        
        array_power = 10.0*numpy.log10((11.25, 5.0))
        array_phase = numpy.array((numpy.angle(2.0+1.0j, deg=True), numpy.angle(-1.0-2.0j, deg=True)))

        self.assertAlmostEqual(mean_power, array_power.mean(), places=2)
        self.assertAlmostEqual(sdev_power, array_power.std(), places=2)
        self.assertAlmostEqual(mean_phase, array_phase.mean(), places=2)
        self.assertAlmostEqual(sdev_phase, array_phase.std(), places=2)

    def test_power_phase_combination(self):

        # Re-open file and calculate power and phase

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "power_phase3.h5"), mode="r")
        fhdf = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out3.h5"), "w")
        fpdf = PdfPages(os.path.join(self.TEST_DIR, "power_phase_out3.pdf"))
        
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol("LSAR", [self.slc_file.SWATHS], [self.slc_file.FREQUENCIES], [""])
        self.slc_file.create_images()
        try:
            self.slc_file.check_nans()
        except errors_base.WarningError:
            pass
        self.slc_file.check_images(fpdf, fhdf)

        self.slc_file.close()
        fhdf.close()
        fpdf.close()
        
        summary_file = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out3.h5"), "r")
        mean_power = summary_file["/power_phase3/LSAR/ImageAttributes/LSAR A HH-HV/MeanPower"][...]
        sdev_power = summary_file["/power_phase3/LSAR/ImageAttributes/LSAR A HH-HV/SDevPower"][...] 
        mean_phase = summary_file["/power_phase3/LSAR/ImageAttributes/LSAR A HH-HV/MeanPhase"][...]
        sdev_phase = summary_file["/power_phase3/LSAR/ImageAttributes/LSAR A HH-HV/SDevPhase"][...] 
        summary_file.close()

        cnumber = (1.0+2.0j)*(10.0-20.0j)
        array_power = 10.0*numpy.log10(50.0*50.0)
        array_phase = numpy.array(numpy.angle(50.0+0.0j, deg=True))

        self.assertAlmostEqual(mean_power, array_power, places=2)
        self.assertAlmostEqual(sdev_power, 0.0, places=2)
        self.assertAlmostEqual(mean_phase, array_phase, places=2)
        self.assertAlmostEqual(sdev_phase, 0.0, places=2)
        
if __name__ == "__main__":
    unittest.main()

        
        

