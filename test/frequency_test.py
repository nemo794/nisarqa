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

        # Set up parameters

        # Open for the first time and insert dummy data
        
        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "frequency1.h5"), "r+")
 
        dset = self.slc_file["/science/LSAR/SLC/swaths/frequencyA/HH"]
        frequency = self.slc_file["/science/LSAR/SLC/swaths/frequencyA/processedCenterFrequency"]
        rspacing = self.slc_file["/science/LSAR/SLC/swaths/frequencyA/slantRangeSpacing"]
        tspacing = (constants.c/2.0)/rspacing[...]
        tinterval = 1.0/tspacing

        shape = dset.shape
        time = numpy.arange(0, 1.0*shape[1]*tinterval, tinterval)
        freq1 = 1.0*1.0E5
        freq2 = 5.0*1.0E5
        
        raw1 = 2*numpy.pi*freq1*time
        raw2 = 2*numpy.pi*freq2*time
        real1 = numpy.sin(raw1).astype(numpy.float32)
        real2 = numpy.sin(raw2).astype(numpy.float32)
        imag = numpy.zeros(real1.shape, dtype=numpy.float32)

        real = real1+real2
        xdata = real+imag*1j
        xdata = numpy.tile(xdata, (shape[0])).reshape(shape[0], xdata.size)
        dset[...] = xdata
        self.slc_file.close()

        # Re-open file and calculate power vs. frequency

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "frequency1.h5"), "r")
        fhdf = h5py.File(os.path.join(self.TEST_DIR, "frequency1_out.h5"), "w")
        fpdf = PdfPages(os.path.join(self.TEST_DIR, "frequency1_out.pdf"))
        
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.slc_file.create_images()
        self.slc_file.check_nans()
        self.slc_file.check_images(fpdf, fhdf)

        self.slc_file.close()
        fhdf.close()
        fpdf.close()
        
        # Open hdf file and verify frequencies

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

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "power_phase1.h5"), "r+")

        dset = self.slc_file["/science/LSAR/SLC/swaths/frequencyA/VV"]
        shape = dset.shape
        nlines = shape[1]//4

        real = numpy.zeros(shape, dtype=numpy.float32)
        imag = numpy.zeros(shape, dtype=numpy.float32)
        
        real[:, 0:nlines] = -1.0
        real[:, nlines:2*nlines] = -2.0
        real[:, 2*nlines:3*nlines] = 3.0
        real[:, 3*nlines:4*nlines] = 4.0

        imag[:, 0:nlines] = -2.0
        imag[:, nlines:2*nlines] = -4.0
        imag[:, 2*nlines:3*nlines] = 1.5
        imag[:, 3*nlines:4*nlines] = 2.0

        xdata = real + 1.0j*imag
        dset[...] = xdata
        self.slc_file.close()

        # Re-open file and calculate power and phase

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "power_phase1.h5"), "r")
        fhdf = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out1.h5"), "w")
        fpdf = PdfPages(os.path.join(self.TEST_DIR, "power_phase_out1.pdf"))
        
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
        self.slc_file.create_images()
        self.slc_file.check_nans()
        self.slc_file.check_images(fpdf, fhdf)

        self.slc_file.close()
        fhdf.close()
        fpdf.close()
        
        summary_file = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out1.h5"), "r")
        mean_power = summary_file["/power_phase1/LSAR/ImageAttributes/LSAR A VV/MeanPower"][...]
        sdev_power = summary_file["/power_phase1/LSAR/ImageAttributes/LSAR A VV/SDevPower"][...] 
        mean_phase = summary_file["/power_phase1/LSAR/ImageAttributes/LSAR A VV/MeanPhase"][...]
        sdev_phase = summary_file["/power_phase1/LSAR/ImageAttributes/LSAR A VV/SDevPhase"][...] 
        summary_file.close()
        
        array_power = numpy.array((5.0, 20.0, 11.25, 20.0))
        array_power = 10.0*numpy.log10(array_power)
        array_phase = numpy.array((numpy.angle(2.0+1.0j, deg=True), numpy.angle(-1.0-2.0j, deg=True)))

        self.assertAlmostEqual(mean_power, array_power.mean(), places=2)
        self.assertAlmostEqual(sdev_power, array_power.std(), places=2)
        self.assertAlmostEqual(mean_phase, array_phase.mean(), places=2)
        self.assertAlmostEqual(sdev_phase, array_phase.std(), places=2)

    def test_power_phase_nan(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "power_phase2.h5"), "r+")

        dset = self.slc_file["/science/LSAR/SLC/swaths/frequencyA/VV"]
        shape = dset.shape
        nlines = shape[1]//4

        real = numpy.zeros(shape, dtype=numpy.float32)
        imag = numpy.zeros(shape, dtype=numpy.float32)
        
        real[:, 0:nlines] = -1.0
        real[:, nlines:2*nlines] = numpy.nan
        real[:, 2*nlines:3*nlines] = 3.0
        real[:, 3*nlines:4*nlines] = 0.0

        imag[:, 0:nlines] = -2.0
        imag[:, nlines:2*nlines] = -4.0
        imag[:, 2*nlines:3*nlines] = 1.5
        imag[:, 3*nlines:4*nlines] = 0.0

        xdata = real + 1.0j*imag
        dset[...] = xdata
        self.slc_file.close()

        # Re-open file and calculate power and phase

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "power_phase2.h5"), "r")
        fhdf = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out2.h5"), "w")
        fpdf = PdfPages(os.path.join(self.TEST_DIR, "power_phase_out2.pdf"))
        
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
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
        mean_power = summary_file["/power_phase2/LSAR/ImageAttributes/LSAR A VV/MeanPower"][...]
        sdev_power = summary_file["/power_phase2/LSAR/ImageAttributes/LSAR A VV/SDevPower"][...] 
        mean_phase = summary_file["/power_phase2/LSAR/ImageAttributes/LSAR A VV/MeanPhase"][...]
        sdev_phase = summary_file["/power_phase2/LSAR/ImageAttributes/LSAR A VV/SDevPhase"][...] 
        summary_file.close()
        
        array_power = 10.0*numpy.log10((11.25, 5.0))
        array_phase = numpy.array((numpy.angle(2.0+1.0j, deg=True), numpy.angle(-1.0-2.0j, deg=True)))

        self.assertAlmostEqual(mean_power, array_power.mean(), places=2)
        self.assertAlmostEqual(sdev_power, array_power.std(), places=2)
        self.assertAlmostEqual(mean_phase, array_phase.mean(), places=2)
        self.assertAlmostEqual(sdev_phase, array_phase.std(), places=2)

    def test_power_phase_combination(self):

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "power_phase3.h5"), "r+")

        dset_hh = self.slc_file["/science/LSAR/SLC/swaths/frequencyA/HH"]
        dset_vv = self.slc_file["/science/LSAR/SLC/swaths/frequencyA/VV"]
        shape = dset_hh.shape
        nlines = shape[1]//4

        real1 = numpy.zeros(shape, dtype=numpy.float32) 
        imag1 = numpy.zeros(shape, dtype=numpy.float32)
        real2 = numpy.zeros(shape, dtype=numpy.float32) 
        imag2 = numpy.zeros(shape, dtype=numpy.float32)
        
        real1[:, 0:2*nlines] = 1.0
        real1[:, 2*nlines:4*nlines] = 0.0 
        real2[:, 0:2*nlines] = 10.0
        real2[:, 2*nlines:4*nlines] = numpy.nan

        imag1[:, 0:2*nlines] = 2.0
        imag1[:, 2*nlines:4*nlines] = 0.0
        imag2[:, 0:2*nlines] = 20.0
        imag2[:, 2*nlines:4*nlines] = 200.0

        xdata1 = real1 + 1.0j*imag1
        xdata2 = real2 + 1.0j*imag2
        dset_hh[...] = xdata1
        dset_vv[...] = xdata2
        self.slc_file.close()

        # Re-open file and calculate power and phase

        self.slc_file = SLCFile(os.path.join(self.TEST_DIR, "power_phase3.h5"), "r")
        fhdf = h5py.File(os.path.join(self.TEST_DIR, "power_phase_out3.h5"), "w")
        fpdf = PdfPages(os.path.join(self.TEST_DIR, "power_phase_out3.pdf"))
        
        self.slc_file.get_bands()
        self.slc_file.get_freq_pol()
        self.slc_file.check_freq_pol()
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
        mean_power = summary_file["/power_phase3/LSAR/ImageAttributes/LSAR A HH-VV/MeanPower"][...]
        sdev_power = summary_file["/power_phase3/LSAR/ImageAttributes/LSAR A HH-VV/SDevPower"][...] 
        mean_phase = summary_file["/power_phase3/LSAR/ImageAttributes/LSAR A HH-VV/MeanPhase"][...]
        sdev_phase = summary_file["/power_phase3/LSAR/ImageAttributes/LSAR A HH-VV/SDevPhase"][...] 
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

        
        

