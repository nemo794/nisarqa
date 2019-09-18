import errors_base
import errors_derived
import params
import utility

import h5py
from matplotlib import cm, pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np

import copy
import datetime
import logging
import os
import sys
import tempfile
import traceback

class SLCFile(h5py.File):

    def __init__(self, flname, mode):
        self.flname = os.path.basename(flname)
        h5py.File.__init__(self, flname, mode)

        print("Opening file %s" % flname)
        self.SWATHS = self["/science/LSAR/SLC/swaths"]
        self.IDENTIFICATION = self["/science/LSAR/identification/"]
        self.FREQUENCIES = {}

        self.images = {}
        self.polarizations = {}
        self.start_time = self.IDENTIFICATION.get("zeroDopplerStartTime")[...]

        self.frequencies = [f.decode() for f in self.IDENTIFICATION.get("listOfFrequencies")[...]]
        try:
            assert(self.frequencies in params.FREQUENCIES)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            raise errors_derived.IdentificationFatal(self.flname, self.start_time, traceback_string, \
                                                     ["Invalid frequency list of %s" % self.frequencies])

        for f in self.frequencies:
            self.FREQUENCIES[f] = self["/science/LSAR/SLC/swaths/frequency%s" % f]
            self.polarizations[f] = [p.decode() for p in self.FREQUENCIES[f].get("listOfPolarizations")[...]]

        missing_images = []
        for f in self.frequencies:
            try:
                assert(self.polarizations[f] in params.POLARIZATIONS)
            except AssertionError as e:
                traceback_string = [utility.get_traceback(e, AssertionError)]
                raise errors_derived.IdentificationFatal(self.flname, self.start_time, traceback_string, \
                                                         ["Frequency%s has Invalid polarization list %s" \
                                                          % (f, self.polarizations[f])])

            for p in self.polarizations[f]:
                key = "%s_%s" % (f, p)
                try:
                    self.images[key] = self.FREQUENCIES[f]["%s" % p]
                except KeyError:
                    missing_images.append(key)

        try:
            assert(len(missing_images) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            raise errors_derived.ArrayMissingFatal(self.flname, self.start_time, traceback_string, \
                                                   ["Missing %i images: %s" % (len(missing_images), missing_images)])
            
    def check_identification(self):

        error_string = []
        traceback_string = []
    
        orbit = self.IDENTIFICATION.get("absoluteOrbitNumber")[...]
        track = self.IDENTIFICATION.get("trackNumber")[...]
        frame = self.IDENTIFICATION.get("frameNumber")[...]
        #cycle = self.IDENTIFICATION.get("cycleNumber")[...]
        lookdir = self.IDENTIFICATION.get("lookDirection")[...]
        #passdir = self.IDENTIFICATION.get("orbitPassDirection")[...]
        #ptype = self.IDENTIFICATION.get("productType")[...]
        start_time = self.IDENTIFICATION.get("zeroDopplerStartTime")[...]
        end_time = self.IDENTIFICATION.get("zeroDopplerEndTime")[...]

        try:
            assert(str(end_time) < str(start_time))
        except AssertionError as e:
            error_string += ["Start Time %s not less than End Time %s" % (start_time, end_time)]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            assert(orbit > 0)
        except AssertionError as e:
            error_string += ["Invalid Orbit Number: %i" % orbit]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            assert( (track > 0) and (track <= params.NTRACKS) )
        except AssertionError as e:
            error_string += ["Invalid Track Number: %i" % track]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            assert(frame > 0)
        except AssertionError as e:
            error_string += ["Invalid Frame Number: %i" % frame]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        #try:
        #    assert(cycle > 0)
        #except AssertionError as e:
        #    error_string += ["Invalid Cycle Number: %i" % cycle]
        #    traceback_string.append(utility.get_traceback(e, AssertionError))

        #try:
        #    assert(str(ptype) in ("b'RRST'", "b'RRSD'", "b'RSLC'", "b'RMLC'", \
        #                          "b'RCOV'", "b'RIFG'", "b'RUNW'", "b'GUNW'", \
        #                          "b'CGOV'", "b'GSLC'"))
        #except AssertionError as e:
        #    error_string += ["Invalid Product Type: %i" % ptype]
        #    traceback_string.append(utility.get_traceback(e, AssertionError))

        try:
            assert(str(lookdir) in ("b'left'", "b'right'"))
        except AssertionError as e:
            error_string += ["Invalid Look Number: %s" % lookdir]
            traceback_string.append(utility.get_traceback(e, AssertionError))

        assert(len(error_string) == len(traceback_string))
            
        try:
            assert(len(error_string) == 0)
        except AssertionError as e:
            #print("error_string: %s" % error_string)
            #print("traceback_string: %s" % traceback_string)
            raise errors_derived.IdentificationFatal(self.flname, self.start_time, traceback_string, error_string)

    def check_images(self, fpdf):

        min_value = np.array([np.finfo(np.float64).max, np.finfo(np.float64).max])
        max_value = np.array([np.finfo(np.float64).min, np.finfo(np.float64).min])
        number_nan = {}
        data = {}
        mask_nan = {}

        # Get min/max value of all images and also check for NaNs.
        
        for key in self.images.keys():
            ximg = self.images[key]
            complex32 = False
            try:
                dtype = ximg.dtype
            except TypeError:
                complex32 = True

            if (complex32):
                with ximg.astype(np.complex64):
                    xdata = ximg[...]
            else:
                xdata = ximg[...]

            xnan = np.isnan(xdata.real) | np.isnan(xdata.imag)
            if (np.any(xnan)):
                number_nan[i] = xnan.sum()

            data[key] = xdata
            mask_nan[key] = xnan
                
            xmin = np.array([xdata.real[~xnan].min(), xdata.imag[~xnan].min()])
            xmax = np.array([xdata.imag[~xnan].max(), xdata.imag[~xnan].max()])

            min_value = np.minimum(min_value, xmin)
            max_value = np.maximum(max_value, xmax)
 
        # Generate histograms (real and imaginary components separately) and plot them

        xmax = np.maximum(np.fabs(min_value), np.fabs(max_value)).max()
        for i in range(1, 100):
            if (10**i > xmax):
                break

        if (xmax < 10**i/2.0):
            bounds = 10**i/2
        else:
            bounds = 10**i

        bounds_plot_data = (np.finfo(np.float32).max, np.finfo(np.float32).min)
        bounds_plot_diff = (np.finfo(np.float32).max, np.finfo(np.float32).min)
            
        for key in self.images.keys():

            # Histogram with huge number of bins to find out where the bounds of the real
            # distribution lie and then rehistogram with more sensible bounds.  Do this for
            # individual real and complex components and then the difference between the two.
            
            real = data[key].real[~mask_nan[key]]
            imag = data[key].imag[~mask_nan[key]]
            diff = real - imag

            (counts1r, edges1r) = np.histogram(real, bins=1000)
            (counts1c, edges1c) = np.histogram(imag, bins=1000)
            bounds1 = utility.hist_bounds(counts1r+counts1c, edges1r)
 
            (counts2, edges2) = np.histogram(diff, bins=1000)
            bounds2 = utility.hist_bounds(counts2, edges2)

            if (bounds1 < bounds_plot_data[0]) or (bounds1 > bounds_plot_data[1]):
                bounds_plot_data = (-bounds1, bounds1)

            if (bounds2 < bounds_plot_diff[0]) or (bounds2 > bounds_plot_diff[1]):
                bounds_plot_diff = (-bounds2, bounds2)

        for key in self.images.keys():

            # Use same bounds for plotting all images of a given type
            
            (counts1r, edges1r) = np.histogram(real, range=bounds_plot_data, bins=100)
            (counts1c, edges1c) = np.histogram(imag, range=bounds_plot_data, bins=100)
            (counts2, edges2) = np.histogram(diff, range=bounds_plot_diff, bins=100)
            
            (fig, axes) = pyplot.subplots(nrows=2, ncols=1, sharex=False, sharey=False)
            axes[0].plot(edges1r[:-1], counts1r, label="real")
            axes[0].plot(edges1c[:-1], counts1c, label="imaginary")
            axes[1].plot(edges2[:-1], counts2, label="real-imaginary")
            axes[0].legend(loc="upper right", fontsize="small")
            axes[1].legend(loc="upper right", fontsize="small")
            fig.suptitle("%s\n(Frequency%s)" % (self.flname, key))

            fpdf.savefig(fig)
            pyplot.close(fig)

        # Raise Warning if any NaNs are found
            
        nbad = len(number_nan.keys())
        try:
            assert(nbad == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            raise errors_derived.NaNWarning(self.flname, self.start_time, traceback_string, \
                                            ["%i images had NaN's: %s" % (nbad, number_nan)])

    def check_time(self):

        time = self.SWATHS["zeroDopplerTime"][...]
        spacing = self.SWATHS["zeroDopplerTimeSpacing"][...]
        start_time = self.IDENTIFICATION["zeroDopplerStartTime"][...]
        end_time = self.IDENTIFICATION["zeroDopplerEndTime"][...]

        start_time = str(start_time, "utf-8").rstrip("\x00").split(".")[0]
        end_time = str(end_time, "utf-8").rstrip("\x00").split(".")[0]
    
        try:
            time1 = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
            time2 = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
            assert(time2 > time1)
            assert( (time1.year >= 2000) and (time1.year < 2100) )
            assert( (time2.year >= 2000) and (time2.year < 2100) )
        except (AssertionError, ValueError) as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            raise errors_derived.IdentificationFatal(self.flname, self.start_time, traceback_string, \
                                                     ["Invalid Times of %s and %s" % (start_time, end_time)])

        try:
            utility.check_spacing(self.flname, self.start_time, time, spacing, "Time", \
                                  errors_derived.TimeSpacingWarning, errors_derived.TimeSpacingFatal)
        except (errors_base.WarningError, errors_base.FatalError):
            pass

    def check_frequencies(self):

        nfrequencies = len(self.frequencies)
        if (nfrequencies == 2):
            acquired_freq = {}
            for f in self.frequencies:
                acquired_freq[f] = self.FREQUENCIES[f]["processedCenterFrequency"][...]

            try:
                assert(acquired_freq["A"] > acquired_freq["B"])
            except AssertionError as e:
                traceback_string = [utility.get_traceback(e, AssertionError)]
                raise errors_derived.FrequencyOrderFatal(self.flname, self.start_time, traceback_string, \
                                                         ["Frequency A=%f not less than Frequency B=%f" \
                                                          % (acquired_freq["A"], acquired_freq["B"])])

    def check_slant_range(self):

        for f in self.frequencies:
            slant_path = self.FREQUENCIES[f]["slantRange"]
            spacing = self.FREQUENCIES[f]["slantRangeSpacing"]

            try:
                utility.check_spacing(self.flname, self.start_time, slant_path[...], spacing[...], \
                                      "%sSlantPath" % f, errors_derived.SlantSpacingWarning, \
                                      errors_derived.SlantSpacingFatal)
            except (errors_base.WarningError, errors_base.FatalError):
                pass

        
        for key in self.images.keys():
            (f, p) = key.split("_")
            
            nslant0 = self.FREQUENCIES[f]["slantRange"].shape[0]
            nslant = self.images[key].shape[1]

            try:
                assert(nslant == nslant0)
            except AssertionError as e:
                traceback_string = [utility.get_traceback(e, AssertionError)]
                raise errors_derived.ArraySizeFatal(self.flname, self.start_time, traceback_string, \
                                                    "Dataset %s has slantpath size of %i instead of %i" \
                                                    % (key, nslant, nslant0))


    def check_subswaths(self):
        
       for f in self.frequencies:
           nsubswath = self.FREQUENCIES[f]["numberOfSubSwaths"][...]
           try:
               assert( (nsubswath >= 0) and (nsubswath <= params.NSUBSWATHS) )
           except AssertionError as e:
               traceback_string = [utility.get_traceback(e, AssertionError)]
               raise errors_derived.NumSubswathFatal(self.flname, self.start_time, traceback_string, \
                                                     ["Frequency%s had invalid number of subswaths: %i" % (f, nsubswath)])

           nslantrange = self.FREQUENCIES[f]["slantRange"].size
           
           for isub in range(0, nsubswath):
               try:
                   sub_bounds = self.FREQUENCIES[f]["validSamplesSubSwath%i" % (isub+1)][...]
               except KeyError as e:
                   traceback_string = [utility.get_traceback(e, KeyError)]
                   raise errors_derived.MissingSubswathFatal(self.flname, self.start_time, traceback_string, \
                                                             ["Frequency%s had missing SubSwath%i bounds" % (f, isub)])

               try:
                   assert(np.all(sub_bounds[:, 0] < sub_bounds[:, 1]))
                   assert(np.all(sub_bounds[:, 0] >= 0))
                   assert(np.all(sub_bounds[:, 1] <= nslantrange))
               except AssertionError as e:
                   traceback_string = [utility.get_traceback(e, KeyError)]
                   raise errors_derived.BoundsSubSwathFatal(self.flname, self.start_time, traceback_string, \
                                                            ["Frequency%s with nSlantRange %i had invalid SubSwath bounds: %s" \
                                                             % (f, nslantrange, sub_bounds)])
               

               

