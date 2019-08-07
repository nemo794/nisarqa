import errors_base
import errors_derived
import params
import utility

import h5py
from matplotlib import cm, pyplot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np

import datetime
import os

class SLCFile(h5py.File):

    def __init__(self, flname, mode):
        self.flname = flname
        h5py.File.__init__(self, flname, mode)

        self.SWATHS = self["/science/LSAR/SLC/swaths"]
        self.IDENTIFICATION = self["/science/LSAR/identification/"]
        self.FREQUENCIES = {"A": self["/science/LSAR/SLC/swaths/frequencyA"], \
                            "B": self["/science/LSAR/SLC/swaths/frequencyB"]}

        self.images = {}
        self.polarizations = {}

        self.frequencies = [f.decode() for f in self.IDENTIFICATION.get("listOfFrequencies")[...]]
        try:
            assert(self.frequencies in params.FREQUENCIES)
        except AssertionError:
            raise errors_derived.IdentificationFatal("Invalid frequency list of %s" % self.frequencies)

        for f in self.frequencies:
            self.polarizations[f] = [p.decode() for p in self.FREQUENCIES[f].get("listOfPolarizations")[...]]

        missing_images = []
        for f in self.frequencies:
            try:
                assert(self.polarizations[f] in params.POLARIZATIONS)
            except AssertionError:
                raise errors_derived.IdentificationFatal("Frequency%s has Invalid polarization list %s" \
                                                         % (f, self.polarizations[f]))

            for p in self.polarizations[f]:
                key = "%s_%s" % (f, p)
                try:
                    self.images[key] = self.FREQUENCIES[f]["%s" % p]
                except KeyError:
                    missing_images.append(key)

        if (len(missing_images) > 0):
            raise errors_derived.ArrayMissingFatal("Missing %i images in file: %s" \
                                                   % (len(missing_images), missing_images))
            
    def check_identification(self):

        error_string = []
    
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
            assert(str(end_time) > str(start_time))
        except AssertionError:
            error_string += ["Start Time %s not less than End Time %s" % (time_start, time_end)]

        try:
            assert(orbit > 0)
        except AssertionError:
            error_string += ["Invalid Orbit Number: %i" % orbit]

        try:
            assert( (track > 0) and (track <= params.NTRACKS) )
        except AssertionError:
            error_string += ["Invalid Track Number: %i" % track]

        try:
            assert(frame > 0)
        except AssertionError:
            error_string += ["Invalid Frame Number: %i" % frame]

        #try:
        #    assert(cycle > 0)
        #except AssertionError:
        #    error_string += ["Invalid Cycle Number: %i" % cycle]

        #try:
        #    assert(str(ptype) in ("b'RRST'", "b'RRSD'", "b'RSLC'", "b'RMLC'", \
        #                          "b'RCOV'", "b'RIFG'", "b'RUNW'", "b'GUNW'", \
        #                          "b'CGOV'", "b'GSLC'"))
        #except AssertionError:
        #    error_string += ["Invalid Product Type: %i" % ptype]

        try:
            assert(str(lookdir) in ("b'left'", "b'right'"))
        except AssertionError:
            error_string += ["Invalid Look Number: %s" % lookdir]

        if (len(error_string) > 0):
            raise errors_derived.IdentificationFatal(error_string)

    def check_images(self, flname_pdf):

        min_value = np.array([np.finfo(np.float64).max, np.finfo(np.float64).max])
        max_value = np.array([np.finfo(np.float64).min, np.finfo(np.float64).min])
        number_nan = {}
        data = {}
        mask_nan = {}

        # Get min/max value of all images and also check for NaNs.
        
        for key in self.images.keys():
            xdata = self.images[key][...]
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

        counts = {}
        edges = {}
        fpdf = PdfPages(flname_pdf)
        
        for key in self.images.keys():

            # Histogram with huge number of bins to find out where the bounds of the real
            # distribution lie
            
            real = data[key].real[~mask_nan[key]]
            imag = data[key].imag[~mask_nan[key]]
            bounds = np.maximum(np.fabs(real), np.fabs(imag)).max()
            (counts[0], edges[0]) = np.histogram(real, range=(-bounds, bounds), bins=1000)
            (counts[1], edges[1]) = np.histogram(imag, range=(-bounds, bounds), bins=1000)

            # Histogram again, this time with sensible bounds (set at 90% of the peak value).

            bounds = utility.hist_bounds(counts[0]+counts[1], edges[0], thresh=0.10)
            (counts[0], edges[0]) = np.histogram(real, range=(-bounds, bounds), bins=100)
            (counts[1], edges[1]) = np.histogram(imag, range=(-bounds, bounds), bins=100)
            
            (fig, axes) = pyplot.subplots(nrows=1, ncols=1, sharex=True, sharey=False)
            axes.plot(edges[0][:-1], counts[0], label="real")
            axes.plot(edges[1][:-1], counts[1], label="imaginary")
            #axes[0].set_title("Real")
            #axes[1].set_title("Imaginary")
            axes.legend(loc="upper right", fontsize="small")
            fig.suptitle("%s\n(Frequency%s)" % (os.path.basename(self.flname), key))

            fpdf.savefig(fig)
            pyplot.close(fig)

        fpdf.close()
        
        # Raise Warning if any NaNs are found
            
        nbad = len(number_nan.keys())
        if (nbad > 0):
            raise errors_base.NaNWarning("%i images had NaN's: %s" % (nbad, number_nan))

    def check_time(self):

        time = self.SWATHS["zeroDopplerTime"][...]
        spacing = self.SWATHS["zeroDopplerTimeSpacing"][...]
        start_time = self.IDENTIFICATION["zeroDopplerStartTime"][...]
        end_time = self.IDENTIFICATION["zeroDopplerEndTime"][...]

        start_time = str(start_time, "utf-8").rstrip("\x00").split(".")[0]
        end_time = str(end_time, "utf-8").rstrip("\x00").split(".")[0]
    
        print("Times %s to %s" % (start_time, end_time))
        try:
            time1 = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
            time2 = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
            assert(time2 < time1)
            assert( (time1.year >= 2000) and (time1.year < 2100) )
            assert( (time2.year >= 2000) and (time2.year < 2100) )
        except (AssertionError, ValueError):
            raise errors_derived.IdentificationFatal("Invalid Times of %s and %s" % (start_time, end_time))

        try:
            utility.check_spacing(time, spacing, "Time", errors_derived.TimeSpacingWarning, \
                                  errors_derived.TimeSpacingFatal)
        except (errors_base.WarningError, errors_base.FatalError):
            pass

    def check_frequencies(self):

        nfrequencies = len(self.frequencies)
        if (nfrequencies == 2):
            acquired_freq = {}
            for f in self.frequencies:
                acquired_freq[f] = self.FREQUENCIES[f]["processedCenterFrequency"][...]

            try:
                assert(acquired_freq["A"] < acquired_freq["B"])
            except AssertionError:
                raise errors_derived.FrequencyOrderFatal("Frequency A=%f not less than Frequency B=%f" \
                                                         % (acquired_freq["A"], acquired_freq["B"]))

    def check_slant_range(self):

        for f in self.frequencies:
            slant_path = self.FREQUENCIES[f]["slantRange"]
            spacing = self.FREQUENCIES[f]["slantRangeSpacing"]

            try:
                utility.check_spacing(slant_path[...], spacing[...], "%sSlantPath" % f, \
                                      errors_derived.SlantSpacingWarning, \
                                      errors_derived.SlantSpacingFatal)
            except (errors_base.WarningError, errors_base.FatalError):
                pass

        
        for key in self.images.keys():
            (f, p) = key.split("_")
            
            nslant0 = self.FREQUENCIES[f]["slantRange"].shape[0]
            nslant = self.images[key].shape[1]

            try:
                assert(nslant == nslant0)
            except AssertionError:
                raise errors_derived.ArraySizeFatal("Dataset %s has slantpath size of %i instead of %i" \
                                                    % (key, nslant, nslant0))


    def check_subswaths(self):
        
       for f in self.frequencies:
           nsubswath = self.FREQUENCIES[f]["numberOfSubSwaths"][...]
           try:
               assert( (nsubswath >= 0) and (nsubswath <= params.NSUBSWATHS) )
           except AssertionError:
               raise errors_derived.NumSubswathFatal("Frequency%s had invalid number of subswaths: %i" \
                                                     % (f, nsubswath))

           nslantrange = self.FREQUENCIES[f]["slantRange"].size
           
           for isub in range(0, nsubswath):
               try:
                   sub_bounds = self.FREQUENCIES[f]["validSamplesSubSwath%i" % (isub+1)][...]
               except KeyError:
                   raise errors_derived.MissingSubswathFatal("Frequency%s had missing SubSwath%i bounds" \
                                                             % (f, isub))

               try:
                   assert(np.all(sub_bounds[:, 0] < sub_bounds[:, 1]))
                   assert(np.all(sub_bounds[:, 0] >= 0))
                   assert(np.all(sub_bounds[:, 1] <= nslantrange))
               except AssertionError:
                   raise errors_derived.BoundsSubSwathFatal("Frequency%s with nSlantRange %i had invalid SubSwath bounds: %s" \
                                                            % (f, nslantrange, sub_bounds))
               

               

