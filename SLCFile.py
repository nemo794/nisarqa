import errors_base
import errors_derived
import params
from SLCImage import SLCImage
import utility

import h5py
from matplotlib import cm, pyplot, ticker
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

    def __init__(self, flname, mode, time_step=1, range_step=1):
        self.flname = os.path.basename(flname)
        h5py.File.__init__(self, flname, mode)

        print("Opening file %s" % flname)
        self.SWATHS = self["/science/LSAR/SLC/swaths"]
        self.METADATA = self["/science/LSAR/SLC/metadata"]
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
                key = "%s %s" % (f, p)
                try:
                    self.images[key] = SLCImage(f, p, self.FREQUENCIES[f]["processedCenterFrequency"][...], \
                                                self.FREQUENCIES[f]["slantRangeSpacing"][...])
                    self.images[key].read(self.FREQUENCIES, time_step=time_step, range_step=range_step)
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

        print("start_time %s, end_time %s" % (start_time, end_time))
        
        try:
            assert(str(end_time) > str(start_time))
        except AssertionError as e:
            error_string += ["Start Time %s not less than End Time %s" % (start_time, end_time)]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            assert(orbit > 0)
        except AssertionError as e:
            error_string += ["Invalid Orbit Number: %i" % orbit]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            print("Track type %s shape %s" % (type(track), track.shape))
            dummy = (track >= 0)
        except TypeError:
            track = int(track)

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

    def check_images(self, fpdf, fhdf):

        min_value = np.array([np.finfo(np.float64).max, np.finfo(np.float64).max])
        max_value = np.array([np.finfo(np.float64).min, np.finfo(np.float64).min])
        nan_warning = []
        nan_fatal = []

        # Get min/max value of all images and also check for NaNs.
        
        for key in self.images.keys():
            ximg = self.images[key]
            ximg.check_for_nan()
            if (ximg.num_nan > 0):
                if (not ximg.empty):
                    nan_warning.append(key)
                else:
                    nan_fatal.append(key)
 
        # Generate histograms (real and imaginary components separately) and plot them

        xmax = np.maximum(np.fabs(min_value), np.fabs(max_value)).max()
        for i in range(1, 100):
            if (10**i > xmax):
                break

        if (xmax < 10**i/2.0):
            bounds = 10**i/2
        else:
            bounds = 10**i

        sums_linear = {}
        sums_power = {}
            
        for key in self.images.keys():

            # Histogram with huge number of bins to find out where the bounds of the real
            # distribution lie and then rehistogram with more sensible bounds.  Do this for
            # individual real and complex components and then the difference between the two.

            (f, p) = key.split()
            ximg = self.images[key]
            ximg.calc()

            # Use same bounds for plotting all linear images of a given type

            #bounds_linear = [np.finfo(np.float32).max, np.finfo(np.float32).min]
            #bounds_power = [np.finfo(np.float32).max, np.finfo(np.float32).min]
            if (f not in sums_linear.keys()):
                sums_linear[f] = np.zeros(ximg.real.shape, dtype=np.float32)
                sums_power[f] = np.zeros(ximg.power.shape, dtype=np.float32)

            sums_linear[f] += ximg.real
            sums_linear[f] += ximg.imag
            sums_power[f] += ximg.power

        for (i, f) in enumerate(sums_linear.keys()):

            (counts, edges) = np.histogram(sums_linear[f], bins=1000)
            if (i == 0):
                counts_linear = np.copy(counts)
                edges_linear = np.copy(edges)
            else:
                counts_linear += counts

            (counts, edges) = np.histogram(sums_power[f], bins=1000)
            if (i == 0):
                counts_power = np.copy(counts)
                edges_power = np.copy(edges)
            else:
                counts_power += counts

        bounds_linear = utility.hist_bounds(counts_linear, edges_linear)
                
        # Generate figures
        
        for key in self.images.keys():

            ximg = self.images[key]
            fig = ximg.plot4a("%s\n(Frequency%s)" % (self.flname, key), (-1.0*bounds_linear, bounds_linear), \
                              (-100.0, 0))
            fpdf.savefig(fig)

        # Plot and summarize polarization-differences

        polarizations_all = ("HH", "VV", "HV", "VH")
        for f in self.frequencies:
            (fig, axes) = pyplot.subplots(nrows=4, ncols=4, sharex=False, sharey=False, \
                                          constrained_layout=True)

            diff_plots = np.zeros((len(polarizations_all), len(polarizations_all)), dtype=np.bool)
            for (ip1, p1) in enumerate(polarizations_all):
                for (ip2, p2) in enumerate(polarizations_all):
                    if (p1 in self.polarizations[f]) and (p2 in self.polarizations[f]) and \
                       (p1 != p2) and (not diff_plots[ip1, ip2]):

                        diff_plots[ip1, ip2] = True
                        diff_plots[ip2, ip1] = True
                                     
                        ref_img = self.images["%s %s" % (f, p1)]
                        cmp_img = self.images["%s %s" % (f, p2)]
                        xdata = ref_img.xdata*cmp_img.xdata.conj()

                        key_new = "%s %s-%s" % (f, p1, p2)
                        self.images[key_new] = SLCImage(f, "%s-%s" % (p1, p2), \
                                                        self.FREQUENCIES[f]["processedCenterFrequency"][...], \
                                                        self.FREQUENCIES[f]["slantRangeSpacing"][...])
                        self.images[key_new].initialize(xdata, ref_img.nan_mask | cmp_img.nan_mask)
                        self.images[key_new].calc()
                        self.images[key_new].plot4b(axes[ip1, ip2], "%s - %s" % (p1, p2))
                        
            #for (i, a) in enumerate(axes.flatten()):
            #    if (i < 3):
            #        key = keys_differences[i]
            #        self.images[key].plot4b(a, key.split()[-1])


            axes[0, 0].set_xlabel("SLC Phase\n(degrees)")
            axes[0, 0].set_ylabel("Number\nof Counts")
            fig.suptitle("%s\nFrequency %s" % (self.flname, f))
            fpdf.savefig(fig)
            pyplot.close(fig)


        # Plot power spectrum

        nplots = 0
        for f in self.frequencies:
            (fig, axis) = pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, \
                                          constrained_layout=True)
            for p in self.polarizations[f]:
                nplots += 1
                key = "%s %s" % (f, p)
                ximg = self.images[key]
                xpower = 20.0*np.log10(ximg.avg_power)
                axis.plot(ximg.fft_space, xpower, label=key)
                #print("Image %s, has Frequency %f to %f with acquired frequency %s and shape %s" \
                #      % (key, ximg.fft_space.min(), ximg.fft_space.max(), 1.0/(ximg.tspacing), ximg.shape))

            axis.legend(loc="upper right", fontsize="small")
            axis.set_xlabel("Frequency (MHz)")
            axis.set_ylabel("Power Spectrum (dB)")
            #axis.set_xlim(left=-100.0, right=100.0)
            fig.suptitle("Power Spectrum for Frequency %s" % f)
            fpdf.savefig(fig)
                
        # Write histogram summaries to an hdf5 file
            
        print("File %s mode %s" % (fhdf.filename, fhdf.mode))
        fname_in = os.path.basename(self.filename)
        extension = fname_in.split(".")[-1]
        group1 = fhdf.create_group("%s/LSAR/ImageAttributes" % fname_in.replace(".%s" % extension, ""))
        for key in self.images.keys():
            ximg = self.images[key]
            group = group1.create_group(key)
            for (name, data) in zip( ("MeanPower", "SDevPower", "MeanPhase", "SDevPhase"), \
                                     ("mean_power", "sdev_power", "mean_phase", "sdev_phase") ):
                xdata = getattr(ximg, data)
                dset = group.create_dataset(name, (), dtype='f4')
                dset.write_direct(np.array(xdata).astype(np.float32))

            dset = group.create_dataset("FFT Spacing", ximg.fft_space.shape, dtype='f4')
            dset.write_direct(ximg.fft_space.astype(np.float32))
            dset = group.create_dataset("Average Power", ximg.fft_space.shape, dtype='f4')
            dset.write_direct(20.0*np.log10(ximg.avg_power).astype(np.float32))

        #fhdf.close()
                    
        # Raise Warning if any NaNs are found and Fatal if an image is entirely NaN
            
        try:
            assert(len(nan_warning) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = "%i images had NaN's:" % len(nan_warning)
            for key in nan_warnings:
                error_string += " (%s, %i=%f%%)" % (key, self.images[key].num_nan, self.images[key].perc_nan)
            raise errors_derived.NaNWarning(self.flname, self.start_time, traceback_string, error_string)

        try:
            assert(len(nan_fatal) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = "%i images were empty: %s" % (len(nan_fatal), nan_fatal)
            raise errors_derived.NaNFatal(self.flname, self.start_time, traceback_string, error_string)

    def check_time(self):

        time = self.SWATHS["zeroDopplerTime"][...]
        spacing = self.SWATHS["zeroDopplerTimeSpacing"][...]
        start_time = self.IDENTIFICATION["zeroDopplerStartTime"][...]
        end_time = self.IDENTIFICATION["zeroDopplerEndTime"][...]

        start_time = str(start_time, "utf-8").rstrip("\x00").split(".")[0]
        end_time = str(end_time, "utf-8").rstrip("\x00").split(".")[0]
        print("time %f to %f" % (time.min(), time.max()))
    
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
            utility.check_spacing(self.flname, self.start_time, time, spacing, "zeroDopplerTime", \
                                  errors_derived.TimeSpacingWarning, errors_derived.TimeSpacingFatal)
        except (errors_base.WarningError, errors_base.FatalError):
            pass

        time = self.METADATA["orbit/time"]
        try:
            utility.check_spacing(self.flname, time[0], time, time[1] - time[0], "orbitTime", \
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
                assert(acquired_freq["A"] < acquired_freq["B"])
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
            #print("Looking at key %s" % key)
            (f, p) = key.split()
            
            nslant0 = self.FREQUENCIES[f]["slantRange"].shape[0]
            nslant = self.images[key].xdata.shape[1]

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
               

               

