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

    def __init__(self, flname, mode):
        self.flname = os.path.basename(flname)
        h5py.File.__init__(self, flname, mode)
        print("Opening file %s" % flname)

        self.bands = []
        self.SWATHS = {}
        self.METADATA = {}
        self.IDENTIFICATION = {}

        self.images = {}
        self.start_time = {}
        self.plotted_slant = False

    def get_bands(self):
        
        for band in ("LSAR", "SSAR"):
            try:
                xband = self["/science/%s" % band]
            except KeyError:
                print("%s not present" % band)
                pass
            else:
                print("Found band %s" % band)
                try:
                    self.bands.append(band)
                    self.SWATHS[band] = self["/science/%s/SLC/swaths" % band]
                    self.METADATA[band] = self["/science/%s/SLC/metadata" % band]
                    self.IDENTIFICATION[band] = self["/science/%s/identification/" % band]
                except KeyError as e:
                    traceback_string = [utility.get_traceback(e, KeyError)]
                    raise errors_derived.IdentificationFatal(self.flname, "0000-00-00T00:00:00.000000", \
                                                             ["File missing swath, metadata or identification data for %s." % band])


        print("Found bands: %s" % self.bands)
                

    def get_freq_pol(self):

        self.FREQUENCIES = {}
        self.polarizations = {}
        
        for b in self.bands:

            # Find list of frequencies by directly querying dataset

            self.FREQUENCIES[b] = {}
            self.polarizations[b] = {}
            for f in ("A", "B"):
                try:
                    f2 = self["/science/%s/SLC/swaths/frequency%s" % (b, f)]
                except KeyError:
                    pass
                else:
                    self.FREQUENCIES[b][f] = f2
                    self.polarizations[b][f] = []
                    for p in ("HH", "VV", "HV", "VH"):
                        try:
                            p2 = self.FREQUENCIES[b][f][p]
                        except KeyError:
                            pass
                        else:
                            self.polarizations[b][f].append(p)

            print("%s has Frequencies %s and polarizations %s" % (b, self.FREQUENCIES[b].keys(), self.polarizations[b]))

    def check_freq_pol(self):

        # Check for correct frequencies and polarizations

        error_string = []
        traceback_string = []
         
        for b in self.bands:

            self.start_time[b] = bytes(self.IDENTIFICATION[b].get("zeroDopplerStartTime")[...])[0:params.TIMELEN]

            try:
                frequencies = [f.decode() for f in self.IDENTIFICATION[b].get("listOfFrequencies")[...]]
                assert(frequencies in params.FREQUENCIES)
            except (AssertionError, KeyError, TypeError, UnicodeDecodeError) as e:
                traceback_string += [utility.get_traceback(e, AssertionError)]
                error_string += ["%s Band has invalid frequency list" % b]

            print("Finished logging Frequency List error")
            for f in self.FREQUENCIES[b].keys():
                print("Looking at %s Frequency%s" % (b, f))
                try:
                    polarization_ok = False
                    polarizations_found = [p.decode() for p in self.FREQUENCIES[b][f].get("listOfPolarizations")[...]]
                    for plist in params.POLARIZATIONS:
                        if (set(polarizations_found) == set(plist)):
                            polarization_ok = True
                            break
                    assert(polarization_ok)
                except (AssertionError, KeyError, UnicodeDecodeError) as e:
                    traceback_string += [utility.get_traceback(e, AssertionError)]
                    error_string += ["%s Frequency%s has invalid polarization list" % (b, f)]

        assert(len(traceback_string) == len(error_string))
        try:
            assert(len(error_string) == 0)
        except AssertionError as e:
            #print("error_string: %s" % error_string)
            #print("traceback_string: %s" % traceback_string)
            raise errors_derived.IdentificationFatal(self.flname, self.start_time[b], traceback_string, error_string)

    def create_images(self, time_step=1, range_step=1):

        missing_images = []
        print("Polarizations: %s" % self.polarizations)
        
        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                for p in self.polarizations[b][f]:
                    p2 = p
                    #if (p == "VV"):
                    #    p2 = "VH"
                    #elif (p == "VH"):
                    #    p2 = "VV"
                    #else:
                    #    p2 = p
                    key = "%s %s %s" % (b, f, p)

                    try:
                        print("Creating image %s %s %s" % (b, f, p2))
                        self.images[key] = SLCImage(b, f, p2, self.FREQUENCIES[b][f]["processedCenterFrequency"][...], \
                                                    self.FREQUENCIES[b][f]["slantRangeSpacing"][...])
                        self.images[key].read(self.FREQUENCIES[b], time_step=time_step, range_step=range_step)
                        print("Created image with key %s" % key)
                    except KeyError:
                        missing_images.append(key)

                try:
                    assert(len(missing_images) == 0)
                except AssertionError as e:
                    traceback_string = [utility.get_traceback(e, AssertionError)]
                    raise errors_derived.ArrayMissingFatal(self.flname, self.start_time[b], traceback_string, \
                                                           ["Missing %i images: %s" % (len(missing_images), missing_images)])

            
    def check_identification(self):

        error_string = []
        traceback_string = []

        identifications = {}
        for dname in ("absoluteOrbitNumber", "trackNumber", "frameNumber", "cycleNumber", "lookDirection", \
                      "orbitPassDirection", "productType", "zeroDopplerStartTime", "zeroDopplerEndTime"):
            for b in self.bands:
                identifications[dname] = []
                try:
                    identifications[dname].append(self.IDENTIFICATION[b].get(dname)[...])
                    #print("%s: Read %s of type %s and size %s and values %s" \
                    #      % (b, dname, type(identifications[dname][0]), identifications[dname][0]))
                except KeyError:
                    error_string += ["%s missing dataset %s" % (b, dname)]
                except TypeError as e:
                    if (dname == "cycleNumber"):
                        identifications[dname].append(-9999)

            try:
                #print("%s: values %s %s" % (dname, type(identifications[dname]), identifications[dname]))
                assert( (len(self.bands) == 1) or (identifications[dname][0] == identifications[dname][1]) )
            except AssertionError as e:
                traceback_string.append(utility.get_traceback(e, AssertionError))
                error_string += ["Values of %s differ between bands" % dname]
                
            #orbit = self.IDENTIFICATION.get("absoluteOrbitNumber")[...]
            #track = self.IDENTIFICATION.get("trackNumber")[...]
            #frame = self.IDENTIFICATION.get("frameNumber")[...]
            #try:
            #    cycle = self.IDENTIFICATION.get("cycleNumber")[...]
            #except TypeError:
            #    cycle = -9999
            #    lookdir = self.IDENTIFICATION.get("lookDirection")[...]
            #    passdir = self.IDENTIFICATION.get("orbitPassDirection")[...]
            #    ptype = self.IDENTIFICATION.get("productType")[...]
            #    start_time = self.IDENTIFICATION.get("zeroDopplerStartTime")[...]
            #    end_time = self.IDENTIFICATION.get("zeroDopplerEndTime")[...]

        # Verify that all identification parameters are correct
            
        try:
            start_time = str(identifications["zeroDopplerStartTime"][0])
            end_time = str(identifications["zeroDopplerEndTime"][0])
            assert(end_time > start_time)
        except AssertionError as e:
            error_string += ["Start Time %s not less than End Time %s" % (start_time, end_time)]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            orbit = int(identifications["absoluteOrbitNumber"][0])
            assert(orbit > 0)
        except AssertionError as e:
            error_string += ["Invalid Orbit Number: %i" % orbit]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            track = int(identifications["trackNumber"][0])
            dummy = (track >= 0)
        except TypeError:
            track = int(track)

        try:
            assert( (track > 0) and (track <= params.NTRACKS) )
        except AssertionError as e:
            error_string += ["Invalid Track Number: %i" % track]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            frame = int(identifications["frameNumber"][0])
            assert(frame > 0)
        except AssertionError as e:
            error_string += ["Invalid Frame Number: %i" % frame]
            traceback_string.append(utility.get_traceback(e, AssertionError))
            
        try:
            cycle = int(identifications["cycleNumber"][0])
            assert(cycle > 0)
        except AssertionError as e:
            error_string += ["Invalid Cycle Number: %i" % cycle]
            traceback_string.append(utility.get_traceback(e, AssertionError))

        try:
            ptype = str(identifications["productType"][0])
            assert(str(ptype) in ("RRST'", "RRSD", "RSLC", "RMLC", \
                                  "RCOV", "RIFG", "RUNW", "GUNW", \
                                  "CGOV", "GSLC"))
        except AssertionError as e:
            error_string += ["Invalid Product Type"]
            traceback_string.append(utility.get_traceback(e, AssertionError))

        try:
            lookdir = str(identifications["lookDirection"][0])
            assert(str(lookdir).lower() in ("b'left'", "b'right'"))
        except AssertionError as e:
            error_string += ["Invalid Look Number:"]
            traceback_string.append(utility.get_traceback(e, AssertionError))

        # raise errors if needed
            
        assert(len(error_string) == len(traceback_string))
            
        try:
            assert(len(error_string) == 0)
        except AssertionError as e:
            #print("error_string: %s" % error_string)
            #print("traceback_string: %s" % traceback_string)
            raise errors_derived.IdentificationFatal(self.flname, self.start_time[b], traceback_string, error_string)

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

        print("Found %i images: %s" % (len(self.images.keys()), self.images.keys()))
        for key in self.images.keys():

            # Histogram with huge number of bins to find out where the bounds of the real
            # distribution lie and then rehistogram with more sensible bounds.  Do this for
            # individual real and complex components.

            (b, f, p) = key.split()
            ximg = self.images[key]
            ximg.calc()

            # Use same bounds for plotting all linear images of a given type

            #bounds_linear = [np.finfo(np.float32).max, np.finfo(np.float32).min]
            #bounds_power = [np.finfo(np.float32).max, np.finfo(np.float32).min]
            if (key not in sums_linear.keys()):
                sums_linear[key] = np.zeros(ximg.real.shape, dtype=np.float32)
                sums_power[key] = np.zeros(ximg.power.shape, dtype=np.float32)

            sums_linear[key] += ximg.real
            sums_linear[key] += ximg.imag
            sums_power[key] += ximg.power

        for (i, key) in enumerate(sums_linear.keys()):

            (counts, edges) = np.histogram(sums_linear[key], bins=1000)
            if (i == 0):
                counts_linear = np.copy(counts)
                edges_linear = np.copy(edges)
            else:
                counts_linear += counts

            (counts, edges) = np.histogram(sums_power[key], bins=1000)
            if (i == 0):
                counts_power = np.copy(counts)
                edges_power = np.copy(edges)
            else:
                counts_power += counts

        bounds_linear = utility.hist_bounds(counts_linear, edges_linear)
                
        # Generate figures
        
        for key in self.images.keys():

            (b, f, p) = key.split()
            ximg = self.images[key]
            fig = ximg.plot4a("%s\n(%s Frequency%s %s)" % (self.flname, b, f, p), \
                              (-1.0*bounds_linear, bounds_linear), (-100.0, 100.0))
            fpdf.savefig(fig)

        # Plot and summarize polarization-differences

        polarizations_all = ("HH", "VV", "HV", "VH")
        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                (fig, axes) = pyplot.subplots(nrows=4, ncols=4, sharex=False, sharey=False, \
                                              constrained_layout=True)

                diff_plots = np.zeros((len(polarizations_all), len(polarizations_all)), dtype=np.bool)
                for (ip1, p1) in enumerate(polarizations_all):
                    for (ip2, p2) in enumerate(polarizations_all):
                        if (p1 in self.polarizations[b][f]) and (p2 in self.polarizations[b][f]) and \
                           (p1 != p2) and (not diff_plots[ip1, ip2]):

                            diff_plots[ip1, ip2] = True
                            diff_plots[ip2, ip1] = True
                                     
                            ref_img = self.images["%s %s %s" % (b, f, p1)]
                            cmp_img = self.images["%s %s %s" % (b, f, p2)]
                            xdata = ref_img.xdata*cmp_img.xdata.conj()

                            key_new = "%s %s %s-%s" % (b, f, p1, p2)
                            self.images[key_new] = SLCImage(b, f, "%s-%s" % (p1, p2), \
                                                            self.FREQUENCIES[b][f]["processedCenterFrequency"][...], \
                                                            self.FREQUENCIES[b][f]["slantRangeSpacing"][...])
                            self.images[key_new].initialize(xdata, ref_img.nan_mask | cmp_img.nan_mask)
                            self.images[key_new].calc()
                            self.images[key_new].plot4b(axes[ip1, ip2], "%s - %s" % (p1, p2))
                        
                #for (i, a) in enumerate(axes.flatten()):
                #    if (i < 3):
                #        key = keys_differences[i]
                #        self.images[key].plot4b(a, key.split()[-1])


                axes[0, 0].set_xlabel("SLC Phase\n(degrees)")
                axes[0, 0].set_ylabel("Number\nof Counts")
                fig.suptitle("%s\n%s Frequency %s Phase Histograms" % (self.flname, b, f))
                fpdf.savefig(fig)
                pyplot.close(fig)


        # Plot power spectrum

        nplots = 0
        groups = {}
        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                (fig, axis) = pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, \
                                              constrained_layout=True)
                print("List of polarizations for %s %s = %s" % (b, f, self.polarizations[b][f]))
                for p in self.polarizations[b][f]:
                    nplots += 1
                    key = "%s %s %s" % (b, f, p)
                    ximg = self.images[key]
                    xpower = 20.0*np.log10(ximg.avg_power)
                    axis.plot(ximg.fft_space, xpower, label=key)
                    print("Image %s, has Frequency %f to %f with acquired frequency %s and shape %s and power %f to %f" \
                          % (key, ximg.fft_space.min(), ximg.fft_space.max(), 1.0/(ximg.tspacing), ximg.shape, \
                             xpower.min(), xpower.max()))

                #raise RuntimeError("Stopping")
                    
                axis.legend(loc="upper right", fontsize="small")
                axis.set_xlabel("Frequency (MHz)")
                axis.set_ylabel("Power Spectrum (dB)")
                #axis.set_ylim(bottom=40.0, top=100.0)
                #axis.set_xlim(left=-100.0, right=100.0)
                fig.suptitle("%s\nPower Spectrum for %s Frequency %s" % (self.flname, b, f))
                fpdf.savefig(fig)

                if (self.plotted_slant):
                    for f in self.FREQUENCIES[b].keys():
                        print("Saving power spectrum plot")
                        fpdf.savefig(self.figures_slant[f])
                
            # Write histogram summaries to an hdf5 file
            
            print("File %s mode %s" % (fhdf.filename, fhdf.mode))
            fname_in = os.path.basename(self.filename)
            extension = fname_in.split(".")[-1]
            groups[b] = fhdf.create_group("%s/%s/ImageAttributes" % (fname_in.replace(".%s" % extension, ""), b))

        for key in self.images.keys():
            (b, f, p) = key.split()
            ximg = self.images[key]
            group2 = groups[b].create_group(key)
            for (name, data) in zip( ("MeanPower", "SDevPower", "MeanPhase", "SDevPhase"), \
                                     ("mean_power", "sdev_power", "mean_phase", "sdev_phase") ):
                xdata = getattr(ximg, data)
                dset = group2.create_dataset(name, (), dtype='f4')
                dset.write_direct(np.array(xdata).astype(np.float32))

            dset = group2.create_dataset("FFT Spacing", ximg.fft_space.shape, dtype='f4')
            dset.write_direct(ximg.fft_space.astype(np.float32))
            dset = group2.create_dataset("Average Power", ximg.fft_space.shape, dtype='f4')
            dset.write_direct(20.0*np.log10(ximg.avg_power).astype(np.float32))

            # Raise Warning if any NaNs are found and Fatal if an image is entirely NaN
            
            try:
                assert(len(nan_warning) == 0)
            except AssertionError as e:
                traceback_string = [utility.get_traceback(e, AssertionError)]
                error_string = "%i images had NaN's:" % len(nan_warning)
                for key in nan_warnings:
                    error_string += " (%s, %i=%f%%)" % (key, self.images[key].num_nan, self.images[key].perc_nan)
                    raise errors_derived.NaNWarning(self.flname, self.start_time[b], traceback_string, error_string)

            try:
                assert(len(nan_fatal) == 0)
            except AssertionError as e:
                traceback_string = [utility.get_traceback(e, AssertionError)]
                error_string = "%i images were empty: %s" % (len(nan_fatal), nan_fatal)
                raise errors_derived.NaNFatal(self.flname, self.start_time[b], traceback_string, error_string)

    def check_time(self):

        for b in self.bands:
        
            time = self.SWATHS[b]["zeroDopplerTime"][...]
            spacing = self.SWATHS[b]["zeroDopplerTimeSpacing"][...]
            start_time = self.IDENTIFICATION[b]["zeroDopplerStartTime"][...]
            end_time = self.IDENTIFICATION[b]["zeroDopplerEndTime"][...]

            try:
                start_time = bytes(start_time).split(b".")[0].decode("utf-8")
                end_time = bytes(end_time).split(b".")[0].decode("utf-8")
            except UnicodeDecodeError as e:
                traceback_string = [utility.get_traceback(e, UnicodeDecodeError)]
                raise errors_derived.IdentificationFatal(self.flname, self.start_time[b], traceback_string, \
                                                         ["%s Start/End Times could not be read." % b])
            else:
                print("time %f to %f" % (time.min(), time.max()))
                try:
                    time1 = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
                    time2 = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
                    assert(time2 > time1)
                    assert( (time1.year >= 2000) and (time1.year < 2100) )
                    assert( (time2.year >= 2000) and (time2.year < 2100) )
                except (AssertionError, ValueError) as e:
                    traceback_string = [utility.get_traceback(e, AssertionError)]
                    raise errors_derived.IdentificationFatal(self.flname, self.start_time[b], traceback_string, \
                                                             ["%s Invalid Start and End Times" % b])
                                                              #% (b, start_time[0:10].strip(), end_time[0:10].strip())])

            try:
                utility.check_spacing(self.flname, self.start_time[b], time, spacing, "%s zeroDopplerTime" % b, \
                                      errors_derived.TimeSpacingWarning, errors_derived.TimeSpacingFatal)
            except (errors_base.WarningError, errors_base.FatalError):
                pass

            time = self.METADATA[b]["orbit/time"]
            try:
                utility.check_spacing(self.flname, time[0], time, time[1] - time[0], "%s orbitTime" % b, \
                                      errors_derived.TimeSpacingWarning, errors_derived.TimeSpacingFatal)
            except (errors_base.WarningError, errors_base.FatalError):
                pass
        
    def check_frequencies(self):

        for b in self.bands:
            nfrequencies = len(self.FREQUENCIES[b])
            if (nfrequencies == 2):
                acquired_freq = {}
                for f in list(self.FREQUENCIES[b].keys()):
                    acquired_freq[f] = self.FREQUENCIES[b][f]["processedCenterFrequency"][...]

                try:
                    assert(acquired_freq["A"] < acquired_freq["B"])
                except AssertionError as e:
                    traceback_string = [utility.get_traceback(e, AssertionError)]
                    raise errors_derived.FrequencyOrderFatal(self.flname, self.start_time[b], traceback_string, \
                                                             ["Frequency A=%f not less than Frequency B=%f" \
                                                              % (acquired_freq["A"], acquired_freq["B"])])

    def check_slant_range(self):

        self.figures_slant = {}

        for key in self.images.keys():
            #print("Looking at key %s" % key)
            (b, f, p) = key.split()

            ximg = self.images[key]
            nslant = self.FREQUENCIES[b][f]["slantRange"].shape[0]
            ntime = self.SWATHS[b]["zeroDopplerTime"].shape[0]

            try:
                assert(ximg.shape == (ntime, nslant))
            except AssertionError as e:
                traceback_string = [utility.get_traceback(e, AssertionError)]
                raise errors_derived.ArraySizeFatal(self.flname, self.start_time[b], traceback_string, \
                                                    ["Dataset %s has shape %s, expected (%i, %i)" \
                                                     % (key, ximg.shape, ntime, nslant)])

        for b in self.bands:
            for f in list(self.FREQUENCIES[b].keys()):
                slant_path = self.FREQUENCIES[b][f]["slantRange"]
                spacing = self.FREQUENCIES[b][f]["slantRangeSpacing"]

            try:
                utility.check_spacing(self.flname, self.start_time[b], slant_path[...], spacing[...], \
                                      "%s %s SlantPath" % (b, f), errors_derived.SlantSpacingWarning, \
                                      errors_derived.SlantSpacingFatal)
            except (errors_base.WarningError, errors_base.FatalError) as e:
                xslant = slant_path[...]
                self.plotted_slant = True
                (self.figures_slant[f], axis) = pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, \
                                                                constrained_layout=True)
                axis.plot(np.arange(0, xslant.size), xslant, label="%s Frequency%s SlantPath" % (b, f))
                self.figures_slant[f].suptitle("%s Frequency%s slantRange" % (b, f))

    def check_subswaths(self):

        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                nsubswath = self.FREQUENCIES[b][f]["numberOfSubSwaths"][...]
                print("%s Frequency%s has %s subswaths" % (b, f, nsubswath))
                try:
                    assert( (nsubswath >= 0) and (nsubswath <= params.NSUBSWATHS) )
                except AssertionError as e:
                    traceback_string = [utility.get_traceback(e, AssertionError)]
                    raise errors_derived.NumSubswathFatal(self.flname, self.start_time[b], traceback_string, \
                                                          ["%s Frequency%s had invalid number of subswaths: %i" \
                                                           % (b, f, nsubswath)])

                nslantrange = self.FREQUENCIES[b][f]["slantRange"].size

                for isub in range(0, int(nsubswath)):
                    try:
                        sub_bounds = self.FREQUENCIES[b][f]["validSamplesSubSwath%i" % (isub+1)][...]
                    except KeyError as e:
                        traceback_string = [utility.get_traceback(e, KeyError)]
                        raise errors_derived.MissingSubswathFatal(self.flname, self.start_time[b], traceback_string, \
                                                                  ["%s Frequency%s had missing SubSwath%i bounds" \
                                                                   % (b, f, isub)])

                    try:
                        assert(np.all(sub_bounds[:, 0] < sub_bounds[:, 1]))
                        assert(np.all(sub_bounds[:, 0] >= 0))
                        assert(np.all(sub_bounds[:, 1] <= nslantrange))
                    except AssertionError as e:
                        traceback_string = [utility.get_traceback(e, KeyError)]
                        raise errors_derived.BoundsSubSwathFatal(self.flname, self.start_time[b], traceback_string, \
                                                                 ["%s Frequency%s with nSlantRange %i had invalid SubSwath bounds: %s" \
                                                                  % (b, f, nslantrange, sub_bounds)])
               

               

