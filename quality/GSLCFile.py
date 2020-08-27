from quality.HdfGroup import HdfGroup
from quality.NISARFile import NISARFile
from quality.SLCImage import SLCImage
from quality import errors_base, errors_derived, logging_base, params, utility

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

class GSLCFile(NISARFile):

    def __init__(self, flname, logger, xml_tree=None, mode="r"):
        NISARFile.__init__(self, flname, logger, xml_tree=xml_tree, mode=mode)

        self.bands = []
        self.SWATHS = {}
        self.METADATA = {}
        self.IDENTIFICATION = {}

        self.images = {}
        self.start_time = {}
        self.plotted_slant = False

        self.ftype_list = ("SLC", "RSLC")
        self.product_type = "SLC"
        self.polarizations_possible = ("HH", "HV", "VH", "VV", "RH", "RV")
        self.polarization_list = params.GSLC_POLARIZATION_LIST
        self.polarization_groups = params.GSLC_POLARIZATION_GROUPS
        self.identification_list = params.SLC_ID_PARAMS
        self.frequency_checks = params.GSLC_FREQUENCY_NAMES
        self.swath_path = "/science/%s/GSLC/grids"
        self.metadata_path = "/science/%s/GSLC/metadata"
        self.identification_path = "/science/%s/identification/"

        self.get_start_time()

        self.logger.log_message(logging_base.LogFilterInfo, "Created file %s" % flname)

    def get_slant_range(self, band, frequency):

        slant_path = self.FREQUENCIES[band][frequency].get("slantRange")
        spacing = self.METADATA[band].get("/radarGrid/slantRangeSpacing")
            
        return slant_path, spacing
            
    def get_bands(self):

        errors_found = False
        
        for band in ("LSAR", "SSAR"):
            try:
                xband = self["/science/%s" % band]
            except KeyError:
                self.logger.log_message(logging_base.LogFilterInfo, "%s not present" % band)
                pass
            else:
                self.logger.log_message(logging_base.LogFilterInfo, "Found band %s" % band)
                self.bands.append(band)

                # Create HdfGroups.
                
                for (gname, xdict, dict_name) in zip( ("/science/%s/GSLC/grids", "/science/%s/GSLC/metadata", \
                                                       "/science/%s/identification"), \
                                                      (self.SWATHS, self.METADATA, self.IDENTIFICATION), \
                                                      ("%s Grid" % band, "%s Metadata" % band, \
                                                       "%s Identification" % band) ):
                    try:
                        gname2 = gname % band
                        group = self[gname2]
                    except KeyError as e:
                        errors_found = True
                        self.logger.log_message(logging_base.LogFilterError, \
                                                "%s does not exist" % gname2)
                    else:
                        xdict[band] = HdfGroup(self, dict_name, gname2, self.logger)

        return errors_found

    def create_images(self, time_step=1, range_step=1):

        missing_images = []
        missing_params = []
        traceback_string = []
        error_string = []

        freq = {}
        srange = {}
        
        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                try:
                    assert("centerFrequency" in self.FREQUENCIES[b][f].keys())
                    assert("slantRangeSpacing" in self.FREQUENCIES[b][f].keys())
                    freq["%s %s" % (b, f)] = self.FREQUENCIES[b][f].get("centerFrequency")
                    srange["%s %s" % (b, f)] = self.FREQUENCIES[b][f].get("slantRangeSpacing")
                except AssertionError:
                    missing_params += ["%s %s" % (b, f)]
                    error_string += ["%s %s cannot initialize GSLCImage due to missing frequency or spacing" % (b, f)]
 
        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                if ("%s %s" % (b, f) in missing_params):
                    continue

                for p in self.polarizations[b][f]:
                    key = "%s %s %s" % (b, f, p)
                    try:
                        self.images[key] = SLCImage(b, f, p, freq["%s %s" % (b, f)][...], \
                                                    srange["%s %s" % (b, f)][...])
                        self.images[key].read(self.FREQUENCIES[b], time_step=time_step, range_step=range_step)
                        self.logger.log_message(logging_base.LogFilterDebug, \
                                                "%s image %s %s %s has shape %s" \
                                                % (os.path.abspath(self.filename), b, f, p, \
                                                   self.images[key].xdata.shape))
                    except KeyError:
                        missing_images.append(key)
                        
        try:
            assert(len(missing_images) == 0)
            assert(len(missing_params) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = ""
            if (len(missing_images) > 0):
                self.logger.log_message(logging_base.LogFilterError, \
                                        "Missing %i images: %s" % (len(missing_images), missing_images))
            if (len(missing_params) > 0):
                self.logger.log_message(logging_base.LogFilterError, \
                                        "Could not initialize %i images: %s" \
                                        % (len(missing_params), missing_params))
            #raise errors_derived.ArrayMissingFatal(self.flname, self.start_time[b], traceback_string, \
            #                                       [error_string])
                

    def check_images(self, fpdf, fhdf):

        min_value = np.array([np.finfo(np.float64).max, np.finfo(np.float64).max])
        max_value = np.array([np.finfo(np.float64).min, np.finfo(np.float64).min])
        nan_warning = []
        nan_fatal = []

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

        self.logger.log_message(logging_base.LogFilterInfo, \
                                "Found %i images: %s" % (len(self.images.keys()), self.images.keys()))
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

            self.logger.log_message(logging_base.LogFilterInfo, "Looking at %i-th image: %s" % (i, key))
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
        self.logger.log_message(logging_base.LogFilterDebug, "bounds_linear %s" % bounds_linear)
                
        # Generate figures
        
        for key in self.images.keys():

            (b, f, p) = key.split()
            ximg = self.images[key]
            fig = ximg.plot4a("%s\n(%s Frequency%s %s GSLC Histograms)" % (self.flname, b, f, p), \
                              (-1.0*bounds_linear, bounds_linear), (-100.0, 100.0))
            fpdf.savefig(fig)

        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                (fig, axis) = pyplot.subplots(nrows=1, ncols=1, constrained_layout=True)
                for p in self.polarizations[b][f]:
                    key = "%s %s %s" % (b, f, p)
                    self.images[key].plot4a1(axis)
                axis.legend(loc="upper right")
                axis.set_xlabel("GSLC Power (dB)")
                axis.set_ylabel("Number of Counts")
                fig.suptitle("%s\n(%s Frequency %s Power Histograms)" % (self.flname, b, f))
                fpdf.savefig(fig)
                pyplot.close(fig)
                    
        # Plot and summarize polarization-differences

        polarizations_all = ("HH", "VV", "HV", "VH")
        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                if (len(self.polarizations[b][f]) <= 1):
                    continue
                
                (fig, axes) = pyplot.subplots(nrows=3, ncols=4, sharex=False, sharey=False, \
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
                                                            self.FREQUENCIES[b][f]["centerFrequency"][...], \
                                                            self.FREQUENCIES[b][f]["slantRangeSpacing"][...])
                            self.images[key_new].initialize(xdata, ref_img.nan_mask | cmp_img.nan_mask)
                            self.images[key_new].calc()
                            self.images[key_new].plot4b(axes[ip1, ip2], title="%s - %s" % (p1, p2))
                        
                axes[0, 0].set_xlabel("GSLC Phase\n(degrees)")
                axes[0, 0].set_ylabel("Number\nof Counts")
                fig.suptitle("%s\n%s Frequency %s Phase Histograms" % (self.flname, b, f))
                fpdf.savefig(fig)
                pyplot.close(fig)

                keys = ["%s %s %s" % (b, f, k) for k in ("HH-VV", "HV-VH")]
                keys = [k for k in keys if k in self.images.keys()]
                if (len(keys) > 0):
                    (fig, axis) = pyplot.subplots(nrows=1, ncols=1, constrained_layout=True)
                    for k in keys:
                        self.images[k].plot4b(axis, label=k.replace(b, "").replace(f, ""))
                    axis.legend(loc="upper right")
                    axis.set_xlabel("GSLC Phase (degrees)")
                    axis.set_ylabel("Number of Counts")
                    fig.suptitle("%s\n(%s Frequency %s Phase Histograms)" % (self.flname, b, f))
                    fpdf.savefig(fig)
                    pyplot.close(fig)

        # Plot power spectrum

        nplots = 0
        groups = {}
        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                (fig, axis) = pyplot.subplots(nrows=1, ncols=1, sharex=False, sharey=False, \
                                              constrained_layout=True)
                for p in self.polarizations[b][f]:
                    nplots += 1
                    key = "%s %s %s" % (b, f, p)
                    ximg = self.images[key]
                    xpower = 20.0*np.log10(ximg.avg_power)
                    axis.plot(ximg.fft_space, xpower, label=key)
                    
                axis.legend(loc="upper right", fontsize="small")
                axis.set_xlabel("Frequency (MHz)")
                axis.set_ylabel("Power Spectrum (dB)")
                #axis.set_ylim(bottom=40.0, top=100.0)
                #axis.set_xlim(left=-100.0, right=100.0)
                fig.suptitle("%s\nPower Spectrum for %s Frequency %s" % (self.flname, b, f))
                fpdf.savefig(fig)

                if (self.plotted_slant):
                    for f in self.FREQUENCIES[b].keys():
                        fpdf.savefig(self.figures_slant[f])
                
            # Write histogram summaries to an hdf5 file
            
            self.logger.log_message(logging_base.LogFilterInfo, \
                                    "Opening File %s in mode %s" % (fhdf.filename, fhdf.mode))
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

    def check_slant_range(self):

        self.figures_slant = {}
        size_error = []
        size_traceback = []

        # Check sizes for all images
        
        for key in self.images.keys():

            (b, f, p) = key.split()
            ximg = self.images[key]

            try:
                xdim = self.FREQUENCIES[b][f].get("xCoordinates")[...]
                ydim = self.FREQUENCIES[b][f].get("yCoordinates")[...]
                assert(ximg.shape == (ydim.size, xdim.size))
            except KeyError:
                continue
            except AssertionError as e:                
                #size_traceback += [utility.get_traceback(e, AssertionError)]
                self.logger.log_message(logging_base.LogFilterError, \
                                        "Dataset %s has shape %s, expected (%i, %i)" \
                                        % (key, ximg.shape, ydim.size, xdim.size))

        # Raise array-size errors if appropriate
                
        #assert(len(size_error) == len(size_traceback))
        #try:
        #    assert(len(size_error) == 0)
        #except AssertionError as e:
        #    raise errors_derived.ArraySizeFatal(self.flname, self.start_time[b], size_traceback, \
        #                                        size_error)

        return
        
        # Check slant-path spacing (ask Heresh for details)
        
        for b in self.bands:
            for f in list(self.FREQUENCIES[b].keys()):
                try:
                    slant_path = self.METADATA[b].get("radarGrid/slantRange")
                    spacing = self.FREQUENCIES[b][f].get("slantRangeSpacing")
                except KeyError:
                    continue

                utility.check_spacing(self.flname, self.start_time[b], slant_path[...], spacing[...], \
                                      "%s %s SlantPath" % (b, f), self.logger)

    def junk_check_subswaths_bounds(self):

        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():

                try:
                    nsubswath = self.check_num_subswaths(b, f)
                    assert(nsubswath > 0)
                except (KeyError, AssertionError):
                    continue
                
                for isub in range(0, int(nsubswath)):
                    try:
                        sub_bounds = self.FREQUENCIES[b][f]["validSamplesSubSwath%i" % (isub+1)][...]
                    except KeyError as e:
                        #traceback_string = [utility.get_traceback(e, KeyError)]
                        self.logger.log_message(logging_base.LogFilterWarning, 
                                                "%s Frequency%s had missing SubSwath%i bounds" \
                                                % (b, f, isub))

                    try:
                        nslantrange = self.FREQUENCIES[b][f]["slantRange"].size
                        assert(np.all(sub_bounds[:, 0] < sub_bounds[:, 1]))
                        assert(np.all(sub_bounds[:, 0] >= 0))
                        assert(np.all(sub_bounds[:, 1] <= nslantrange))
                    except KeyError:
                        continue
                    except AssertionError as e:
                        #traceback_string = [utility.get_traceback(e, KeyError)]
                        self.logger.log_message(logging_base.LogFilterWarning, 
                                                "%s Frequency%s with nSlantRange %i had invalid SubSwath bounds" \
                                                % (b, f, nslantrange))
               

               

