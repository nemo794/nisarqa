from quality import errors_base
from quality import errors_derived
from quality.GCOVImage import GCOVImage
from quality.HdfGroup import HdfGroup
from quality.NISARFile import NISARFile
from quality import params
from quality import utility

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

class GCOVFile(NISARFile):

    def __init__(self, flname, xml_tree=None, mode="r"):
        self.flname = os.path.basename(flname)
        h5py.File.__init__(self, flname, mode)
        print("Opening file %s" % flname)

        self.xml_tree = xml_tree
        
        self.bands = []
        self.SWATHS = {}
        self.METADATA = {}
        self.IDENTIFICATION = {}

        self.images = {}
        self.start_time = {}
        self.plotted_slant = False

        self.ftype_list = ["GCOV"]
        self.product_type = "SLC"
        self.polarization_list = params.GCOV_POLARIZATION_LIST
        self.polarization_groups = params.GCOV_POLARIZATION_GROUPS
        self.identification_list = params.GCOV_ID_PARAMS
        self.frequency_checks = params.GCOV_FREQUENCY_NAMES
        self.swath_path = "/science/%s/GCOV/%s/grids"
        self.metadata_path = "/science/%s/GCOV/%s/metadata"
        self.identification_path = "/science/%s/GCOV/identification/"

        self.get_start_time()        

    def get_slant_range(self, band, frequency):

        slant_path = self.FREQUENCIES[band][frequency].get("slantRange")
        spacing = self.FREQUENCIES[band][frequency].get("slantRangeSpacing")
            
        return slant_path, spacing
            
    def get_fields_from_xml(self, xstring):

        xlist = [i.get("name") for i in self.xml_tree.iter() if ("name" in i.keys())]
        xlist = [x for x in xlist if (x.startswith(xstring))]

        return xlist
        
    def get_bands(self):

        traceback_string = []
        error_string = []
        
        for band in ("LSAR", "SSAR"):
            try:
                xband = self["/science/%s" % band]
            except KeyError:
                print("%s not present" % band)
                pass
            else:
                print("Found %s band" % band)
                self.bands.append(band)

                # Determine if data is SLC or RSLC and create HdfGroups.

                for (gname, xdict, dict_name) in zip( ("/science/%s/GCOV/grids", "/science/%s/GCOV/metadata", \
                                                       "/science/%s/identification"), \
                                                      (self.SWATHS, self.METADATA, self.IDENTIFICATION), \
                                                      ("%s Swath" % band, "%s Metadata" % band, "%s Identification") ):
                    try:
                        gname2 = gname % band
                        group = self[gname2]
                    except KeyError as e:
                        traceback_string += [utility.get_traceback(e, KeyError)]
                        error_string += ["%s does not exist" % gname2]
                    else:
                        xdict[band] = HdfGroup(self, dict_name, gname2)

        # Raise any errors
                        
        assert(len(traceback_string) == len(error_string))
        if (len(error_string) > 0):
            raise errors_derived.IdentificationFatal(self.flname, self.start_time, traceback_string, \
                                                     error_string)
                        
    def junk_get_freq_pol(self):

        self.FREQUENCIES = {}
        self.polarizations = {}
        
        for b in self.bands:

            # Find list of frequencies by directly querying dataset

            self.FREQUENCIES[b] = {}
            self.polarizations[b] = {}
            for f in ("A", "B"):
                try:
                    f2 = self["/science/%s/GCOV/grids/frequency%s" % (b, f)]
                except KeyError:
                    pass
                else:
                    print("Found %s Frequency%s" % (b, f))
                    self.FREQUENCIES[b][f] = f2
                    self.polarizations[b][f] = []
                    for p in self.polarization_list:
                        try:
                            p2 = self.FREQUENCIES[b][f][p]
                        except KeyError:
                            pass
                        else:
                            self.polarizations[b][f].append(p)
                            
    def create_images(self, time_step=1, range_step=1):

        missing_images = []
        traceback_string = []
        error_string = []

        freq = {}
        srange = {}

        for b in self.bands:
            print("%s band has frequencies %s" % (b, self.FREQUENCIES[b].keys()))
            for f in self.FREQUENCIES[b].keys():
                print("%s band frequency%s has polarizations %s" \
                      % (b, f, self.polarizations[b][f]))
                for p in self.polarizations[b][f]:
                    key = "%s %s %s" % (b, f, p)
                    try:
                        print("Creating %s image" % key)
                        self.images[key] = GCOVImage(b, f, p)
                        self.images[key].read(self.FREQUENCIES[b], time_step=time_step, range_step=range_step)
                    except KeyError:
                        print("Missing %s image" % key)
                        missing_images.append(key)
                        
        try:
            assert(len(missing_images) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = ""
            if (len(missing_images) > 0):
                error_string += "Missing %i images: %s" % (len(missing_images), missing_images)
            raise errors_derived.ArrayMissingFatal(self.flname, self.start_time[b], traceback_string, \
                                                   [error_string])
                

    def junk_find_missing_datasets(self):

        for b in self.bands:
            self.SWATHS[b].get_dataset_list(self.xml_tree, b)
            self.METADATA[b].get_dataset_list(self.xml_tree, b)
            self.IDENTIFICATION[b].get_dataset_list(self.xml_tree, b)

        error_string = []
        traceback_string = []
        
        for b in self.bands:
            no_look = []
            for f in ("A", "B"):
                for p in self.polarization_list:
                    if (f not in self.FREQUENCIES[b].keys()):
                        no_look.append("frequency%s" % f)
                    elif (f in self.FREQUENCIES[b].keys()) and (p not in self.polarizations[b][f]):
                        no_look.append("frequency%s/%s" % (f, p))
                    else:
                        try:
                            nsubswaths = self.FREQUENCIES[b][f].get("numberOfSubSwaths")[...]
                        except KeyError:
                            pass

            self.SWATHS[b].verify_dataset_list(no_look=no_look)
            self.METADATA[b].verify_dataset_list(no_look=no_look)
            self.IDENTIFICATION[b].verify_dataset_list(no_look=no_look)

            for xdict in (self.SWATHS[b], self.METADATA[b], self.IDENTIFICATION[b]):
                try:
                    assert(len(xdict.missing) == 0)
                except AssertionError as e:
                    traceback_string += [utility.get_traceback(e, AssertionError)]
                    error_string += ["%s missing %i fields: %s" % (xdict.name, len(xdict.missing), \
                                                                   ":".join(xdict.missing))]

            assert(len(error_string) == len(traceback_string))
            try:
                assert(len(error_string) == 0)
            except AssertionError as e:
                print("Missing %i datasets: %s" % (len(error_string), error_string))
                raise errors_derived.MissingDatasetFatal(self.flname, self.start_time, \
                                                         traceback_string, error_string)

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

        print("Found %i images: %s" % (len(self.images.keys()), self.images.keys()))

        traceback_string = []
        error_string = []
        for key in self.images.keys():

            # Histogram with huge number of bins to find out where the bounds of the real
            # distribution lie and then rehistogram with more sensible bounds.  Do this for
            # individual real and complex components.

            (b, f, p) = key.split()
            ximg = self.images[key]
            try:
                ximg.calc()
            except AssertionError as e:
                traceback_string += [utility.get_traceback(e, AssertionError)]
                error_string += ["%s %s %s image has %i negative backscatter pixels" \
                                % (b, f, p, ximg.nnegative)]
                #raise errors_derived.NegativeBackscatterWarning(self.flname, self.start_time, \
                #                                                traceback_string, error_string)
                 

            # Use same bounds for plotting all linear images of a given type

            #bounds_linear = [np.finfo(np.float32).max, np.finfo(np.float32).min]
            #bounds_power = [np.finfo(np.float32).max, np.finfo(np.float32).min]
            if (key not in sums_linear.keys()):
                sums_power[key] = np.zeros(ximg.power.shape, dtype=np.float32)

            sums_power[key] += ximg.power

        for (i, key) in enumerate(sums_linear.keys()):

            print("Looking at %i-th image: %s" % (i, key))
            (counts, edges) = np.histogram(sums_power[key], bins=1000)
            if (i == 0):
                counts_power = np.copy(counts)
                edges_power = np.copy(edges)
            else:
                counts_power += counts

        # Generate figures

        groups = {}
        
        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():
                keys = [k for k in self.images.keys() if k.startswith("%s %s" % (b, f))]
                print("Plotting %i images: %s" % (len(keys), keys))
                (fig, axis) = pyplot.subplots(nrows=1, ncols=1, constrained_layout=True)
                for k in keys:
                    self.images[k].plot4a(axis)

                axis.legend(loc="upper right")
                axis.set_xlabel("Power (dB)")
                axis.set_ylabel("Number of Counts")
                fig.suptitle("%s\n(%s Frequency%s)" % (self.flname, b, f))
                fpdf.savefig(fig)

            # Write histogram summaries to an hdf5 file
            
            print("File %s mode %s" % (fhdf.filename, fhdf.mode))
            fname_in = os.path.basename(self.flname)
            extension = fname_in.split(".")[-1]
            groups[b] = fhdf.create_group("%s/%s/ImageAttributes" % (fname_in.replace(".%s" % extension, ""), b))

        for key in self.images.keys():
            (b, f, p) = key.split()
            ximg = self.images[key]
            group2 = groups[b].create_group(key)

            for (name, data) in zip( ("MeanBackScatter", "SDevBackScatter", "MeanPower", "SDevPower", \
                                      "5PercentileBackScatter", "95PercentileBackScatter"), \
                                     ("mean_backscatter", "sdev_backscatter", "mean_power", "sdev_power", \
                                      "pcnt5", "pcnt95") ):
                xdata = getattr(ximg, data)
                dset = group2.create_dataset(name, (), dtype='f4')
                dset.write_direct(np.array(xdata).astype(np.float32))

        # Raise errors about negative backscatter if applicable

        assert(len(error_string) == len(traceback_string))
        try:
            assert(len(error_string) == 0)
        except AssertionError as e:
            raise errors_derived.NegativeBackscatterWarning(self.flname, self.start_time, \
                                                            traceback_string, error_string)
             
                
    def junk_check_time(self):

        for b in self.bands:

            try:
                time = self.SWATHS[b].get("zeroDopplerTime")[...]
                spacing = self.SWATHS[b].get("zeroDopplerTimeSpacing")[...]
                start_time = self.IDENTIFICATION[b].get("zeroDopplerStartTime")[...]
                end_time = self.IDENTIFICATION[b].get("zeroDopplerEndTime")[...]
            except KeyError:
                pass
            else:

                try:
                    start_time = bytes(start_time).split(b".")[0].decode("utf-8")
                    end_time = bytes(end_time).split(b".")[0].decode("utf-8")
                except UnicodeDecodeError as e:
                    traceback_string = [utility.get_traceback(e, UnicodeDecodeError)]
                    raise errors_derived.IdentificationFatal(self.flname, self.start_time[b], traceback_string, \
                                                         ["%s Start/End Times could not be read." % b])
                else:
                
                    try:
                        time1 = datetime.datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
                        time2 = datetime.datetime.strptime(end_time, "%Y-%m-%dT%H:%M:%S")
                        assert( (time2 > time1) )
                        assert( (time1.year >= 2000) and (time1.year < 2100) )
                        assert( (time2.year >= 2000) and (time2.year < 2100) )
                    except (AssertionError, ValueError) as e:
                        traceback_string = [utility.get_traceback(e, AssertionError)]
                        raise errors_derived.IdentificationFatal(self.flname, self.start_time[b], traceback_string, \
                                                                 ["%s Invalid Start and End Times" % b])
                    
                    try:
                        utility.check_spacing(self.flname, self.start_time[b], time, spacing, "%s zeroDopplerTime" % b, \
                                              errors_derived.TimeSpacingWarning, errors_derived.TimeSpacingFatal)
                    except (KeyError, errors_base.WarningError, errors_base.FatalError):
                        pass


        try:
            time = self.METADATA[b].get("orbit/time")
        except KeyError:
            contine
        else:
            try:
                utility.check_spacing(self.flname, time[0], time, time[1] - time[0], "%s orbitTime" % b, \
                                      errors_derived.TimeSpacingWarning, errors_derived.TimeSpacingFatal)
            except (errors_base.WarningError, errors_base.FatalError):
                pass
        
    def junk_check_frequencies(self):

        for b in self.bands:
            nfrequencies = len(self.FREQUENCIES[b])
            if (nfrequencies == 2):
                for freq_name in ("acquiredCenterFrequency", "processedCenterFrequency"):
                    freq = {}
                    for f in list(self.FREQUENCIES[b].keys()):
                        freq[f] = self.FREQUENCIES[b][f][freq_name][...]

                    try:
                        assert(freq["A"] < freq["B"])
                    except AssertionError as e:
                        traceback_string = [utility.get_traceback(e, AssertionError)]
                        raise errors_derived.FrequencyOrderFatal(self.flname, self.start_time[b], traceback_string, \
                                                                 ["%s A=%f not less than B=%f" \
                                                                  % (freq_name, freq["A"], freq["B"])])

    def check_slant_range(self):

        self.figures_slant = {}
        traceback_string = []
        error_string = []

        for key in self.images.keys():

            (b, f, p) = key.split()
            ximg = self.images[key]

            try:
                (srange, spacing) = self.get_slant_range(b, f)
                nslant = srange.shape[0]
                ntime = self.SWATHS[b].get("zeroDopplerTime").shape[0]
                assert(ximg.shape == (ntime, nslant))
            except KeyError:
                continue
            except AssertionError as e:
                traceback_string += [utility.get_traceback(e, AssertionError)]
                error_string += ["Dataset %s has shape %s, expected (%i, %i)" \
                                 % (key, ximg.shape, ntime, nslant)]


        assert(len(error_string) == len(traceback_string))
        try:
            assert(len(error_string) == 0)
        except AssertionError:
            raise errors_derived.ArraySizeFatal(self.flname, self.start_time[b], traceback_string, error_string)

    def check_subswaths_bounds(self):

        for b in self.bands:
            for f in self.FREQUENCIES[b].keys():

                try:
                    nsubswath = self.check_num_subswaths(b, f)
                    assert(nsubswath >= 0)
                except KeyError:
                    continue
                except AssertionError as e:
                    traceback_string = [utility.get_traceback(e, AssertionError)]
                    raise errors_derived.MissingDatasetFatal(self.flname, self.start_time, traceback_string, \
                                                             ["%s %s has invalid numberofSubSwaths: %i", \
                                                              (b, f, nsubswath)])
                
                

               

