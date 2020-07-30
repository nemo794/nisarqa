from quality import errors_base
from quality import errors_derived
from quality.GUNWGridImage import GUNWGridImage
from quality.GUNWSwathImage import GUNWSwathImage
from quality.GUNWOffsetImage import GUNWOffsetImage
from quality.HdfGroup import HdfGroup
from quality.NISARFile import NISARFile
from quality import params
from quality.SLCImage import SLCImage
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

class GUNWFile(NISARFile):

    def __init__(self, flname, xml_tree=None, mode="r"):
        self.flname = os.path.basename(flname)
        h5py.File.__init__(self, flname, mode)
        print("Opening file %s" % flname)

        self.xml_tree = xml_tree
        
        self.bands = []
        self.GRIDS = {}
        self.SWATHS = {}
        self.METADATA = {}
        self.IDENTIFICATION = {}

        self.images = {}
        self.grid_images = {}
        self.swath_images = {}
        self.offset_images = {}
        self.start_time = {}
        self.plotted_slant = False

        self.ftype_list = ["GUNW"]
        self.product_type = "GUNW"
        self.polarization_list = params.GUNW_POLARIZATION_LIST
        self.polarization_groups = params.GUNW_POLARIZATION_GROUPS
        self.identification_list = params.GUNW_ID_PARAMS
        self.frequency_checks = params.GUNW_FREQUENCY_NAMES
        self.grid_path = "/science/%s/GUNW/grids"
        self.swath_path = "/science/%s/UNW/swaths"
        self.metadata_path = "/science/%s/GUNW/metadata"
        self.identification_path = "/science/%s/identification/"

        self.get_start_time()
        self.has_swath = {}

    def get_slant_range(self, band, frequency):

        slant_path = self.FREQUENCIES[band][frequency].get("slantRange")
        spacing = self.FREQUENCIES[band][frequency].get("slantRangeSpacing")
            
        return slant_path, spacing
            
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
                print("Found band %s" % band)
                self.bands.append(band)

                # Create HdfGroups.
                
                for (gname, xdict, dict_name) in zip( ("/science/%s/GUNW/grids", "/science/%s/UNW/swaths", \
                                                       "/science/%s/GUNW/metadata", "/science/%s/identification"), \
                                                      (self.GRIDS, self.SWATHS, self.METADATA, self.IDENTIFICATION), \
                                                      ("%s Grid" % band, "%s Swath" % band, \
                                                       "%s Metadata" % band, "%s Identification" % band) ):
                    try:
                        gname2 = gname % band
                        group = self[gname2]
                    except KeyError as e:
                        if ("Swath" in dict_name):
                            self.has_swath[band] = False
                        else:
                            traceback_string += [utility.get_traceback(e, KeyError)]
                            error_string += ["%s does not exist" % gname2]
                    else:
                        xdict[band] = HdfGroup(self, dict_name, gname2)
                        if ("Swath" in dict_name):
                            self.has_swath[band] = True

        # Raise any errors
                        
        assert(len(traceback_string) == len(error_string))
        if (len(error_string) > 0):
            raise errors_derived.MissingDatasetFatal(self.flname, self.start_time[self.bands[0]], \
                                                     traceback_string, error_string)

    def get_freq_pol(self):

        self.FREQUENCIES_GRID = {}
        self.FREQUENCIES_SWATH = {}
        self.polarizations_grid = {}
        self.polarizations_swath = {}
        
        for b in self.bands:

            # Find list of frequencies by directly querying dataset

            for (freq, pol, group) in zip( (self.FREQUENCIES_GRID, self.FREQUENCIES_SWATH), \
                                           (self.polarizations_grid, self.polarizations_swath), \
                                           (self.GRIDS, self.SWATHS) ):

                print("has_swath", self.has_swath)
                print("group", group)
                if (not self.has_swath[b]) and (group is self.SWATHS):
                    continue
                
                freq[b] = {}
                pol[b] = {}
                for f in ("A", "B"):
                    try:
                        f2 = group[b].get("frequency%s" % (f))
                    except KeyError:
                        pass
                    else:
                        print("Found %s Frequency%s" % (b, f))
                        freq[b][f] = f2
                        pol[b][f] = []
                        for p in self.polarization_list:
                            try:
                                p2 = freq[b][f][p]
                            except KeyError:
                                pass
                            else:
                                pol[b][f].append(p)        

    def find_missing_datasets(self):

        for b in self.bands:
            self.GRIDS[b].get_dataset_list(self.xml_tree, b)
            self.METADATA[b].get_dataset_list(self.xml_tree, b)
            self.IDENTIFICATION[b].get_dataset_list(self.xml_tree, b)
            if (self.has_swath[b]):
                self.SWATHS[b].get_dataset_list(self.xml_tree, b)                

        error_string = []
        traceback_string = []
        
        for b in self.bands:
            no_look = {}
            for (freq, swath, pol, name) in zip( (self.FREQUENCIES_GRID, self.FREQUENCIES_SWATH), \
                                                 (self.GRIDS, self.SWATHS), \
                                                 (self.polarizations_grid, self.polarizations_swath), \
                                                 ("Grid", "Swath") ):

                if (not self.has_swath[b]) and ("Swath" in name):
                    no_look["Swath"] = []
                    continue

                no_look[name] = []
                for f in ("A", "B"):
                    if (f not in freq[b].keys()):
                        no_look[name].append("frequency%s" % f)
                        continue

                    print("Looking at %s %s" % (b, f))
                    print("keys: %s" % freq[b][f].keys())
                    try:
                        print("freq type %s" % type(freq[b][f]))
                        subswaths = freq[b][f]["numberOfSubSwaths"]
                    except KeyError:
                        nsubswaths = 0
                        pass
                    else:
                        nsubswaths = subswaths[...]
                        for isub in range(nsubswaths+1, params.NSUBSWATHS+1):
                            no_look[name].append("frequency%s/validSamplesSubSwath%i" % (f, isub))

                    found_polarizations = pol[b][f]

                    for p in list(params.SLC_POLARIZATION_LIST):
                        if (f in freq[b].keys()) and (p not in found_polarizations):
                            #print("Skipping %s frequency %s %s" % (b, f, p))
                            no_look[name].append("frequency%s/%s" % (f, p))

            print("%s: no_look=%s" % (b, no_look))

            self.GRIDS[b].verify_dataset_list(no_look=no_look["Grid"])
            self.METADATA[b].verify_dataset_list(no_look=no_look["Grid"])
            self.IDENTIFICATION[b].verify_dataset_list(no_look=set(no_look["Swath"]+no_look["Grid"]))
            if (self.has_swath[b]):
                self.SWATHS[b].verify_dataset_list(no_look=no_look["Swath"])                

            hgroups = [self.GRIDS[b], self.METADATA[b], self.IDENTIFICATION[b]]
            if (self.has_swath[b]):
                hgroups.append(self.SWATHS[b])
                
            for xdict in (hgroups):
                try:
                    assert(len(xdict.missing) == 0)
                except AssertionError as e:
                    print("Dict %s is missing %i fields" % (xdict.name, len(xdict.missing)))
                    traceback_string += [utility.get_traceback(e, AssertionError)]
                    error_string += ["%s missing %i fields: %s" % (xdict.name, len(xdict.missing), \
                                                                   ":".join(xdict.missing))]

            assert(len(error_string) == len(traceback_string))
            try:
                assert(len(error_string) == 0)
            except AssertionError as e:
                #print("Missing %i datasets: %s" % (len(error_string), error_string))
                raise errors_derived.MissingDatasetWarning(self.flname, self.start_time[b], \
                                                           traceback_string, error_string)
            
            
                                
    def create_images(self, time_step=1, range_step=1):

        missing_images = []
        missing_params = []

        freq = {}
        srange = {}

        # Check for Missing Parameters needed in image creation
        
        for b in self.bands:
            for f in self.FREQUENCIES_GRID[b].keys():
                fgroup = "/science/%s/GUNW/grids/frequency%s" % (b, f)
                ogroup = fgroup.replace("frequency%s" % f, "pixelOffsets")
                try:
                    assert("%s/centerFrequency" % fgroup in self.keys())
                    assert("%s/slantRangeSpacing" % fgroup in self.keys())
                except AssertionError:
                    missing_params += ["Grid: %s %s" % (b, f)]

                for p in self.polarizations_grid[b][f]:
                    pgroup = "%s/%s" % (fgroup, p)
                    try:
                       assert("%s/phaseSigmaCoherence" % pgroup in self.keys())
                       assert("%s/unwrappedPhase" % pgroup in self.keys())
                    except AssertionError:
                        missing_images += ["Grid: %s %s %s" % (b, f, p)]
                        print("Looking for %s/phaseSignalCoherence and %s/unwrappedPhase in %s" \
                              % (pgroup, pgroup, self[pgroup].keys()))

                    try:
                        assert("%s/%s" % (ogroup, p) in self.keys())
                    except AssertionError:
                        missing_images += ["Offset: %s %s" % (b, p)]

            if (not self.has_swath[b]):
                continue  #  No Swath Images in file

            for f in self.FREQUENCIES_SWATH[b].keys():
                fgroup = "/science/%s/UNW/swaths/frequency%s" % (b, f)
                for p in self.polarizations_swath[b][f]:
                    pgroup = "%s/%s" % (fgroup, p)
                    try:
                       assert("%s/connectedComponents" % pgroup in self.keys())
                       assert("%s/ionospherePhaseScreen" % pgroup in self.keys())
                       assert("%s/ionospherePhaseScreenUncertainty" % pgroup in self.keys())
                    except AssertionError:
                        missing_images += ["Swath: %s %s %s" % (b, f, p)]

        # Create both Grid and Swath Images
                        
        for b in self.bands:
            for f in self.FREQUENCIES_GRID[b].keys():
                fgroup = "/science/%s/GUNW/grids/frequency%s" % (b, f)
                ogroup = fgroup.replace("frequency%s" % f, "pixelOffsets")
                if ("Grid: %s %s" % (b, f) in missing_params):
                    continue

                for p in self.polarizations_grid[b][f]:
                    if ("Grid: %s %s %s" % (b, f, p) in missing_images):
                        continue
                    pgroup = "%s/%s" % (fgroup, p)
                    key = "%s %s %s" % (b, f, p)
                    self.grid_images[key] = GUNWGridImage(b, f, p, self["%s/centerFrequency" % fgroup][...], \
                                                          self["%s/slantRangeSpacing" % fgroup][...])
                    self.images["Grid: %s" % key] = self.grid_images[key]

                for p in self.polarizations_grid[b][f]:
                    if ("Offset: %s %s" % (b, p) in missing_images):
                        continue
                    key = "%s %s" % (b, p)
                    self.offset_images[key] = GUNWOffsetImage(b, p)
                    self.images["Offset: %s" % key] = self.offset_images[key]

            if (not self.has_swath[b]):
                continue  # No Swath Images in file
                    
            for f in self.FREQUENCIES_SWATH[b].keys():
                fgroup = "/science/%s/UNW/swaths/frequency%s" % (b, f)
                if ("Swath: %s %s" % (b, f) in missing_params):
                    continue

                print("Creating %s swath images" % self.polarizations_swath[b][f])
                for p in self.polarizations_swath[b][f]:
                    if ("Swath: %s %s %s" % (b, f, p) in missing_images):
                        continue
                    pgroup = "%s/%s" % (fgroup, p)
                    key = "%s %s %s" % (b, f, p)
                    self.swath_images[key] = GUNWSwathImage(b, f, p)
                    self.images["Swath: %s" % key] = self.swath_images[key]
                    

        # Raise all detected errors (if any)
            
        try:
            assert(len(missing_images) == 0)
            assert(len(missing_params) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = ""
            if (len(missing_images) > 0):
                error_string += "Missing %i images: %s" % (len(missing_images), missing_images)
            if (len(missing_params) > 0):
                error_string += "Could not initialize %i images: %s" % (len(missing_params), missing_params)
            raise errors_derived.ArrayMissingFatal(self.flname, self.start_time[b], traceback_string, \
                                                   [error_string])
 
        # Read in image data and check for array-size mismatch

        size_errors = []
        
        for key in self.grid_images.keys():
            (b, f, p) = key.split()
            try:
                print("Looking at grid: %s" % "/science/%s/GUNW/grids/frequency%s/%s" % (b, f, p))
                self.grid_images[key].read(self["/science/%s/GUNW/grids/frequency%s/%s" % (b, f, p)])
            except AssertionError as e:
                size_errors += ["Grid: %s" % key]

        for key in self.swath_images.keys():
            (b, f, p) = key.split()
            try:
                self.swath_images[key].read(self["/science/%s/UNW/swaths/frequency%s/%s" % (b, f, p)])
            except AssertionError as e:
                size_errors += ["Swath: %s" % key]

        for key in self.offset_images.keys():
            (b, p) = key.split()
            try:
                self.offset_images[key].read(self["/science/%s/GUNW/grids/pixelOffsets/%s" % (b, p)])
            except AssertionError as e:
                size_errors += ["Offset: %s" % key]

        try:
            assert(len(size_errors) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = ["Found %i array-size mismatches for images: %s" % (len(size_errors), size_errors)]
            raise errors_derived.ArraySizeFatal(self.flname, self.start_time[b], traceback_string, \
                                                error_string)

               

    def check_images(self, fpdf, fhdf):

        # Generate figures
        
        for key in self.grid_images.keys():
            (b, f, p) = key.split()
            ximg = self.grid_images[key]
            ximg.calc()
            fig = ximg.plot("%s\n(%s Frequency%s %s GUNW Histograms)" % (self.flname, b, f, p))
            fpdf.savefig(fig)

        for key in self.swath_images.keys():
            (b, f, p) = key.split()
            ximg = self.swath_images[key]
            ximg.calc()
            fig = ximg.plot("%s\n(%s Frequency%s %s UNW Histograms)" % (self.flname, b, f, p))
            fpdf.savefig(fig)

        for key in self.offset_images.keys():
            (b, p) = key.split()
            ximg = self.offset_images[key]
            ximg.calc()
            fig = ximg.plot("%s\n(%s Offset %s GUNW Histograms)" % (self.flname, b, p))
            fpdf.savefig(fig)
            

        # Write histogram summaries to an hdf5 file
            
        print("File %s mode %s" % (fhdf.filename, fhdf.mode))
        groups = {}
        fname_in = os.path.basename(self.filename)
        extension = fname_in.split(".")[-1]
        groups[b] = fhdf.create_group("%s/%s/ImageAttributes" % (fname_in.replace(".%s" % extension, ""), b))

        for images in (self.grid_images, self.swath_images, self.offset_images):
        
            for key in images.keys():
                try:
                    (b, f, p) = key.split()
                except ValueError:
                    (b, p) = key.split()
                ximg = images[key]
                print("Creating group: '%s: %s'" % (ximg.type, key))
                group2 = groups[b].create_group("%s: %s" % (ximg.type, key))

                for dname in ximg.data_names.keys():
                    dset = group2.create_dataset("mean_%s" % dname, data=ximg.means[dname])
                    dset = group2.create_dataset("sdev_%s" % dname, data=ximg.sdev[dname])
                    dset = group2.create_dataset("histedges_%s" % dname, data=ximg.hist_edges[dname])
                    dset = group2.create_dataset("histcounts_%s" % dname, data=ximg.hist_counts[dname])

    def check_slant_range(self):

        return # NOT SURE YET WHAT TO DO HERE
        
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
                size_traceback += [utility.get_traceback(e, AssertionError)]
                size_error += ["Dataset %s has shape %s, expected (%i, %i)" \
                               % (key, ximg.shape, ydim.size, xdim.size)]

        # Raise array-size errors if appropriate
                
        assert(len(size_error) == len(size_traceback))
        try:
            assert(len(size_error) == 0)
        except AssertionError as e:
            raise errors_derived.ArraySizeFatal(self.flname, self.start_time[b], size_traceback, \
                                                size_error)

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
                                      "%s %s SlantPath" % (b, f), errors_derived.SlantSpacingWarning, \
                                      errors_derived.SlantSpacingFatal)

