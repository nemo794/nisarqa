from quality import errors_base
from quality import errors_derived
from quality import logging_base
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

    def __init__(self, flname, logger, xml_tree=None, mode="r"):
        NISARFile.__init__(self, flname, logger, xml_tree=xml_tree, mode=mode)
        
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
        self.xcoordinates = {}
        self.ycoordinates = {}
        self.plotted_slant = False

        self.ftype_list = ["GUNW"]
        self.product_type = "GUNW"
        self.polarizations_possible = ("HH", "HV", "VH", "VV")        
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

        self.grid_params = {"phaseSigmaCoherence": "phase_coherence", \
                            "unwrappedPhase": "unwrapped_phase", \
                            "connectedComponents": "components", \
                            "coherenceMask": "mask"}
                           #"ionospherePhaseScreen": "phase_ioscreen", \
                           #"ionospherePhaseScreenUncertainty": "phase_uncertainty"}
        self.swath_params = self.grid_params
        self.offset_params = {"alongTrackOffset": "ltr_offset", \
                              "correlation": "correlation", \
                              "slantRangeOffset": "slr_offset"}
        self.frequency_params = ["xCoordinates", "yCoordinates"]

    def get_coordinates(self):

        for key in self.images.keys():
            ximg = self.images[key]
            if (not ximg.has_coords):
                continue

            for c in ("x", "y"):
                coordinates = getattr(ximg, "%scoord" % c)
                try:
                    spacing = ximg.handle_coords["%sCoordinatesSpacing" % c][...]
                except KeyError:
                    self.logger.log_message(logging_base.LogFilterWarning, \
                                            "handle %s doesn't have dataset %sCoordinateSpacing" \
                                            % (ximg.handle_coords.name, c))
                    continue

                self.logger.log_message(logging_base.LogFilterDebug, \
                                        "%s %s coord: %s" % (key, c, coordinates))
                self.logger.log_message(logging_base.LogFilterDebug, \
                                        "%s %s spacing: %s" % (key, c, spacing))
                
                error_string = utility.check_spacing(self.flname, self.start_time[ximg.band], coordinates,\
                                                     spacing, "%s %sCoordinates" % (key, c)) #, \
                                                     #errors_derived.CoordinateSpacingWarning, \
                                                     #errors_derived.CoordinateSpacingFatal)
                if (len(error_string) > 0):
                    for e in error_string:
                        self.logger.log_message(logging_base.LogFilterWarning, e)
        
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
                self.logger.log_message(logging_base.LogFilterInfo, "%s not present" % band)
                pass
            else:
                self.logger.log_message(logging_base.LogFilterInfo, "Found band %s" % band)
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
                            self.logger.log_message(logging_base.LogFilterError, \
                                                    "%s does not exist" % gname2)
                    else:
                        xdict[band] = HdfGroup(self, dict_name, gname2, self.logger)
                        if ("Swath" in dict_name):
                            self.has_swath[band] = True


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
                        self.logger.log_message(logging_base.LogFilterInfo, "Found %s Frequency%s" % (b, f))
                        freq[b][f] = f2
                        pol[b][f] = []
                        for p in self.polarization_list:
                            try:
                                p2 = freq[b][f]["interferogram/%s" % p]
                            except KeyError:
                                pass
                            else:
                                pol[b][f].append(p)


    def find_missing_datasets(self):

        hgroups = []
        missing_groups = []
        
        for b in self.bands:
            for gname in ("GRIDS", "METADATA", "IDENTIFICATION"):
                try:
                    group = getattr(self, gname)
                    group = group[b]
                except KeyError:
                    missing_groups.append(gname)
                else:
                    group.get_dataset_list(self.xml_tree, b)
                    hgroups.append(group)
                    
            #self.GRIDS[b].get_dataset_list(self.xml_tree, b)
            #self.METADATA[b].get_dataset_list(self.xml_tree, b)
            #self.IDENTIFICATION[b].get_dataset_list(self.xml_tree, b)
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

                    self.logger.log_message(logging_base.LogFilterInfo, "Looking at %s %s" % (b, f))
                    try:
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

            if ("GRIDS" not in missing_groups):
                self.GRIDS[b].verify_dataset_list(no_look=no_look["Grid"])
            if ("METADATA" not in missing_groups):
                self.METADATA[b].verify_dataset_list(no_look=no_look["Grid"])
            if ("IDENTIFICATION" not in missing_groups):
                self.IDENTIFICATION[b].verify_dataset_list(no_look=set(no_look["Swath"]+no_look["Grid"]))
            if (self.has_swath[b]):
                self.SWATHS[b].verify_dataset_list(no_look=no_look["Swath"])                

            #hgroups = [self.GRIDS[b], self.METADATA[b], self.IDENTIFICATION[b]]
            if (self.has_swath[b]):
                hgroups.append(self.SWATHS[b])
                
        for xdict in hgroups:
            try:
                assert(len(xdict.missing) == 0)
            except AssertionError as e:
                self.logger.log_message(logging_base.LogFilterError, \
                                        "%s missing %i fields: %s" % (xdict.name, len(xdict.missing), \
                                                                      ":".join(xdict.missing)))

    def create_images(self, xstep=1, ystep=1):

        missing_images = []
        missing_params = []

        freq = {}
        srange = {}

        # Check for Missing Parameters needed in image creation

        self.logger.log_message(logging_base.LogFilterInfo, \
                                "Grid frequencies %s" % self.FREQUENCIES_GRID["LSAR"].keys())
        
        for b in self.bands:
            for f in self.FREQUENCIES_GRID[b].keys():
                fgroup = "/science/%s/GUNW/grids/frequency%s/interferogram" % (b, f)
                ogroup = fgroup.replace("interferogram", "pixelOffsets")
                for p in self.polarizations_grid[b][f]:
                    if ("%s/%s" % (fgroup, p) not in self.keys()):
                        missing_images.append("Grid: (%s, %s, %s)" % (b, f, p))
                    if ("%s/%s" % (ogroup, p) not in self.keys()):
                        missing_images.append("Offset: (%s, %s, %s)" % (b, f, p))
                         

            if (not self.has_swath[b]):
                continue  #  No Swath Images in file

            for f in self.FREQUENCIES_SWATH[b].keys():
                fgroup = "/science/%s/UNW/swaths/frequency%s" % (b, f)
                for p in self.polarizations_swath[b][f]:
                    if ("%s/%s" % (fgroup, p) not in self.keys()):
                        missing_images.append("Offset: (%f, %f, %f)" % (b, f, p))

        if (len(missing_images) > 0):
            self.logger.log_message(logging_base.LogFilterError, "File is missing %i images: %s" \
                                    % (len(missing_images), missing_images))

        # Create both Grid and Swath Images
                        
        for b in self.bands:
            for f in self.FREQUENCIES_GRID[b].keys():
                #print("%s Frequency%s has polarizations %s" % (b, f, self.polarizations_grid[b][f]))
                fgroup = "/science/%s/GUNW/grids/frequency%s/interferogram" % (b, f)
                ogroup = fgroup.replace("interferogram", "pixelOffsets")

                for p in self.polarizations_grid[b][f]:
                    if ("Grid: (%s, %s, %s)" % (b, f, p) in missing_images):
                        continue
                    pgroup = "%s/%s" % (fgroup, p)
                    key = "(%s %s %s)" % (b, f, p)
                    self.grid_images[key] = GUNWGridImage(b, f, p, self.grid_params)
                    self.images["Grid: %s" % key] = self.grid_images[key]

                for p in self.polarizations_grid[b][f]:
                    key = "(%s %s %s)" % (b, f, p)
                    if ("Offset: (%s, %s, %s)" % (b, f, p) in missing_images):
                        continue
                    self.offset_images[key] = GUNWOffsetImage(b, p, self.offset_params)
                    self.images["Offset: %s" % key] = self.offset_images[key]

            if (not self.has_swath[b]):
                continue  # No Swath Images in file
                    
            for f in self.FREQUENCIES_SWATH[b].keys():
                fgroup = "/science/%s/UNW/swaths/frequency%s" % (b, f)
                for p in self.polarizations_swath[b][f]:
                    if ("Swath: (%s, %s, %s)" % (b, f, p) in missing_images):
                        continue
                    key = "(%s %s %s)" % (b, f, p)
                    self.swath_images[key] = GUNWSwathImage(b, f, p, self.swath_params)
                    self.images["Swath: %s" % key] = self.swath_images[key]

        self.logger.log_message(logging_base.LogFilterInfo, \
                                "File is missing images %s" % (missing_images))
        self.logger.log_message(logging_base.LogFilterInfo, \
                                "Found grid images: %s" % self.grid_images.keys())
        self.logger.log_message(logging_base.LogFilterInfo, \
                                "Found offset images: %s" % self.offset_images.keys())
                    
        # Read in image data and check for array-size mismatch

        size_errors = []
        
        for key in self.grid_images.keys():
            (b, f, p) = key.replace("(", "").replace(")", "").split()
            name1 = "/science/%s/GUNW/grids/frequency%s/interferogram/%s" % (b, f, p)
            name2 = "/science/%s/GUNW/grids/frequency%s" % (b, f)
            self.logger.log_message(logging_base.LogFilterInfo, "Reading grid: %s" \
                                    % "/science/%s/GUNW/grids/frequency%s/%s" % (b, f, p))
            self.grid_images[key].read(self["/science/%s/GUNW/grids/frequency%s/interferogram/%s" % (b, f, p)], \
                                       self["/science/%s/GUNW/grids/frequency%s" % (b, f)], \
                                       xstep=xstep, ystep=ystep)

        for key in self.swath_images.keys():
            (b, f, p) = key.replace("(", "").replace(")", "").split()
            self.swath_images[key].read(self["/science/%s/UNW/swaths/frequency%s/%s" % (b, f, p)], \
                                        self["/science/%s/UNW/swaths/frequency%s" % (b, f)], \
                                        xstep=xstep, ystep=ystep)
 
        for key in self.offset_images.keys():
            (b, f, p) = key.replace("(", "").replace(")", "").split()
            self.offset_images[key].read(self["/science/%s/GUNW/grids/frequency%s/pixelOffsets/%s" % (b, f, p)], \
                                         self["/science/%s/GUNW/grids/frequency%s" % (b, f)], \
                                         xstep=xstep, ystep=ystep)
        # Log all errors

        self.missing_images = []
        for key in self.images:
            ximg = self.images[key]
            if (ximg.is_empty):
                self.missing_images.append(key)
                self.logger.log_message(logging_base.LogFilterError, \
                                        "%s is missing all datasets" % (key))
                continue
            if (len(ximg.missing_datasets) > 0):
                self.logger.log_message(logging_base.LogFilterError, \
                                        "%s is missing datasets: %s" % (key, ximg.missing_datasets))
            if (len(ximg.wrong_shape_inconsistent) > 0):
                self.logger.log_message(logging_base.LogFilterWarning, \
                                        "%s inconsistent size in params %s" \
                                        % (key, ximg.wrong_shape_inconsistent))
            if (len(ximg.wrong_shape_coords) > 0):
                self.logger.log_message(logging_base.LogFilterWarning, \
                                        "%s size doesn't match coordinates" % key)# in params %s"\
                                        # % (key, ximg.wrong_shape_coords))
            if (not ximg.has_coords):
                self.logger.log_message(logging_base.LogFilterError, \
                                        "%s doesn't have x/y coordinates" % key)
                
    def check_images(self, fpdf, fhdf):

        # Generate figures

        figures = []

        for key in self.grid_images.keys():
            (b, f, p) = key.split()
            ximg = self.grid_images[key]
            ximg.calc()
            ximg.find_regions("connectedComponents")
            ximg.calc_connect(ximg.region_map, ximg.components, contiguous=True)
            ximg.calc_connect(ximg.components, ximg.components, contiguous=False)

            self.logger.log_message(logging_base.LogFilterDebug, ximg.region_debug)
            if (len(ximg.empty_error_list) > 0):
                self.logger.log_message(logging_base.LogFilterError, \
                                        "RegionGrowing images are empty: %s" % ximg.empty_error_list)
            if (len(ximg.region_error_list) > 0):
                self.logger.log_message(logging_base.LogFilterError, \
                                        "RegionGrowing Failed for %s" % ximg.region_error_list)

            figures += ximg.plot("%s\n(%s Frequency%s %s GUNW Histograms)" % (self.flname, b, f, p))
            figures += ximg.plot_region_map("%s\n(%s Frequency%s %s RegionMaps" % (self.flname, b, f, p))
            figures += ximg.plot_region_hists("%s\n(%s Frequency%s %s RegionSummaries" % (self.flname, b, f, p))

        for key in self.swath_images.keys():
            (b, f, p) = key.split()
            ximg = self.swath_images[key]
            ximg.calc()
            figures += ximg.plot("%s\n(%s Frequency%s %s UNW Histograms)" % (self.flname, b, f, p))

        for key in self.offset_images.keys():
            (b, f, p) = key.split()
            ximg = self.offset_images[key]
            ximg.calc()
            figures += ximg.plot("%s\n(%s Frequency%s Offset %s GUNW Histograms)" % (self.flname, b, f, p))
            
        for fig in figures:
            fpdf.savefig(fig)

        # Write histogram summaries to an hdf5 file
            
        groups = {}
        fname_in = os.path.basename(self.filename)
        extension = fname_in.split(".")[-1]
        for b in self.bands:
            groups[b] = fhdf.create_group("%s/%s/ImageAttributes" \
                                          % (fname_in.replace(".%s" % extension, ""), b))

        for images in (self.grid_images, self.swath_images, self.offset_images):
        
            for key in images.keys():
                try:
                    (b, f, p) = key.replace("(", "").replace(")", "").split()
                except ValueError:
                    (b, p) = key.replace("(", "").replace(")", "").split()
                ximg = images[key]
                self.logger.log_message(logging_base.LogFilterInfo, \
                                        "Creating group: '%s: %s'" % (ximg.type, key))
                group2 = groups[b].create_group("%s: %s" % (ximg.type, key))

                for dname in ximg.data_names.keys():
                    dset = group2.create_dataset("mean_%s" % dname, data=ximg.means[dname])
                    dset = group2.create_dataset("sdev_%s" % dname, data=ximg.sdev[dname])
                    dset = group2.create_dataset("histedges_%s" % dname, data=ximg.hist_edges[dname])
                    dset = group2.create_dataset("histcounts_%s" % dname, data=ximg.hist_counts[dname])

                if (len(ximg.region_size.keys()) == 0):
                    continue

                dset = group2.create_dataset("pcnt_nonzero_connected (%s)" % ximg.region_dname, \
                                             data=ximg.connect_nonzero)

                for rkey in ximg.region_size.keys():
                    dname = "connectedComponents_%s" % rkey
                    dset = group2.create_dataset("region_value (%s)" % dname, data=ximg.region_size[rkey][0])
                    dset = group2.create_dataset("region_size (%s)" % dname, data=ximg.region_size[rkey][1])
                    dset = group2.create_dataset("region_size_xaxis (%s)" % dname, data=ximg.region_hist[rkey][0])
                    dset = group2.create_dataset("region_size_yaxis (%s)" % dname, data=ximg.region_hist[rkey][1])

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

