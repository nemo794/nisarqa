#!/usr/bin/env python3
from quality import check_time
from quality import errors_base
from quality import errors_derived
from quality import logging_base
from quality.LogError import LogError
from quality.GUNWFile import GUNWFile
from quality import utility

import logging
import optparse
import os, os.path
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

import h5py
# Switch backend to one that doesn't require DISPLAY to be set since we're
# just plotting to file anyway. (Some compute notes do not allow X connections)
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

WORKFLOW_NAME = "Q/A"
PGE_NAME = "Q/A"

if __name__ == "__main__":

    parser = optparse.OptionParser()
    parser.add_option("--log", "--flog", dest="flog", type="string", action="store")
    parser.add_option("--hdf", "--fhdf", dest="fhdf", type="string", action="store")
    parser.add_option("--pdf", "--fpdf", dest="fpdf", type="string", action="store")
    parser.add_option("--xstep", dest="xstep", type="int", action="store", default=1)
    parser.add_option("--ystep", dest="ystep", type="int", action="store", default=1)
    parser.add_option("--validate", dest="validate", action="store_true", default=False)
    parser.add_option("--quality", dest="quality", action="store_true", default=False)
    parser.add_option("--xml_dir", dest="xml_dir", type="string", action="store", default="xml")
    parser.add_option("--xml_file", dest="xml_file", type="string", action="store", default="nisar_L2_GUNW.xml")

    (kwds, args) = utility.parse_args(parser)
    if ("flog" not in kwds.keys()) and ("fpdf" not in kwds.keys()) and \
       ("fhdf" not in kwds.keys()):
        (kwds, args) = utility.parse_yaml(kwds, args)
    
    logger = logging_base.NISARLogger(kwds["flog"])
    time1 = time.time()
    
    if (kwds["quality"]):
        assert("fpdf" in kwds.keys())
        fhdf_out = h5py.File(kwds["fhdf"], "w")
        fpdf_out = PdfPages(kwds["fpdf"])

    xml_tree = None
    xml_path = os.path.realpath(pathlib.Path(__file__))
    xml_path = os.path.join(pathlib.Path(xml_path).parent, kwds["xml_dir"], kwds["xml_file"])

    try:
        assert(os.path.exists(xml_path))
    except AssertionError:
        logger.log_message(logging_base.LogFilterError, "XML file %s does not exist" % xml_path)
        logger.close()
        sys.exit(1)
    else:
        try:
            xml_tree = ET.parse(xml_path)
        except:
            logger.log_message(logging_base.LogFilterError, \
                               "Could not parse XML file %s" % xml_path)
            sys.exit(1)
        else:
            logger.log_message(logging_base.LogFilterInfo, \
                               "Successfully parsed XML file %s" % xml_path)

    bad_files = []
    for gunw_file in args:
        
        fhdf = GUNWFile(gunw_file, logger, xml_tree=xml_tree, mode="r")
        if (not fhdf.is_open):
            continue

        file_bad = False
        fhdf.get_bands()
        fhdf.get_freq_pol()
        for b in fhdf.bands:
            if (fhdf.has_swath[b]):
                fgroups1 = [fhdf.GRIDS, fhdf.SWATHS]
                fgroups2 = [fhdf.FREQUENCIES_GRID, fhdf.FREQUENCIES_SWATH]
                fnames2 = ["Grids", "Swaths"]
            else:
                fgroups1 = [fhdf.GRIDS]
                fgroups2 = [fhdf.FREQUENCIES_GRID]
                fnames2 = ["Grids"]
        
            errors = fhdf.check_freq_pol(b, fgroups1, fgroups2, fnames2)
            if (len(errors) > 0):
                file_bad = True
                fhdf.close()
                logger.log_message(logging_base.LogFilterError, "File %s has a Fatal Error" % gunw_file)
                bad_files.append(gunw_file)
                break

        if (file_bad):
            continue
                
        if (kwds["validate"]):
            fhdf.find_missing_datasets()
            fhdf.check_identification()
            for band in fhdf.bands:
                fhdf.check_frequencies(band, fhdf.FREQUENCIES_GRID[band])
                if (fhdf.has_swath[b]):
                    fhdf.check_frequencies(band, fhdf.FREQUENCIES_SWATH[band])

            # Not sure if these are needed for GUNW files
            #fhdf.check_time()
            #fhdf.check_slant_range()
            #fhdf.check_subswaths_bounds()

        # Check for NaN's and plot images

        if (kwds["quality"]):
            fhdf.create_images(xstep=kwds["xstep"], ystep=kwds["ystep"])
            fhdf.check_nans()
            fhdf.check_images(fpdf_out, fhdf_out)
    
        # Close files

        fhdf.close()

    # Close pdf file

    if (kwds["quality"]):
        fpdf_out.close()
        fhdf_out.close()
                                       
    
    time2 = time.time()
    logger.log_message(logging_base.LogFilterInfo, "Runtime = %i seconds" % (time2-time1))
    logger.close()

    if (len(bad_files) == 0):
        print("Successful completion.")
    else:
        print("Fatal Errors encountered in %i files: %s." \
              % (len(bad_files), bad_files))
        
    
