from quality import check_time
from quality import errors_base
from quality import errors_derived
from quality import logging_base
from quality.LogError import LogError
from quality.SLCFile import SLCFile
from quality import utility

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
    parser.add_option("--time_step", dest="time_step", type="int", action="store", default=1)
    parser.add_option("--range_step", dest="range_step", type="int", action="store", default=1)
    parser.add_option("--validate", dest="validate", action="store_true", default=False)
    parser.add_option("--quality", dest="quality", action="store_true", default=False)
    parser.add_option("--xml_dir", dest="xml_dir", type="string", action="store", default="xml")
    parser.add_option("--xml_file", dest="xml_file", type="string", action="store", default="nisar_L1_RSLC.xml")

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
    for slc_file in args:

        logger.log_message(logging_base.LogFilterInfo, \
                           "Opening file %s with xml spec %s" % (slc_file, xml_path))
        fhdf = SLCFile(slc_file, logger, xml_tree=xml_tree, mode="r")
        if (not fhdf.is_open):
            continue
        
        fhdf.get_start_time()

        file_bad = False
        errors = fhdf.get_bands()
        try:
            assert(len(errors) == 0)
        except AssertionError:
            file_bad = True
            bad_files.append(slc_file)
            logger.log_message(logging_base.LogFilterError, \
                               "File %s has a Fatal Error(s): %s" % (slc_file, errors))
            fhdf.close()
            continue

        fhdf.get_freq_pol()
        for band in fhdf.bands:
            errors = fhdf.check_freq_pol(band, [fhdf.SWATHS], [fhdf.FREQUENCIES], [""])            
            if (len(errors) > 0):
                file_bad = True
                bad_files.append(slc_file)
                logger.log_message(logging_base.LogFilterError, \
                                   "File %s has a Fatal Error(s): %s" % (slc_file, errors))
                fhdf.close()
                break

        if (file_bad):
            continue
             
        if (kwds["validate"]):
            fhdf.find_missing_datasets([fhdf.SWATHS], [fhdf.FREQUENCIES])
            fhdf.check_identification()
            for band in fhdf.bands:
                fhdf.check_frequencies(band, fhdf.FREQUENCIES[band])
            fhdf.check_time()
            fhdf.check_slant_range()
            fhdf.check_subswaths_bounds()

        # Check for NaN's and plot images

        if (kwds["quality"]):
            fhdf.create_images(time_step=kwds["time_step"], range_step=kwds["range_step"])
            fhdf.check_nans()
            fhdf.check_images(fpdf_out, fhdf_out)

        # Close files

        fhdf.close()
        #if (kwds["validate"]):
            #flog.print_file_logs(os.path.basename(slc_file))
            #flog.print_error_matrix(os.path.basename(slc_file))

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

        
        
    
