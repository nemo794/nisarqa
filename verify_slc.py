import check_time
import errors_base
import errors_derived
from LogError import LogError
from SLCFile import SLCFile
import utility

import optparse
import os, os.path
import sys
import time

import h5py
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
    (kwds, args) = utility.parse_args(parser)

    time1 = time.time()
    
    if (kwds["quality"]):
        assert("fpdf" in kwds.keys())
        fhdf_out = h5py.File(kwds["fhdf"], "w")
        fpdf_out = PdfPages(kwds["fpdf"])

    if (kwds["validate"]):
        flog = LogError(kwds["flog"])
        flog.make_header(args)
    
    for slc_file in args:

        #errors_base.WarningError.reset(errors_base.WarningError)
        #errors_base.FatalError.reset(errors_base.FatalError)
        
        fhdf = SLCFile(slc_file, "r")
        
        try:
            fhdf.get_bands()
        except errors_base.FatalError:
            print("File %s has a Fatal Error" % slc_file)

        fhdf.get_freq_pol()
            
        try:
            fhdf.check_freq_pol()
        except errors_base.FatalError:
            print("File %s has a Fatal Error" % slc_file)

        # Verify identification information

        if (kwds["validate"]):
            try:
                fhdf.check_identification()
            except errors_base.FatalError:
                pass

        # Verify frequencies and polarizations

        if (kwds["validate"]):
            try:
                fhdf.check_frequencies()
            except errors_base.FatalError:
                pass

        # Verify time tags

        if (kwds["validate"]):
            try:
                fhdf.check_time()
            except (errors_base.WarningError, errors_base.FatalError):
                pass
    
        # Verify slant path tags

        if (kwds["validate"]):
            try:
                fhdf.check_slant_range()
            except (errors_base.WarningError, errors_base.FatalError):
                pass

        # Verify SubSwath boundaries

        if (kwds["validate"]):
            try:
                fhdf.check_subswaths()
            except errors_base.FatalError:
                pass
    
        # Check for NaN's and plot images

        if (kwds["quality"]):

            try:
                fhdf.create_images(time_step=kwds["time_step"], range_step=kwds["range_step"])
            except errors_base.FatalError:
                pass
            else:
                try:
                    fhdf.check_images(fpdf_out, fhdf_out)
                except (errors_base.WarningError, errors_base.FatalError):
                    pass
    
        # Close files

        fhdf.close()
        if (kwds["validate"]):
            flog.print_file_logs(os.path.basename(slc_file))
            #flog.print_error_matrix(os.path.basename(slc_file))

    # Close pdf file

    if (kwds["quality"]):
        fpdf_out.close()
        fhdf_out.close()
                                       
    
    time2 = time.time()
    print("Runtime = %i seconds" % (time2-time1))
        

        
        
    
