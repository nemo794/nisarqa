import check_time
import errors_base
import errors_derived
from SLCFile import SLCFile
import utility

import optparse
import os, os.path
import sys

import h5py
from matplotlib.backends.backend_pdf import PdfPages

if __name__ == "__main__":

    parser = optparse.OptionParser()
    parser.add_option("--log", "--flog", dest="flog", type="string", action="store")
    parser.add_option("--hdf", "--fhdf", dest="fhdf", type="string", action="store")
    parser.add_option("--pdf", "--fpdf", dest="fpdf", type="string", action="store")
    (kwds, args) = utility.parse_args(parser)

    fpdf = PdfPages(kwds["fpdf"])
    
    for slc_file in args:

        errors_base.WarningError.reset(errors_base.WarningError)
        errors_base.FatalError.reset(errors_base.FatalError)
        
        try:
            fhdf = SLCFile(slc_file, "r")
        except errors_base.FatalError:
            errors_base.FatalError.print_log(errors_base.FatalError, os.path.basename(args[0]), \
                                             kwds["flog"])        
            sys.exit(1)

        # Verify identification information

        try:
            fhdf.check_identification()
        except errors_base.FatalError:
            pass

        # Verify frequencies and polarizations

        try:
            fhdf.check_frequencies()
        except errors_base.FatalError:
            pass

        # Verify time tags
    
        try:
            fhdf.check_time()
        except (errors_base.WarningError, errors_base.FatalError):
            pass
    
        # Verify slant path tags

        try:
            fhdf.check_slant_range()
        except (errors_base.WarningError, errors_base.FatalError):
            pass

        # Verify SubSwath boundaries

        try:
            fhdf.check_subswaths()
        except errors_base.FatalError:
            pass
    
        # Check for NaN's and plot images

        try:
            fhdf.check_images(fpdf)
        except errors_base.WarningError:
            pass
    
        # Close files

        fhdf.close()
        errors_base.WarningError.print_log(errors_base.WarningError, \
                                           os.path.basename(slc_file), kwds["flog"])
        errors_base.FatalError.print_log(errors_base.FatalError, \
                                         os.path.basename(slc_file), kwds["flog"])

    # Print summary

    fpdf.close()
                                       
    
    
        

        
        
    
