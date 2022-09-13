#!/usr/bin/env python3
import h5py
# Switch backend to one that doesn't require DISPLAY to be set since we're
# just plotting to file anyway. (Some compute notes do not allow X connections)
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

# from products import goff
from utils import parsing
from utils import utils

def main(args=None):
    """
    Main executable script for QA checks of NISAR GOFF products.
    """

    print("TODO: Complete GOFF QA script and checks.")


if __name__ == "__main__":
    args = parsing.parse_args('goff')
    main(args)
    
