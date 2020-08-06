from quality.GUNWAbstractImage import GUNWAbstractImage
from quality import errors_derived
from quality import utility

from matplotlib import cm, pyplot, ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy import constants, fftpack

import copy
import traceback

class GUNWGridImage(GUNWAbstractImage):

    BADVALUE = -9999
    EPS = 1.0e-03
    
    def __init__(self, band, frequency, polarization, data_names):

        GUNWAbstractImage.__init__(self, band, frequency, polarization, data_names)

        self.type = "Grid"

    def junk_read(self, handle, xstep=1, ystep=1):

        print("Keys: %s" % handle.keys())
        for long_name in self.data_names.keys():
            short_name = self.data_names[long_name]
            xdata = handle[long_name][::xstep, ::ystep]
            setattr(self, short_name, xdata)

            
            
        self.size = data1.size
        self.shape = data1.shape
        
        print("Read %s %s %s Grid image" % (self.band, self.frequency, self.polarization))

 
