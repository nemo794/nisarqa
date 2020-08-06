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

class GUNWOffsetImage(GUNWAbstractImage):

    BADVALUE = -9999
    EPS = 1.0e-03
    
    def __init__(self, band, polarization, data_names):

        GUNWAbstractImage.__init__(self, band, "None", polarization, data_names)

        self.type = "Offset"

    def junk_read(self, handle, xstep=1, ystep=1):

        for dname in self.data_names.keys():
            xdata = handle[dname][::xstep, ::ystep]
            setattr(self, self.data_names[dname], xdata)

        xdata0 = getattr(self, list(self.data_names.values())[0])
        wrong_shape = []
        for dname in self.data_names.keys():
            xdata = getattr(self, self.data_names[dname])
            if (xdata.shape != xdata0.shape):
                wrong_shape.append(dname)

        try:
            assert(len(wrong_shape) == 0)
        except AssertionError as e:
            traceback_string = [utility.get_traceback(e, AssertionError)]
            error_string = ["% Offset Image has size mismatches in %s" \
                            % (self.key, wrong_shape)]

        self.size = xdata0.size
        self.shape = xdata0.shape
        
        print("Read %s %s %s Offset image" % (self.band, self.frequency, self.polarization))

 
