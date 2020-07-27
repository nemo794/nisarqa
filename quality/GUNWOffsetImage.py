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
    
    def __init__(self, band, polarization):

        GUNWAbstractImage.__init__(self, band, "None", polarization)

        self.type = "Offset"
        self.empty = False

    def read(self, handle, xstep=1, ystep=1):

        self.ltr_offset = handle["alongTrackOffset"][::xstep, ::ystep]
        self.correlation = handle["correlation"][::xstep, ::ystep]
        self.slr_offset = handle["slantRangeOffset"][::xstep, ::ystep]
        self.data_names = {"alongTrackOffset": "ltr_offset", \
                           "correlation": "correlation", \
                           "slantRangeOffset": "slr_offset"}

        assert(self.correlation.shape == self.ltr_offset.shape)
        assert(self.slr_offset.shape == self.slr_offset.shape)

        self.size = self.ltr_offset.size
        self.shape = self.ltr_offset.shape
        
        print("Read %s %s %s Offset image" % (self.band, self.frequency, self.polarization))

 
