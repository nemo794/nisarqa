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

class GUNWSwathImage(GUNWAbstractImage):

    BADVALUE = -9999
    EPS = 1.0e-03
    
    def __init__(self, band, frequency, polarization):

        GUNWAbstractImage.__init__(self, band, frequency, polarization)

        self.type = "Swath"
        self.empty = False

    def read(self, handle, xstep=1, ystep=1):

        self.components = handle["connectedComponents"][::xstep, ::ystep]
        self.phase_screen = handle["ionospherePhaseScreen"][::xstep, ::ystep]
        self.phase_uncertainty = handle["ionospherePhaseScreenUncertainty"][::xstep, ::ystep]
        self.data_names = {"connectedComponents": "components", \
                           "ionospherePhaseScreen": "phase_screen", \
                           "ionospherePhaseScreenUncertainty": "phase_uncertainty"}

        assert(self.components.shape == self.phase_screen.shape)
        assert(self.components.shape == self.phase_uncertainty.shape)

        self.size = self.components.size
        self.shape = self.components.shape

        print("Read %s %s %s swath image" % (self.band, self.frequency, self.polarization))


 
