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
    
    def __init__(self, band, frequency, polarization, acquired_frequency, spacing):

        GUNWAbstractImage.__init__(self, band, frequency, polarization)
        self.acquired_frequency = np.copy(acquired_frequency)
        self.spacing = np.copy(spacing)

        self.type = "Grid"
        self.empty = False

    def read(self, handle, xstep=1, ystep=1):

        self.phase_coherence = handle["phaseSigmaCoherence"][::xstep, ::ystep]
        self.unwrapped_phase = handle["unwrappedPhase"][::xstep, ::ystep]
        self.data_names = {"phaseSigmaCoherence": "phase_coherence", \
                           "unwrappedPhase": "unwrapped_phase"}

        assert(self.phase_coherence.shape == self.unwrapped_phase.shape)

        self.size = self.phase_coherence.size
        self.shape = self.phase_coherence.shape
        
        print("Read %s %s %s Grid image" % (self.band, self.frequency, self.polarization))

 
