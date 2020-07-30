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

        self.data_names = {"phaseSigmaCoherence": "phase_coherence", \
                           "unwrappedPhase": "unwrapped_phase", \
                           "connectedComponents": "components", \
                           "ionopherePhaseScreen": "phase_ioscreen", \
                           "ionospherePhaseScreenUncertainty": "phase_uncertainty"}

        print("Keys: %s" % handle.keys())
        for long_name in self.data_names.keys():
            short_name = self.data_names[long_name]
            xdata = handle[long_name][::xstep, ::ystep]
            setattr(self, short_name, xdata)

        data1 = getattr(self, "phase_coherence")
        for short_name in self.data_names.values():
            data2 = getattr(self, short_name)
            assert(data2.shape == data1.shape)

        self.size = data1.size
        self.shape = data1.shape
        
        print("Read %s %s %s Grid image" % (self.band, self.frequency, self.polarization))

 
