from quality import errors_derived
from quality import utility

from matplotlib import cm, pyplot, ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy import constants, fftpack

import copy
import traceback

class GUNWAbstractImage(object):

    BADVALUE = -9999
    EPS = 1.0e-03
    
    def __init__(self, band, frequency, polarization):

        self.band = band
        self.frequency = frequency
        self.polarization = polarization

        self.empty = False

    def check_for_nan(self):

        self.num_nan = 0
        self.num_zero = 0

        for dname in self.data_names.keys():
            xdata = getattr(self, self.data_names[dname])
            self.nan_mask = np.isnan(xdata) | np.isinf(xdata)
            self.num_nan = max(self.nan_mask.sum(), self.num_nan)
            self.perc_nan = 100.0*self.num_nan/self.size

        self.empty = False
        
        if (self.num_nan == self.size):
            self.empty_string = ["%s: %s %s_%s is entirely NaN" % (self.type, self.band, self.frequency, self.polarization)]
            self.empty = True
        elif (self.num_nan > 0) and (self.num_nan < self.xdata.size):
            self.nan_string = ["%s: %s %s_%s has %i NaN's=%s%%" % (self.type, self.band, self.frequency, \
                                                                   self.polarization, self.num_nan, \
                                                                   round(self.perc_nan, 1))]

    def calc(self):

        nslices = self.shape[-1]

        self.means = {}
        self.sdev = {}
        
        for dname in self.data_names.keys():
            xdata = getattr(self, self.data_names[dname])
            self.means[dname] = np.zeros((nslices), dtype=np.float32) + self.BADVALUE
            self.sdev[dname] = np.zeros((nslices), dtype=np.float32) + self.BADVALUE
            
            mask_ok = np.where(~np.isnan(xdata) & ~np.isinf(xdata), True, False)
            self.means[dname] = xdata[mask_ok].mean()
            self.sdev[dname] = xdata[mask_ok].std()

    def plot(self, title):

        # Compute histograms and plot them

        self.hist_edges = {}
        self.hist_counts = {}
        
        (fig, axes) = pyplot.subplots(nrows=len(self.data_names.keys()), ncols=1, sharex=False, sharey=False, \
                                      constrained_layout=True)

        for (i, dname) in enumerate(self.data_names.keys()):

            xdata = getattr(self, self.data_names[dname])
            mask_ok = np.where(~np.isnan(xdata) & ~np.isinf(xdata) & (xdata != 0.0), True, False)
            (counts, edges) = np.histogram(xdata[mask_ok], bins=50)

            self.hist_edges[dname] = np.copy(edges)
            self.hist_counts[dname] = np.copy(counts)
            print("%s: edges %s, counts %s" % (dname, edges, counts))

            idx_mode = np.argmax(counts)
            axes[i].plot(edges[:-1], counts, label="Mode %.1f" % (round(edges[idx_mode], 1)))
            axes[i].legend(loc="upper right", fontsize="small")
            axes[i].set_xlabel(dname)

        fig.suptitle(title)
                
        return fig
                           

