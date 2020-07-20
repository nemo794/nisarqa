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
            
            for i in range(0, nslices):
                xslice = xdata[..., i]
                mask_ok = np.where(~np.isnan(xslice) & ~np.isinf(xslice), True, False)
                if (mask_ok.sum() < xslice.size):
                    self.means[dname][i] = xslice[mask_ok].mean()
                    self.sdev[dname][i] = xslice[mask_ok].std()

    def plot(self, title):

        # Compute histograms and plot them

        self.hist_edges = {}
        self.hist_counts = {}
        
        (fig, axes) = pyplot.subplots(nrows=len(self.data_names.keys()), ncols=1, sharex=False, sharey=False, \
                                      constrained_layout=True)

        for (i, dname) in enumerate(self.data_names.keys()):

            xdata = getattr(self, self.data_names[dname])
            nslices = self.shape[-1]
            for j in range(0, nslices):
                xslice = xdata[..., j]
                mask_ok = np.where(~np.isnan(xslice) & ~np.isinf(xslice), True, False)
                (counts, edges) = np.histogram(xslice[mask_ok], bins=100)

                if (j == 0):
                    self.hist_edges[dname] = np.zeros((edges.size, nslices), dtype=np.float32)
                    self.hist_counts[dname] = np.zeros((counts.size, nslices), dtype=np.uint32)
                self.hist_edges[dname][:, j] = edges[:]
                self.hist_counts[dname][:, j] = counts[:]

                idx_mode = np.argmax(counts)
                axes[i].plot(edges[:-1], counts, label="Slice %i: Mode %.1f" % (j, round(edges[idx_mode], 1)))
            axes[i].legend(loc="upper right", fontsize="small")
            axes[i].set_xlabel(dname)
            

        fig.suptitle(title)
                
        return fig
                           

