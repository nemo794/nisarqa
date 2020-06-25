from quality import errors_derived
from quality import utility

from matplotlib import cm, pyplot, ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy import constants, fftpack

import copy
import traceback

class GCOVImage(object):

    BADVALUE = -9999
    EPS = 1.0e-03
    
    def __init__(self, band, frequency, polarization):

        self.band = band
        self.frequency = frequency
        self.polarization = polarization

        self.empty = False

    def initialize(self, data_in, nan_mask):

        self.xdata = np.copy(data_in)
        self.nan_mask = nan_mask
        self.shape = self.xdata.shape

        self.zero_mask = np.where(np.abs(self.xdata.real < self.EPS), True, False)
        self.mask_ok = np.where(~self.nan_mask & ~self.zero_mask, True, False)

    def read(self, handle, time_step=1, range_step=1):

        ximg = handle[self.frequency][self.polarization]
        self.xdata = ximg[::time_step, ::range_step]
        self.shape = self.xdata.shape

        #print("Read %s %s %s image" % (self.band, self.frequency, self.polarization))
            
    def check_for_nan(self):

        self.zero_mask = np.where( (self.xdata == 0.0), True, False)
        self.nan_mask = np.isnan(self.xdata) | np.isinf(self.xdata)
        self.num_nan = self.nan_mask.sum()
        self.perc_nan = 100.0*self.num_nan/self.xdata.size

        self.num_zero = 0
        self.empty = False
        
        if (self.num_nan == self.xdata.size):
            self.empty_string = ["%s %s_%s is entirely NaN" % (self.band, self.frequency, self.polarization)]
            self.empty = True
        elif (self.num_nan > 0) and (self.num_nan < self.xdata.size):
            self.nan_string = ["%s %s_%s has %i NaN's=%s%%" % (self.band, self.frequency, self.polarization, \
                                                              self.num_nan, round(self.perc_nan, 1))]

    def calc(self):

        try:
            self.mask_ok = np.where(~self.nan_mask & ~self.zero_mask, True, False)
        except AttributeError:
            print("Missing zero mask for image %s (%s)" % (self.frequency, self.polarization))
            raise AttributeError("Missing zero mask")
        
        self.power = np.where(self.mask_ok, self.xdata.real*self.xdata.real, self.BADVALUE)
        self.power = np.where(self.mask_ok, 10.0*np.log10(self.power), self.BADVALUE)
        (self.pcnt5, self.pcnt95) = np.percentile(self.xdata[~self.nan_mask], (5.0, 95.0))

        self.mean_backscatter = self.xdata[~self.nan_mask].mean()
        self.sdev_backscatter = self.xdata[~self.nan_mask].std()
        self.mean_power = self.power[~self.nan_mask].mean()
        self.sdev_power = self.power[~self.nan_mask].std()

        try:
            self.nnegative = np.where( (self.mask_ok) & (self.xdata <= 0.0), True, False).sum()            
            assert(self.nnegative == 0)
        except AssertionError as e:
            raise AssertionError()
            
    def plot4a(self, axis):

        # Compute histograms and plot them

        #print("%s power %f to %f" % (self.polarization, self.power[self.mask_ok].min(), \
        #                             self.power[self.mask_ok].max()))
        
        (counts1pr, edges1pr) = np.histogram(self.power[self.mask_ok], bins=100)
        idx_mode = np.argmax(counts1pr)
        self.modal_power = edges1pr[idx_mode]

        #print("%s edges %s, counts %s" % (self.polarization, edges1pr, counts1pr))
        (l, r) = axis.set_xlim(left=-200, right=10)
        #print("l %f, r %f" % (l, r))
        axis.plot(edges1pr[:-1], counts1pr, label="%s Mode %s" \
                  % (self.polarization, round(self.modal_power, 1)))
        (l, r) = axis.set_xlim(left=-200, right=10)
        #print("l %f, r %f" % (l, r))

        
        
    def plotfft(self, axis, title):

        axis.plot(self.frequency, 20.0*np.log10(self.avg_power), label="%s %s" \
                  % (self.frequency, self.polarization))

        fig.suptitle(title)

        return fig
        
        

        
