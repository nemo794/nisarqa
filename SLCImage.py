import utility

from matplotlib import cm, pyplot, ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy import constants, fftpack

import copy
import traceback

class SLCImage(object):

    BADVALUE = -9999
    
    def __init__(self, frequency, polarization, hertz, rspacing):

        self.frequency = frequency
        self.polarization = polarization
        self.hertz = hertz
        self.tspacing = (constants.c/2.0)/rspacing
        self.diffs = {}

    def initialize(self, data_in, nan_mask):

        self.xdata = np.copy(data_in)
        self.nan_mask = nan_mask
        self.shape = self.xdata.shape

        self.zero_mask = np.where( (self.xdata.real == 0.0) & (self.xdata.imag == 0.0), True, False)
        self.mask_ok = np.where(~self.nan_mask & ~self.zero_mask, True, False)
        
    def read(self, handle, time_step=1, range_step=1):

        self.complex32 = False
        ximg = handle[self.frequency][self.polarization]
        try:
            self.dtype = ximg.dtype
        except TypeError:
            self.complex32 = True

        if (self.complex32):
            with ximg.astype(np.complex64):
                self.xdata = ximg[::time_step, ::range_step]
        else:
            self.xdata = ximg[::time_step, ::range_step]

        self.shape = self.xdata.shape
            
    def check_for_nan(self):

        self.zero_mask = np.where( (self.xdata.real == 0.0) & (self.xdata.imag == 0.0), True, False)
        self.nan_mask = np.isnan(self.xdata.real) | np.isnan(self.xdata.imag) \
                      | np.isinf(self.xdata.real) | np.isinf(self.xdata.imag)
        self.num_nan = self.nan_mask.sum()
        self.perc_nan = 100.0*self.num_nan/self.xdata.size

        if (self.num_nan > 0):
            self.nan_string = "%s_%s had %i NaN's=%f%%" % (self.frequency, self.polarization, \
                                                           self.num_nan, 100.0*self.num_nan/self.xdata.size)

        self.empty = (self.num_nan == self.xdata.size)
        if (self.empty):
            return
        
        self.real = self.xdata.real[~self.nan_mask]
        self.imag = self.xdata.imag[~self.nan_mask]

        self.min = min(self.real.min(), self.imag.min())
        self.max = max(self.real.max(), self.imag.max())


    def calc(self):

        try:
            self.mask_ok = np.where(~self.nan_mask & ~self.zero_mask, True, False)
        except AttributeError:
            print("Missing zero mask for image %s (%s)" % (self.frequency, self.polarization))
            raise AttributeError("Missing zero mask")
        
        self.power = np.where(self.mask_ok, self.xdata.real*self.xdata.real + self.xdata.imag*self.xdata.imag, self.BADVALUE)
        self.power = np.where(self.mask_ok, 10.0*np.log10(self.power), self.BADVALUE)
        self.phase = np.where(self.mask_ok, np.degrees(np.angle(self.xdata)), self.BADVALUE)

        #idx_infinity = np.where(np.isinf(self.power))
        #print("%i Infinity Points: %s" % (len(idx_infinity[0]), idx_infinity))
        #print("Real for infinity %s, Imag for infinity %s" \
        #      % (np.unique(self.xdata.real[idx_infinity]), np.unique(self.xdata.imag[idx_infinity])))

        self.mean_power = self.power[self.mask_ok].mean()
        self.sdev_power = self.power[self.mask_ok].std()

        self.mean_phase = self.phase[self.mask_ok].mean()
        self.sdev_phase = self.phase[self.mask_ok].std()

        self.fft = np.fft.fft(self.xdata)
        self.fft_space = np.fft.fftfreq(self.shape[1], 1.0/self.tspacing)*1.0E-06
        self.avg_power = np.sum(np.abs(self.fft), axis=0)/(1.0*self.shape[0])

        idx = np.argsort(self.fft_space)
        self.fft_space = self.fft_space[idx]
        self.avg_power = self.avg_power[idx]
        
    def compare(self, image, frequency, polarization):

        key = "Frequency%s_%s" % (frequency, polarization)
        mask_nan = self.nan_mask | image.nan_mask | self.zero_mask | image.zero_mask
        self.diff[key] = SLCImage(frequency, polarization)

        xdata = self.xdata*image.data.conj()
        self.diff[key].initialize(mask_nan)

    def plot4a(self, title, bounds_linear, bounds_power):

        # Compute histograms and plot them
            
        (counts1r, edges1r) = np.histogram(self.real, range=bounds_linear, bins=100)
        (counts1c, edges1c) = np.histogram(self.imag, range=bounds_linear, bins=100)
        (counts1pr, edges1pr) = np.histogram(self.power, range=bounds_power, bins=100)
        (counts1ph, edges1ph) = np.histogram(self.phase, range=(-180.0, 180.0), bins=100)
            
        (fig, axes) = pyplot.subplots(nrows=2, ncols=2, sharex=False, sharey=False, constrained_layout=True)
        axes[0][0].plot(edges1r[:-1], counts1r, label="real")
        axes[0][1].plot(edges1c[:-1], counts1c, label="imaginary")
        axes[1][0].plot(edges1pr[:-1], counts1pr, label="power")
        axes[1][1].plot(edges1ph[:-1], counts1ph, label="phase")
        for (i, a) in enumerate(axes.flatten()):
            a.legend(loc="upper right", fontsize="small")
            a.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
            if (i%2 == 1):
                a.yaxis.tick_right()
                a.yaxis.set_label_position("right")
                a.set_ylim(bottom=0)
            a.set_ylabel("Number of Counts")
            if (i <= 1):
                a.set_xlabel("SLC (linear)")
            elif (i == 2):
                a.set_xlabel("SLC Power (dB)")
            else:
                a.set_xlabel("SLC Phase (degrees)")

        fig.suptitle(title)
        return fig
        
    def plot4b(self, axis, title):

        (counts, edges) = np.histogram(self.phase, range=(-180.0, 180.0), bins=100)
        axis.plot(edges[:-1], counts, label="phase")
        axis.set_title(title)
        #axis.legend(loc="upper right", fontsize="small")
        #axis.set_xlabel("SLC Phase (degrees)")
        #axis.set_ylabel("Number of Counts")

    def plotfft(self, axis, title):

        axis.plot(self.frequency, 20.0*np.log10(self.avg_power), label="%s %s" % (self.frequency, self.polarization))

        fig.suptitle(title)

        return fig
        
        

        
