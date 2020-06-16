from quality import utility

from matplotlib import cm, pyplot, ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy import constants, fftpack

import copy
import traceback

class SLCImage(object):

    BADVALUE = -9999
    EPS = 1.0e-06
    
    def __init__(self, band, frequency, polarization, hertz, rspacing):

        self.band = band
        self.frequency = frequency
        self.polarization = polarization
        self.hertz = hertz
        self.tspacing = (constants.c/2.0)/rspacing
        self.diffs = {}
        self.empty = False

    def initialize(self, data_in, nan_mask):

        self.xdata = np.copy(data_in)
        self.nan_mask = nan_mask
        self.shape = self.xdata.shape

        #self.zero_real = np.where( (self.xdata.real == 0.0), True, False)
        #self.zero_imag = np.where( (self.xdata.imag == 0.0), True, False)
        self.zero_real = np.where(np.abs(self.xdata.real) < self.EPS, True, False)
        self.zero_imag = np.where(np.abs(self.xdata.imag) < self.EPS, True, False)
                                  
        self.zero_mask = np.where( self.zero_real & self.zero_imag, True, False)
        self.mask_ok = np.where(~self.nan_mask & ~self.zero_mask, True, False)

        #print("%s (%s) Nzero real elements %i=%f%%" \
        #      % (self.frequency, self.polarization, self.zero_real.sum(), \
        #         100.0*self.zero_real.sum()/self.xdata.size))
        #print("%s (%s) Nzero imag elements %i=%f%%" \
        #      % (self.frequency, self.polarization, self.zero_imag.sum(), \
        #         100.0*self.zero_imag.sum()/self.xdata.size))
        #print("%s (%s) Nzero both elements %i=%f%%" \
        #      % (self.frequency, self.polarization, self.zero_mask.sum(), \
        #         100.0*self.zero_mask.sum()/self.xdata.size))
        #print("\n")

        
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

        #print("Read %s %s %s image" % (self.band, self.frequency, self.polarization))
            
    def check_for_nan(self):

        #self.zero_mask = np.where( (self.xdata.real == 0.0) & (self.xdata.imag == 0.0), True, False)
        self.zero_real = np.where( np.fabs(self.xdata.real) < self.EPS, True, False)
        self.zero_imag = np.where( np.fabs(self.xdata.imag) < self.EPS, True, False)
        self.zero_mask = np.where( self.zero_real & self.zero_imag, True, False)

        self.nan_mask = np.isnan(self.xdata.real) | np.isnan(self.xdata.imag) \
                      | np.isinf(self.xdata.real) | np.isinf(self.xdata.imag)
        self.num_nan = self.nan_mask.sum()
        self.num_zero = self.zero_mask.sum()
        self.perc_nan = 100.0*self.num_nan/self.xdata.size
        self.perc_zero = 100.0*self.num_zero/self.xdata.size
        self.empty = (self.num_nan == self.xdata.size) or (self.num_zero == self.xdata.size) or \
                     ((self.num_nan + self.num_zero) == self.xdata.size)

        if (self.empty):
            if (self.num_nan == self.xdata.size):
                self.empty_string = ["%s %s_%s is entirely NaN" % (self.band, self.frequency, self.polarization)]
            elif (self.num_zero == self.xdata.size):
                self.empty_string = ["%s %s_%s is entirely Zeros" % (self.band, self.frequency, self.polarization)]
            else:
                self.empty_string = ["%s %s_%s is entirely NaNs or Zeros" % (self.band, self.frequency, self.polarization)]
            return
        
        if (self.num_nan > 0):
            self.nan_string = ["%s %s_%s has %i NaN's=%s%%" % (self.band, self.frequency, self.polarization, \
                                                               self.num_nan, round(self.perc_nan, 1))]
        if (self.num_zero > 0):
            self.zero_string = ["%s %s_%s has %i Zeros=%s%%" % (self.band, self.frequency, self.polarization, \
                                                                self.num_zero, round(self.perc_zero, 1))]

        self.real = self.xdata.real[~self.nan_mask]
        self.imag = self.xdata.imag[~self.nan_mask]

        self.min = min(self.real.min(), self.imag.min())
        self.max = max(self.real.max(), self.imag.max())


    def calc(self):

        import matplotlib.pyplot as pyplot
        
        try:
            self.mask_ok = np.where(~self.nan_mask & ~self.zero_mask, True, False)
        except AttributeError:
            print("Missing zero mask for image %s (%s)" % (self.frequency, self.polarization))
            raise AttributeError("Missing zero mask")
        
        self.power = np.where(self.mask_ok, self.xdata.real*self.xdata.real + self.xdata.imag*self.xdata.imag, self.BADVALUE)
        self.power = np.where(self.mask_ok, 10.0*np.log10(self.power), self.BADVALUE)
        self.phase = np.where(self.mask_ok, np.angle(self.xdata, deg=True), self.BADVALUE)

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

        #self.phase.astype(np.float32).tofile("%s_%s_%s.phase.bin" % (self.band, self.frequency, self.polarization))
        #print("%s %s %s: data[0, 2] %s, phase[0, 2] %s" % (self.band, self.frequency, self.polarization, \
        #                                                   self.xdata[0, 2], self.phase[0, 2]))
        #print("%s %s %s: phase %f to %f" % (self.band, self.frequency, self.polarization, \
        #                                    self.phase[self.mask_ok].min(), self.phase[self.mask_ok].max()))
        
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
        (counts1pr, edges1pr) = np.histogram(self.power[self.mask_ok], range=bounds_power, bins=100)
        (counts1ph, edges1ph) = np.histogram(self.phase[self.mask_ok], range=(-180.0, 180.0), bins=100)

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

    def plot4a1(self, axis):

        (counts, edges) = np.histogram(self.power[self.mask_ok], bins=100)
        axis.plot(edges[:-1], counts, label=self.polarization)
        
        axis.plot
        
    def plot4b(self, axis, title="", label=""):

        (counts, edges) = np.histogram(self.phase[self.mask_ok], range=(-180.0, 180.0), bins=100)
        axis.plot(edges[:-1], counts, label=label)
        axis.xaxis.set_tick_params(rotation=90)
        #axis.yaxis.set_tick_params(rotation=90)
        axis.set_title(title)

    def plotfft(self, axis, title):

        axis.plot(self.frequency, 20.0*np.log10(self.avg_power), label="%s %s" % (self.frequency, self.polarization))

        fig.suptitle(title)

        return fig
        
        

        
