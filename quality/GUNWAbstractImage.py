from quality import errors_derived
from quality import utility

from matplotlib import cm, colorbar, pyplot, ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy import constants, fftpack
from skimage import measure

import copy
import traceback

class GUNWAbstractImage(object):

    BADVALUE = -9999
    EPS = 1.0e-03
    
    def __init__(self, band, frequency, polarization, data_names):

        self.band = band
        self.frequency = frequency
        self.polarization = polarization

        self.empty = False
        self.data_names = data_names
        self.regions = {}

    def read(self, handle_img, handle_coords, xstep=1, ystep=1):

        self.handle_img = handle_img
        self.handle_coords = handle_coords
        self.has_coords = True
        try:
            self.xcoord = handle_coords["xCoordinates"][...]
            self.ycoord = handle_coords["yCoordinates"][...]
            assert(self.xcoord.ndim == 1)
            assert(self.ycoord.ndim == 1)
        except (KeyError, AssertionError):
            self.has_coords = False
        
        for dname in self.data_names.keys():
            xdata = handle_img[dname][::xstep, ::ystep]
            setattr(self, self.data_names[dname], xdata)
            
        xdata0 = getattr(self, list(self.data_names.values())[0])
        self.wrong_shape_inconsistent = []
        self.wrong_shape_coords = []

        for dname in self.data_names.keys():
            xdata = getattr(self, self.data_names[dname])
            if (xdata.shape != xdata0.shape):
                self.wrong_shape_inconsistent.append(dname)
            elif (self.has_coords) and (xdata.shape != (self.xcoord.size, self.ycoord.size)):
                 self.wrong_shape_coords.append(dname)

        self.correct_size = (len(self.wrong_shape_inconsistent) == 0) and \
                            (len(self.wrong_shape_coords) == 0)

        self.size = xdata0.size
        self.shape = xdata0.shape
        
        
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
        
        (fig_hist, axes) = pyplot.subplots(nrows=len(self.data_names.keys()), ncols=1, sharex=False, sharey=False, \
                                           constrained_layout=True)

        for (i, dname) in enumerate(self.data_names.keys()):

            xdata = getattr(self, self.data_names[dname])
            mask_ok = np.where(~np.isnan(xdata) & ~np.isinf(xdata) & (xdata != 0.0), True, False)
            (counts, edges) = np.histogram(xdata[mask_ok], bins=50)

            self.hist_edges[dname] = np.copy(edges)
            self.hist_counts[dname] = np.copy(counts)
            #print("%s: edges %s, counts %s" % (dname, edges, counts))

            idx_mode = np.argmax(counts)
            axes[i].plot(edges[:-1], counts, label="Mode %.1f" % (round(edges[idx_mode], 1)))
            axes[i].legend(loc="upper right", fontsize="small")
            axes[i].set_xlabel(dname)

        fig_hist.suptitle(title)

        return [fig_hist]

    def plot_region_map(self, title):

        if (not hasattr(self, "region_map")):
            return []
        
        # Plot region map
        
        short_name = self.data_names[self.region_dname]
        xdata = getattr(self, short_name)

        (fig, axes) = pyplot.subplots(nrows=2, ncols=1, sharex=True, sharey=True, \
                                      constrained_layout=True)

        img = axes[0].imshow(xdata, origin="upper", cmap=cm.jet)
        (caxis, kw) = colorbar.make_axes([axes[0]], location="right")
        fig.colorbar(img, cax=caxis, orientation="vertical")
        axes[0].set_title("%s image" % self.region_dname )

        img = axes[1].imshow(self.region_map, origin="upper", cmap=cm.jet)
        (caxis, kw) = colorbar.make_axes([axes[1]], location="right")
        fig.colorbar(img, cax=caxis, orientation="vertical")
        axes[1].set_title("%s regions" % self.region_dname)

        fig.suptitle(title)

        return [fig]

    def plot_region_hists(self, title):

        if (not hasattr(self, "region_map")):
            return []
        
        # Plot region histograms

        (fig, axes) = pyplot.subplots(nrows=2, ncols=1, sharex=False, sharey=False, \
                                      constrained_layout=True)          
        
        idx_id = self.idx_sorted_region_id
        (data_id, rsize) = self.region_size
        axes[0].plot(data_id[idx_id], rsize[idx_id], label=self.region_dname)

        (counts, edges) = np.histogram(rsize)
        axes[1].plot(edges[:-1], counts, label=self.region_dname)

        idx_size = self.idx_sorted_region_size
        self.region_size = (data_id[idx_size], rsize[idx_size])
        self.region_hist = (edges[:-1], counts)

        print("Regions (sorted)", self.region_size)

        axes[0].set_xlabel("Data Value")
        axes[0].set_ylabel("Region size (percent of total)")
        axes[1].set_xlabel("Region size (percent of total)")
        axes[1].set_ylabel("Number of counts")

        for a in axes:
            a.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            a.legend(loc="upper right", fontsize="small")
            a.legend(loc="upper right", fontsize="small")

        fig.suptitle(title)

        return [fig]
        
    def calc_connect(self):

        xdata = getattr(self, self.data_names[self.region_dname])

        self.nregions = np.unique(self.region_map).size
        self.connect_nonzero = 100.0*np.where(xdata > 0.0, True, False).sum()/xdata.size

        region_id = []
        data_id = []
        num = []
        self.region_error_list = []

        print("%s array has %s unique values" % (self.region_dname, np.unique(xdata)))
        for r in np.unique(self.region_map):
            mask = np.where(self.region_map == r, True, False)
            data_id = np.unique(xdata[mask])
            if (data_id.size > 1):
                self.region_error_list += ["%s %s %s (%s)" % (self.band, self.frequency, \
                                                              self.polarization, self.region_dname)]
                break
            elif (data_id[0] == self.region_fill):
                continue

            num_id = mask.sum()
            num.append(num_id)
            region_id.append(data_id[0])
            print("Region id %i has %i population and %s unique values" \
                  % (r, num_id, np.unique(xdata[mask])))

        assert(len(self.region_error_list) == 0)
            
        num_array = np.array(num)
        id_array = np.array(region_id).astype(np.int32)
        self.idx_sorted_region_size = np.argsort(-num_array)
        self.idx_sorted_region_id = np.argsort(id_array)
        self.region_size = (id_array, 100.0*num_array/xdata.size)
        print("%s Regions: %s" % (self.region_dname, self.region_size))

    def find_regions(self, dname, fill=0.0):

        self.region_dname = dname
        self.region_fill = fill
        data = getattr(self, self.data_names[dname])
        (nlines, nsmp) = data.shape

        self.empty_error_list = []
        if (np.all(data == fill)):
            self.empty_error_list += ["%s %s %s (%s)" % (self.band, self.frequency, \
                                                         self.polarization, dname)]

        assert(len(self.empty_error_list) == 0)

        regions = measure.label(data, connectivity=2)
        self.region_map = np.copy(regions)

        print("Unique data values %s, Unique regions %s" \
              % (np.unique(data), np.unique(regions)))
    
    def find_regions_broken(self, dname):

        data = getattr(self, self.data_names[dname])
        (nlines, nsmp) = data.shape
        regions = np.zeros(data.shape, dtype=np.uint8) 
        regions[0, 0] = 1

        radius = 1
        region_id = 1

        print("Finding %s regions for %s %s %s" % (dname, self.band, self.frequency, self.polarization))
        print("Unique values: %s" % np.unique(data))

        
        for iline in range(0, nlines):
            for ismp in range(0, nsmp):

                if ( (iline == 0) and (ismp == 0) ):
                    continue

                iline1 = max(iline - radius, 0)
                iline2 = min(iline1 + 2*radius + 1, nlines)
                ismp1 = max(ismp - radius, 0)
                ismp2 = min(ismp1 + 2*radius + 1, nsmp)
                window = data[iline1:iline2, ismp1:ismp2]
                xregion = regions[iline1:iline2, ismp1:ismp2]

                print("Iline %i, Ismp %i, Pixel %f, Window %s, Region %s, nregions %i" \
                      % (iline, ismp, data[iline, ismp], window, xregion, region_id))
                
                xdata = data[iline, ismp]
                mask = np.where( (window == xdata) & (xregion >= 1), True, False)
                if (np.any(mask)):
                    regions[iline, ismp] = xregion[mask][0]
                    print("  Adding pixel to region %i" % xregion[mask][0])
                else:
                    print("%i existing regions" % region_id)
                    region_id += 1
                    regions[iline, ismp] = region_id
                    print("Starting new region %i" % region_id)

        self.regions[dname] = np.copy(regions)
        
                
                    

