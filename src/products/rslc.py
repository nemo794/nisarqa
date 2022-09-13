from dataclasses import dataclass, field
import h5py
import os

from typing import Any
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt

from param_files.QA_params import params
from utils import input_verification as iv
from utils import tiling
from utils import multilook as ml


class DataDecoder(object):
    """Wrapper to read in NISAR product datasets that are e.g. '<c4' type, 
    which raise an TypeError if accessed naively by h5py.

    Indexing operatations always return data converted to `type_to_read_data_as`.

    Notes
    -----
    Based on https://github-fn.jpl.nasa.gov/isce-3/isce/blob/develop/python/packages/nisar/products/readers/Raw/DataDecoder.py

    The DataDecoder class is an example of what the NumPy folks call a "duck array", 
    i.e. a class that exports some subset of np.ndarray's API so that it can be used 
    as a drop-in replacement for np.ndarray in some cases. This is different from an 
    "array_like" object, which is simply an object that can be converted to a numpy array. 
    Reference: https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
    """
    def __init__(self, h5dataset, type_to_read_data_as=np.dtype('complex64')):
        # TODO - turn these into properties
        self.dataset = h5dataset
        self.shape = self.dataset.shape
        self.ndim = self.dataset.ndim        
        self.dtype = type_to_read_data_as
        self.group = h5dataset.parent

        # Have h5py convert to the desired dtype on the fly when reading in data
        self.decoder = lambda key: self.dataset.astype(self.dtype)[key]

    def __getitem__(self, key):
        return self.decoder(key)


@dataclass
class RSLCRaster:
    """RSLC image dataset with mask."""
    # Raster data
    data: npt.ArrayLike

    # identifying name of this Raster; can be used for logging
    # e.g. "LSAR_A_HH"
    name: str

    # Mask of where the valid (True) and invalid (False)
    # pixels are in `data`.
    mask_ok: npt.ArrayLike = field(init=False)

    def __init__(self, h5dataset, name):
        """After the default initialition, create the mask_ok.
        """
        self.name = name
        self.data = DataDecoder(h5dataset, \
                                type_to_read_data_as=np.dtype('c8'))

        # print("Beginning Generate mask_ok for image: ", name)
        # self.mask_ok = \
        #         tiling.compute_mask_ok_by_tiling(self.data, max_tile_size=(1024, self.data.shape[1]))
        # print("Complete: Generate mask_ok for image: ", name)
        # print("Number of nan's: ", np.sum(~self.mask_ok))
        # print("Percentage of pixels are invalid: ", (np.sum(~self.mask_ok)/self.mask_ok.size) * 100)


def get_bands_freq_pols(h5_file):
    """
    TODO

    Parameters
    ----------
    h5_file : h5py file handle
        Handle to the input product h5 file

    Returns
    -------
    bands : dict of h5py Groups
        Dict of the h5py Groups for each band in `h5_file`,
        where the keys are the available bands (i.e. "SSAR" or "LSAR").
        Format: bands[<band>] -> a h5py Group
        Ex: bands['LSAR'] -> the h5py Group for LSAR

    freqs : dict of h5py Groups
        Dict of the h5py Groups for each freq in `h5_file`,
        where the keys are the available bands-freqs (i.e. "LSAR B" or "SSAR A").
        Format: freqs[<band>][<freq>] -> a h5py Group
        Ex: freqs['LSAR']['A'] -> the h5py Group for LSAR's FrequencyA

    pols : nested dict of RSLCRaster
        Nested dict of RSLCRaster objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRaster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a RSLCRaster object

    """

    bands = _get_bands(h5_file)
    freqs = _get_freqs(h5_file, bands)
    pols = _get_pols(h5_file, freqs)

    return bands, freqs, pols


def _get_bands(h5_file):
    """
    Finds the available bands in the input file
    and stores their paths in a nested dictionary.

    Parameters
    ----------
    h5_file : h5py file handle
        File handle to a valid NISAR RSLC hdf5 file.
        Bands must be located in the h5 file in the path: /science/<band>
        or they will not be found.

    Returns
    -------
    bands : dict of h5py Groups
        Dict of the h5py Groups for each band in `h5_file`,
        where the keys are the available bands (i.e. "SSAR" and/or "LSAR").
        Format: bands[<band>] -> a h5py Group
        Ex: bands['LSAR'] -> the h5py Group for LSAR
    """

    bands = {}
    for band in params.BANDS:
        path = f"/science/{band}"
        if path in h5_file:
            # self.logger.log_message(logging_base.LogFilterInfo, "Found band %s" % band)
            bands[band] = h5_file[path]
        else:
            # self.logger.log_message(logging_base.LogFilterInfo, "%s not present" % band)
            pass

    return bands


def _get_freqs(h5_file, bands):
    """
    Finds the available frequencies in the input file
    and stores their paths in a nested dictionary.

    Parameters
    ----------
    h5_file : h5py file handle
        File handle to a valid NISAR RSLC hdf5 file.
        Frequencies must be located in the h5 file in the path: 
        /science/<band>/RSLC/swaths/freqency<freq>
        or they will not be found.
    bands : list_like
        An iterable of the bands in `h5_file`.

    Returns
    -------
    freqs : dict of h5py Groups
        Dict of the h5py Groups for each freq in `h5_file`,
        where the keys are the available bands-freqs (i.e. "LSAR B" or "SSAR A").
        Format: freqs[<band>][<freq>] -> a h5py Group
        Ex: freqs['LSAR']['A'] -> the h5py Group for LSAR's FrequencyA

    See Also
    --------
    get_bands : function to generate the `bands` input argument
    """

    freqs = {}
    for band in bands.keys():
        freqs[band] = {}
        for freq in params.RSLC_FREQS:
            path = f"/science/{band}/RSLC/swaths/frequency{freq}"
            if path in h5_file:
                # self.logger.log_message(logging_base.LogFilterInfo, "Found band %s" % band)
                freqs[band][freq] = h5_file[path]

            # TODO - The original test datasets were created with only the "SLC"
            # filepath. New NISAR RSLC Products should only contain "RSLC" file paths.
            # Once the test datasets have been updated to "RSLC", then remove this
            # warning, and raise a fatal error.
            elif path.replace("RSLC", "SLC") in h5_file:
                freqs[band][freq] = h5_file[path.replace("RSLC", "SLC")]
                print("WARNING!! This product uses the deprecated `SLC` group. Update to `RSLC`.")
            else:
                # self.logger.log_message(logging_base.LogFilterInfo, "%s not present" % band)
                pass

    # Sanity Check - if a band does not have any frequencies, this is a validation error.
    # This check should be handled during the validation process before this function was called,
    # not the quality process, so raise an error.
    # In the future, this step might be moved earlier in the processing, and changed to
    # be handled via: "log the error and remove the band from the dictionary" 
    for band in freqs.keys():
        # Empty dictionaries evaluate to False in Python
        if not freqs[band]:
            raise ValueError(f"Provided input file's band {band} does not "
                              "contain any frequency groups.")

    return freqs


def _get_pols(h5_file, freqs):
    """
    Finds the available polarization rasters in the input file
    and stores their paths in a nested dictionary.

    Parameters
    ----------
    h5_file : h5py file handle
        File handle to a valid NISAR RSLC hdf5 file.
        frequencies must be located in the h5 file in the path: 
        /science/<band>/RSLC/swaths/freqency<freq>/<pol>
        or they will not be found.
    freqs : dict of h5py Groups
        Dict of the h5py Groups for each freq in `h5_file`,
        where the keys are the available bands-freqs (i.e. "LSAR B" or "SSAR A").
        Format: freqs[<band>][<freq>] -> a h5py Group
        Ex: freqs['LSAR']['A'] -> the h5py Group for LSAR's FrequencyA

    Returns
    -------
    pols : nested dict of RSLCRaster
        Nested dict of RSLCRaster objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRaster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a RSLCRaster object

    See Also
    --------
    get_freqs : function to generate the `freqs` input argument
    """

    pols = {}
    for band in freqs:
        pols[band] = {}
        for freq in freqs[band]:
            pols[band][freq] = {}
            for pol in params.RSLC_POLS:
                path = f"/science/{band}/RSLC/swaths/frequency{freq}/{pol}"
                if path in h5_file:
                    # self.logger.log_message(logging_base.LogFilterInfo, "Found band %s" % band)

                    band_freq_pol_str = f"{band}_{freq}_{pol}"
                    pols[band][freq][pol] = RSLCRaster(h5_file[path], \
                                                        name=band_freq_pol_str)

                # TODO - The original test datasets were created with only the "SLC"
                # filepath. New NISAR RSLC Products should only contain "RSLC" file paths.
                # Once the test datasets have been updated to "RSLC", then remove this
                # warning, and raise a fatal error.
                elif path.replace("RSLC", "SLC") in h5_file:
                    band_freq_pol_str = f"{band}_{freq}_{pol}"
                    pols[band][freq][pol] = RSLCRaster(h5_file[path.replace("RSLC", "SLC")], \
                                                        name=band_freq_pol_str)
                    print("WARNING!! This product uses the deprecated `SLC` group. Update to `RSLC`.")

                else:
                    # self.logger.log_message(logging_base.LogFilterInfo, "%s not present" % band)
                    pass

    # Sanity Check - if a band/freq does not have any polarizations, this is a validation error.
    # This check should be handled during the validation process before this function was called,
    # not the quality process, so raise an error.
    # In the future, this step might be moved earlier in the processing, and changed to
    # be handled via: "log the error and remove the band from the dictionary" 
    for band in pols.keys():
        for freq in pols[band].keys():
            # Empty dictionaries evaluate to False in Python
            if not pols[band][freq]:
                raise ValueError(f"Provided input file does not have any polarizations"
                            f"included under band {band}, frequency {freq}.")

    return pols


def process_power_image(pols, plots_pdf, 
            nlooks=None, linear_units=True, num_MPix=4.0, \
            highlight_inf_pixels=False, \
            middle_percentile=95.0, \
            browse_image_dir=".", browse_image_prefix=None):
    """
    Generate the RSLC Power Image plots for the `plots_pdf` and
    corresponding browse image products.

    The browse image products will follow this naming convention:
        <prefix>_<product name>_BAND_F_PP_qqq
            <prefix>        : `browse_image_prefix`, supplied from SDS
            <product name>  : RSLC, GLSC, etc.
            BAND            : LSAR or SSAR
            F               : frequency A or B 
            PP              : polarization
            qqq             : pow (because this function is for power images)

    Parameters
    ----------
    pols : nested dict of RSLCRaster
        Nested dict of RSLCRaster objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRaster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a RSLCRaster object
    plots_pdf : PdfPages object
        The output file to append the power image plot to
    nlooks : int or iterable of int
        Number of looks along each axis of the input array.
    linear_units : bool
        True to compute power in linear units, False for decibel units.
        Defaults to True.
    num_MPix : scalar
        The approx. size (in megapixels) for the final multilooked image.
        Defaults to 4.0 MPix.
    highlight_inf_pixels : bool
        True to color invalid pixels green in saved images.
        Defaults to black.
    middle_percentile : numeric
        Defines the middle percentile range of the `image_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 95.0.
    browse_image_dir : string
        Path to directory to save the browse image product.
    browse_image_prefix : string
        String to pre-pend to the name of the generated browse image product.
    """

    nlooks_arg = nlooks

    for band in pols:
        for freq in pols[band]:
            for pol in pols[band][freq]:

                img = pols[band][freq][pol]

                # print("Starting multilooking for Image: ", img.name)

                # Get the window size for multilooking
                if nlooks_arg is None: 
                    # From the xml Product Spec, sceneCenterAlongTrackSpacing is the 
                    # "Nominal along track spacing in meters between consecutive lines 
                    # near mid swath of the RSLC image."
                    az_spacing = img.data.group['sceneCenterAlongTrackSpacing'][...]

                    # From the xml Product Spec, sceneCenterGroundRangeSpacing is the 
                    # "Nominal ground range spacing in meters between consecutive pixels
                    # near mid swath of the RSLC image."
                    range_spacing = img.data.group['sceneCenterGroundRangeSpacing'][...]

                    print(f"\nImage {img.name}: ")
                    print("shape of image data: ", img.data.shape)
                    print("sceneCenterAlongTrackSpacing: ", az_spacing)
                    print("sceneCenterGroundRangeSpacing: ", range_spacing)

                    nlooks = ml.compute_square_pixel_nlooks(img.data.shape, \
                                                            sample_spacing=(az_spacing, range_spacing), \
                                                            num_MPix=num_MPix)
                    print("nlooks: ", nlooks)
                    num_pix = ((img.data.shape[0] // nlooks[0]) * (img.data.shape[1] // nlooks[1])) / 1e6
                    print("Estimated final size in MPix: ", num_pix)
                else:
                    nlooks = nlooks_arg

                # Multilook
                multilook_power_img = tiling.compute_multilooked_power_by_tiling(arr = img.data, \
                                                                                nlooks=nlooks, \
                                                                                linear_units=linear_units, \
                                                                                tile_shape=(1024,-1))
                
                print("shape of original data: ", img.data.shape)
                print("shape of multilook_power_img: ", multilook_power_img.shape)
                # print("multilook_power_img: \n", multilook_power_img)

                # Plot and Save Power Image as Browse Image Product
                browse_img_file = get_browse_product_filename(product_name="RSLC", \
                                                              band=band, \
                                                              freq=freq, \
                                                              pol=pol, \
                                                              quantity="pow", \
                                                              browse_image_dir=browse_image_dir, \
                                                              browse_image_prefix=browse_image_prefix)

                plot2png(img_arr=multilook_power_img,  \
                        filepath=browse_img_file, \
                        middle_percentile=middle_percentile, \
                        highlight_inf_pixels=highlight_inf_pixels, \
                        )

                # Plot and Save Power Image to graphical summary pdf
                if linear_units:
                    title=f"RSLC Multilooked Power (linear)\n{img.name}"
                else:
                    title=f"RSLC Multilooked Power (dB)\n{img.name}"

                plot2pdf(img_arr=multilook_power_img,  \
                        middle_percentile=middle_percentile, \
                        title=title, \
                        highlight_inf_pixels=highlight_inf_pixels, \
                        plots_pdf=plots_pdf
                        )


def plot2png(img_arr,  \
        filepath, \
        middle_percentile=95.0, \
        highlight_inf_pixels=False
        ):
    """
    Plot the clipped image array and save it to a browse image png.

    Parameters
    ----------
    img_arr : array_like
        Image to plot
    filepath : string
        Full filepath the browse image product.
    middle_percentile : numeric
        Defines the middle percentile range of the `image_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 95.0.
    highlight_inf_pixels : bool
        True to color pixels with an infinite value green in saved images.
        Defaults to matplotlib's default.
    """

    # Instantiate the figure object
    # (Need to instantiate it outside of the plotting function
    # in order to later modify the plot for saving purposes.)
    f = plt.figure()

    DPI = f.get_dpi()
    H = img_arr.shape[0]
    W = img_arr.shape[1]
    f.set_size_inches(w=float(W)/float(DPI), \
                        h=float(H)/float(DPI))

    # Get Plot
    f = plot_img_to_figure(fig=f, \
                         image_arr=img_arr, \
                         middle_percentile=middle_percentile, \
                         highlight_inf_pixels=highlight_inf_pixels)

    f.subplots_adjust(bottom=0.,left=0.,right=1.,top=1.)

    # Save plot to png (Browse Image Product)
    plt.axis('off')
    plt.savefig(filepath, \
                bbox_inches='tight', pad_inches=0, \
                dpi=DPI
                )

    plt.close()


def get_browse_product_filename(product_name, band, freq, pol, quantity, \
            browse_image_dir, browse_image_prefix=None):
    """
    Returns the full filename (with path) for Browse Image Product.

    The browse image products should follow this naming convention,
    (Convention modified from Phil Callahan's slides on 11 Aug 2022.)
        <prefix>_<product name>_BAND_F_PP[PP]_qqq
            <prefix>        : browse image prefix, supplied by SDS
            <product name>  : RSLC, GLSC, etc.
            BAND            : LSAR or SSAR
            F               : frequency A or B 
            PP              : polarization, e.g. "HH" or "HV".
                              [PP] additional polarization for GCOV 
            qqq             : quantity: mag, phs, coh, cov, rof, aof, cnc, iph 
                                        (see product list)
    """
    filename = f"{product_name.upper()}_{band}_{freq}_{pol}_{quantity}.png"
    if browse_image_prefix is not None:
        filename = f"{browse_image_prefix}_{filename}"
    filename = os.path.join(browse_image_dir, filename)

    return filename


def plot2pdf(img_arr,  \
        plots_pdf, \
        title="", \
        middle_percentile=95.0, \
        highlight_inf_pixels=False
        ):
    """
    Plot the clipped image array and append it to the pdf.

    Parameters
    ----------
    img_arr : array_like
        Image to plot
    plots_pdf : PdfPages object
        The output pdf file to append the power image plot to
    title : string
        The full title for the plot
    middle_percentile : numeric
        Defines the middle percentile range of the `image_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 95.0.
    highlight_inf_pixels : bool
        True to color pixels with an infinite value green in saved images.
        Defaults to matplotlib's default.
    """

    # Instantiate the figure object
    # (Need to instantiate it outside of the plotting function
    # in order to later modify the plot for saving purposes.)
    f = plt.figure()

    # Get Plot
    f = plot_img_to_figure(fig=f, \
                         image_arr=img_arr, \
                         middle_percentile=middle_percentile, \
                         highlight_inf_pixels=highlight_inf_pixels)

    plt.colorbar(ax=plt.gca())
    plt.title(title)
    plots_pdf.savefig(plt.gcf())

    # Close the plot
    plt.close()


def plot_img_to_figure(fig, image_arr, \
                            highlight_inf_pixels, \
                            middle_percentile=95.0):
    """
    Clip and plot `image_arr` onto `fig` and return that figure.

    For example, this function can be used to plot the power image
    for an RSLC product.

    Parameters
    ----------
    fig : matplotlib Figure
        The figure object to plot the image on.
    image_arr : array_like
        The image data, such as matches matplotlib.plt.imshow's
        specifications for `X`
    highlight_inf_pixels : bool
        True to color pixels with an infinite value green in saved images.
        Defaults to matplotlib's default.
    middle_percentile : numeric
        Defines the middle percentile range of the `image_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 95.0.

    Returns
    -------
    fig_out : matplotlib Figure
        `image_arr` clipped to the `middle_percentile` and plotted on `fig`

    Notes
    -----
    1) In this function, the `image_arr` will be manually clipped
    before being passed to plt.imshow().
    While plt.imshow() can do the clipping automatically if the
    vmin and vmax values are passed in, in practise, doing so 
    causes the resultant size of the output .pdf files that contain 
    these figures to grow from e.g. 537KB to 877MB.
    A workaround is to clip the image data in advance.
    2) The interpolation method is imshow()'s default of antialiasing.
    Setting interpolation='none' causes the size of the output
    .pdf files that contain these figures to grow from e.g. 537KB to 877MB.
    """

    iv.verify_valid_percentile(middle_percentile)

    # Get vmin and vmax to set the desired range of the colorbar
    vmin, vmax = calc_vmin_vmax(image_arr, percent=middle_percentile)

    # Manually clip the image data (See `Notes` in function description)
    clipped_array = np.clip(image_arr, a_min = vmin, a_max = vmax)

    # TODO (Sam) - saving the clipped image data to an array will (temporarily)
    # use another big chunk of memory. Revisit this code later if/when this
    # becomes an issue.

    fig.add_subplot(1,1,1) #  add a subplot to fig

    # Highlight infinite pixels, if requested.
    cmap=plt.cm.gray
    if highlight_inf_pixels:
        cmap.set_bad('g')

    # Plot the image_arr image.
    plt.imshow(X=clipped_array, \
                # Place the [0, 0] index of the array in the upper left corner of the Axes.
                # origin="upper", \
                cmap=cmap, \
                )

    return fig


def calc_vmin_vmax(data_in, middle_percentile=95.0):
    """Calculate the values of vmin and vmax for the 
    input array using the given middle percentile.

    For example, if `middle_percentile` is 95.0, then this will
    return the value of the 2.5th quantile and the 97.5th quantile.

    Parameters
    ----------
    data_in : array_like
        Input array
    middle_percentile : numeric
        Defines the middle percentile range of the `image_arr`. 
        Must be in the range [0, 100].
        Defaults to 95.0.

    Returns
    -------
    vmin, vmax : numeric
        The lower and upper values (respectively) of the middle 
        percentile.

    """
    iv.verify_valid_percentile(middle_percentile)

    fraction = 0.5*(1.0 - middle_percentile/100.0)

    # Get the value of the e.g. 2.5th quantile and the 97.5th quantile
    vmin, vmax = np.quantile(data_in, [fraction, 1-fraction])

    return vmin, vmax

