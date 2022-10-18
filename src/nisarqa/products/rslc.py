from dataclasses import dataclass, field
import h5py
import os
import time

from typing import Any
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import nisarqa


@dataclass
class QAPlotsAndMetricsParams:
    '''
    Data structure to hold the parameters for the
    QA code's output plots and statistics files that are
    common to all NISAR products.

    TODO - add the `stats_h5` attribute. (This will come
    with the RSLC histogram PR.)

    Parameters
    ----------
    plots_pdf : PdfPages
        The output file to append the power image plot to
    browse_image_dir : str
        Path to directory to save the browse image product.
    browse_image_prefix : str
        String to pre-pend to the name of the generated browse image product.
    tile_shape : tuple of ints
        Shape of each tile to be processed. If `tile_shape` is
        larger than the shape of `arr`, or if the dimensions of `arr`
        are not integer multiples of the dimensions of `tile_shape`,
        then smaller tiles may be used.
        -1 to use all rows / all columns (respectively).
        Format: (num_rows, num_cols) 
        Defaults to (512, -1) to use all columns (i.e. full rows of data)
        and leverage Python's row-major ordering.
    '''

    # Attributes that are common to all NISAR Products
    plots_pdf: PdfPages
    browse_image_dir: str = '.'
    browse_image_prefix: str = None
    tile_shape: tuple = (512,-1)


@dataclass
class QAPlotsAndMetricsParamsRSLC(QAPlotsAndMetricsParams):
    '''
    Data structure to hold the parameters for the QA
    code's output plots and statistics files for RSLC.

    Parameters
    ----------
    plots_pdf : PdfPages
        The output file to append the power image plot to
    browse_image_dir : str
        Path to directory to save the browse image product.
    browse_image_prefix : str
        String to pre-pend to the name of the generated browse image product.
    tile_shape : tuple of ints
        Shape of each tile to be processed. If `tile_shape` is
        larger than the shape of `arr`, or if the dimensions of `arr`
        are not integer multiples of the dimensions of `tile_shape`,
        then smaller tiles may be used.
        -1 to use all rows / all columns (respectively).
        Format: (num_rows, num_cols) 
        Defaults to (512, -1) to use all columns (i.e. full rows of data)
        and leverage Python's row-major ordering.
    nlooks_freqa, nlooks_freqb : int or iterable of int
        Number of looks along each axis of the input array 
        for the specified frequency.
    linear_units : bool
        True to compute power in linear units, False for decibel units.
        Defaults to True.
    num_mpix : scalar
        The approx. size (in megapixels) for the final multilooked image.
        Defaults to 4.0 MPix.
    highlight_inf_pixels : bool
        True to color invalid pixels green in saved images.
        Defaults to black.
    middle_percentile : numeric
        Defines the middle percentile range of the `img_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 100.0.
    '''

    # Attributes that are specific to the RSLC product
    nlooks_freqa: int = None
    nlooks_freqb: int = None 
    linear_units: bool = True
    num_mpix: int = 4.0
    highlight_inf_pixels: bool = False
    middle_percentile: int = 100.0


class ComplexFloat16Decoder(object):
    '''Wrapper to read in NISAR product datasets that are '<c4' type,
    which raise an TypeError if accessed naively by h5py.

    Indexing operatations always return data converted to numpy.complex64.

    Parameters
    ----------
    h5dataset : h5py.Dataset
        Dataset to be stored. Dataset should have type '<c4'.

    Notes
    -----
    The ComplexFloat16Decoder class is an example of what the NumPy folks call a 'duck array', 
    i.e. a class that exports some subset of np.ndarray's API so that it can be used 
    as a drop-in replacement for np.ndarray in some cases. This is different from an 
    'array_like' object, which is simply an object that can be converted to a numpy array. 
    Reference: https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
    '''

    def __init__(self, h5dataset):
        self._dataset = h5dataset
        self._dtype = np.complex64

    def __getitem__(self, key):
        # Have h5py convert to the desired dtype on the fly when reading in data
        return self.read_c4_dataset_as_c8(self.dataset, key)

    @staticmethod
    def read_c4_dataset_as_c8(ds: h5py.Dataset, key=np.s_[...]):
        """
        Read a complex float16 HDF5 dataset as a numpy.complex64 array.
        Avoids h5py/numpy dtype bugs and uses numpy float16 -> float32 conversions
        which are about 10x faster than HDF5 ones.
        """
        # This avoids h5py exception:
        # TypeError: data type '<c4' not understood
        # Also note this syntax changed in h5py 3.0 and was deprecated in 3.6, see
        # https://docs.h5py.org/en/stable/whatsnew/3.6.html

        complex32 = np.dtype([('r', np.float16), ('i', np.float16)])
        z = ds.astype(complex32)[key]

        # Define a similar datatype for complex64 to be sure we cast safely.
        complex64 = np.dtype([("r", np.float32), ("i", np.float32)])

        # Cast safely and then view as native complex64 numpy dtype.
        return z.astype(complex64).view(np.complex64)


    @property
    def dataset(self):
        return self._dataset

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self.dataset.shape

    @property
    def ndim(self):
        return self.dataset.ndim


@dataclass
class RSLCRaster:
    '''
    RSLC image dataset with name.
    
    Parameters
    ----------
    data : array_like
        Raster data to be stored.
    name : str
        Name for the dataset

    Notes
    -----
    If data is an HDF5 dataset, suggest initializing using
    the class method `from_h5dataset(..)`.
    '''

    # Raster data
    data: npt.ArrayLike

    # identifying name of this Raster; can be used for logging
    # e.g. 'LSAR_A_HH'
    name: str


    @classmethod
    def from_h5dataset(cls, h5dataset, name, dtype):
        '''
        Initialize an RSLCRaster object for a HDF5 dataset
        that needs to be decoded via a specific dtype.

        This will store the dataset as a ComplexFloat16Decoder
        object instead of a standard Arraylike object.

        Parameters
        ----------
        h5dataset : h5py.Dataset
            Raster data to be stored.
        name : str
            Name for the dataset
        '''
        data = ComplexFloat16Decoder(h5dataset)
        return cls(data, name)


@dataclass
class RSLCRasterQA(RSLCRaster):
    '''
    An RSLCRaster with additional attributes specific to the QA Code.
    
    Parameters
    ----------
    data : array_like
        Raster data to be stored.
    name : str
        Name for the dataset
    band : str
        name of the band for `img`, e.g. 'LSAR'
    freq : str
        name of the frequency for `img`, e.g. 'A' or 'B'
    pol : str
        name of the polarization for `img`, e.g. 'HH' or 'HV'
    az_spacing : float
        Azimuth spacing of pixels of input array
    az_start : float
        The start time of the observation for this
        RSLC Raster
    az_stop : float
        The stopping time of the observation for this
        RSLC Raster
    range_spacing : float
        Range spacing of pixels of input array
    rng_start : float
        Start (near) distance of the range of input array
    rng_stop : float
        End (far) distance of the range of input array
    epoch_starttime : str
        The start of the epoch for this observation,
        in the format 'YYYY-MM-DD HH:MM:SS'

    Notes
    -----
    If data is an HDF5 dataset, suggest initializing using
    the class method `from_h5dataset(..)`.
    '''

    # Attributes of the input array
    band: str
    freq: str
    pol: str

    az_spacing: float
    az_start: float
    az_stop: float

    range_spacing: float
    rng_start: float
    rng_stop: float

    epoch_starttime: str


    @classmethod
    def from_h5dataset(cls,
                       h5dataset, band, freq, pol, dtype,
                       az_spacing, az_start, az_stop,
                       range_spacing, rng_start, rng_stop,
                       epoch_starttime):
        '''
        Initialize an RSLCRasterQA object for a HDF5 dataset
        that needs to be decoded via a specific dtype.

        This will store the dataset as a ComplexFloat16Decoder
        object instead of a standard Arraylike object.

        Parameters
        ----------
        h5dataset : h5py.Dataset
            Raster data to be stored.
        band : str
            name of the band for `img`, e.g. 'LSAR'
        freq : str
            name of the frequency for `img`, e.g. 'A' or 'B'
        pol : str
            name of the polarization for `img`, e.g. 'HH' or 'HV'
        az_spacing : float
            Azimuth spacing of pixels of input array
        az_start : float
            The start time of the observation for this
            RSLC Raster
        az_stop : float
            The stopping time of the observation for this
            RSLC Raster
        range_spacing : float
            Range spacing of pixels of input array
        rng_start : float
            Start (near) distance of the range of input array
        rng_stop : float
            End (far) distance of the range of input array
        epoch_starttime : str
            The start of the epoch for this observation,
            in the format 'YYYY-MM-DD HH:MM:SS'

        Notes
        -----
        Unlike the default constructor for RSLCRasterQA, this
        function does not include a user-specified input 
        parameter `name`. Instead, to maintain consistency of
        the format of the `name` for each NISAR RSLC QA image,
        the `name` attribute will be populated with a string
        of the format: <band>_<freq>_<pol>
        '''
        data = ComplexFloat16Decoder(h5dataset)

        # Format the name
        name = f"{band}_{freq}_{pol}"

        # Note the order of the positional arguments being passed;
        # the attributes of the parent class must be passed first.
        return cls(data, name, band, freq, pol, 
                   az_spacing, az_start, az_stop,
                   range_spacing, rng_start, rng_stop,
                   epoch_starttime)


def get_bands_freqs_pols(h5_file):
    '''
    Locate the available bands, frequencies, and polarizations
    in the input HDF5 file.

    Parameters
    ----------
    h5_file : h5py file handle
        Handle to the input product h5 file

    Returns
    -------
    bands : dict of h5py Groups
        Dict of the h5py Groups for each band in `h5_file`,
        where the keys are the available bands (i.e. 'SSAR' or 'LSAR').
        Format: bands[<band>] -> a h5py Group
        Ex: bands['LSAR'] -> the h5py Group for LSAR
    freqs : dict of h5py Groups
        Dict of the h5py Groups for each freq in `h5_file`,
        where the keys are the available bands-freqs (i.e. 'LSAR B' or 'SSAR A').
        Format: freqs[<band>][<freq>] -> a h5py Group
        Ex: freqs['LSAR']['A'] -> the h5py Group for LSAR's FrequencyA
    pols : nested dict of RSLCRasterQA
        Nested dict of RSLCRasterQA objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRasterQA
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a RSLCRasterQA object

    '''

    bands = _get_bands(h5_file)
    freqs = _get_freqs(h5_file, bands)
    pols = _get_pols(h5_file, freqs)

    return bands, freqs, pols


def _get_bands(h5_file):
    '''
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
        where the keys are the available bands (i.e. 'SSAR' and/or 'LSAR').
        Format: bands[<band>] -> a h5py Group
        Ex: bands['LSAR'] -> the h5py Group for LSAR
    '''

    bands = {}
    for band in nisarqa.BANDS:
        path = f'/science/{band}'
        if path in h5_file:
            # self.logger.log_message(logging_base.LogFilterInfo, 'Found band %s' % band)
            bands[band] = h5_file[path]
        else:
            # self.logger.log_message(logging_base.LogFilterInfo, '%s not present' % band)
            pass

    return bands


def _get_freqs(h5_file, bands):
    '''
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
    freqs : nested dict of h5py Groups
        Dict of the h5py Groups for each freq in `h5_file`,
        where the keys are the available bands-freqs (i.e. 'LSAR B' or 'SSAR A').
        Format: freqs[<band>][<freq>] -> a h5py Group
        Ex: freqs['LSAR']['A'] -> the h5py Group for LSAR's FrequencyA

    See Also
    --------
    get_bands : function to generate the `bands` input argument
    '''

    freqs = {}
    for band in bands.keys():
        freqs[band] = {}
        for freq in nisarqa.RSLC_FREQS:
            path = f'/science/{band}/RSLC/swaths/frequency{freq}'
            if path in h5_file:
                # self.logger.log_message(logging_base.LogFilterInfo, 'Found band %s' % band)
                freqs[band][freq] = h5_file[path]

            # TODO - The original test datasets were created with only the 'SLC'
            # filepath. New NISAR RSLC Products should only contain 'RSLC' file paths.
            # Once the test datasets have been updated to 'RSLC', then remove this
            # warning, and raise a fatal error.
            elif path.replace('RSLC', 'SLC') in h5_file:
                freqs[band][freq] = h5_file[path.replace('RSLC', 'SLC')]
                print('WARNING!! This product uses the deprecated `SLC` group. Update to `RSLC`.')
            else:
                # self.logger.log_message(logging_base.LogFilterInfo, '%s not present' % band)
                pass

    # Sanity Check - if a band does not have any frequencies, this is a validation error.
    # This check should be handled during the validation process before this function was called,
    # not the quality process, so raise an error.
    # In the future, this step might be moved earlier in the processing, and changed to
    # be handled via: 'log the error and remove the band from the dictionary' 
    for band in freqs.keys():
        # Empty dictionaries evaluate to False in Python
        if not freqs[band]:
            raise ValueError(f"Provided input file's band {band} does not "
                              "contain any frequency groups.")

    return freqs


def _get_pols(h5_file, freqs):
    '''
    Finds the available polarization rasters in the input file
    and stores their paths in a nested dictionary.

    Parameters
    ----------
    h5_file : h5py file handle
        File handle to a valid NISAR RSLC hdf5 file.
        frequencies must be located in the h5 file in the path: 
        /science/<band>/RSLC/swaths/freqency<freq>/<pol>
        or they will not be found.
    freqs : nested dict of h5py Groups
        Dict of the h5py Groups for each freq in `h5_file`,
        where the keys are the available bands-freqs (i.e. 'LSAR B' or 'SSAR A').
        Format: freqs[<band>][<freq>] -> a h5py Group
        Ex: freqs['LSAR']['A'] -> the h5py Group for LSAR's FrequencyA

    Returns
    -------
    pols : nested dict of RSLCRasterQA
        Nested dict of RSLCRasterQA objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRasterQA
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored in a RSLCRasterQA object

    See Also
    --------
    get_freqs : function to generate the `freqs` input argument
    '''

    # Discover images in dataset and populate the dictionary
    pols = {}
    for band in freqs:
        pols[band] = {}
        for freq in freqs[band]:
            pols[band][freq] = {}
            for pol in nisarqa.RSLC_POLS:

                tmp_RSLCRasterQA = create_RSLCRasterQA(h5_file, band, freq, pol)

                if isinstance(tmp_RSLCRasterQA, RSLCRasterQA):
                    pols[band][freq][pol] = tmp_RSLCRasterQA

    # Sanity Check - if a band/freq does not have any polarizations, 
    # this is a validation error. This check should be handled during 
    # the validation process before this function was called,
    # not the quality process, so raise an error.
    # In the future, this step might be moved earlier in the 
    # processing, and changed to be handled via: 'log the error 
    # and remove the band from the dictionary' 
    for band in pols.keys():
        for freq in pols[band].keys():
            # Empty dictionaries evaluate to False in Python
            if not pols[band][freq]:
                raise ValueError(f'Provided input file does not have any polarizations'
                            f' included under band {band}, frequency {freq}.')

    return pols


def create_RSLCRasterQA(h5_file, band, freq, pol):
    '''
    Return a RSLCRasterQA instance of the given band-freq-pol
    image in the input HDF5 file, if that image exists.

    Parameters
    ----------
    h5_file : h5py file handle
        File handle to a valid NISAR RSLC hdf5 file.
        Polarization images must be located in the h5 file in the path: 
        /science/<band>/RSLC/swaths/freqency<freq>/<pol>
        or they will not be found. This is the file structure
        as determined from the NISAR Product Spec.
    band : str
        name of the band for `img`, e.g. 'LSAR'
    freq : str
        name of the frequency for `img`, e.g. 'A' or 'B'
    pol : str
        name of the polarization for `img`, e.g. 'HH' or 'HV'

    Returns
    -------
    rslc_raster_image : RSLCRasterQA
        An instance of RSLCRasterQA representing the given
        band-freq-pol image in the input HDF5 file.
        If the image does not exist, None will be returned.

    '''
    # check if this is an RSLC or and SLC file.
    if f'/science/{band}/RSLC' in h5_file:
        slc_type = 'RSLC'
    elif f'/science/{band}/SLC' in h5_file:
        # TODO - The UAVSAR test datasets were created with only the 'SLC'
        # filepath. New NISAR RSLC Products should only contain 'RSLC' file paths.
        # Once the test datasets have been updated to 'RSLC', then remove this
        # warning, and raise a fatal error.
        print('WARNING!! This product uses the deprecated `SLC` group. Update to `RSLC`.')

        slc_type = 'SLC'
    else:
        # self.logger.log_message(logging_base.LogFilterError, 'Invalid file structure.')
        return None

    # Hardcoded paths to various groups in the NISAR RSLC h5 file.
    # These paths are determined by the RSLC .xml product spec
    swaths_path = f'/science/{band}/{slc_type}/swaths'
    freq_path = f'{swaths_path}/frequency{freq}/'
    pol_path = f'{freq_path}/{pol}'
    band_freq_pol_str = f'{band}_{freq}_{pol}'

    if pol_path in h5_file:
        # self.logger.log_message(logging_base.LogFilterInfo, 
        #                         'Found image %s' % band_freq_pol_str)
        pass
    else:
        # self.logger.log_message(logging_base.LogFilterInfo, 
        #                         'Image %s not present' % band_freq_pol_str)
        return None

    # From the xml Product Spec, sceneCenterAlongTrackSpacing is the 
    # 'Nominal along track spacing in meters between consecutive lines 
    # near mid swath of the RSLC image.'
    az_spacing = h5_file[freq_path]['sceneCenterAlongTrackSpacing'][...]

    # Get Azimuth (y-axis) tick range + label
    # path in h5 file: /science/LSAR/RSLC/swaths/zeroDopplerTime
    az_start = float(h5_file[swaths_path]['zeroDopplerTime'][0])
    az_stop =  float(h5_file[swaths_path]['zeroDopplerTime'][-1])

    # From the xml Product Spec, sceneCenterGroundRangeSpacing is the 
    # 'Nominal ground range spacing in meters between consecutive pixels
    # near mid swath of the RSLC image.'
    range_spacing = h5_file[freq_path]['sceneCenterGroundRangeSpacing'][...]

    # Range in meters (units are specified as meters in the product spec)
    rng_start = float(h5_file[freq_path]['slantRange'][0])
    rng_stop = float(h5_file[freq_path]['slantRange'][-1])

    # output of the next line will have the format: 'seconds since YYYY-MM-DD HH:MM:SS'
    sec_since_epoch = h5_file[swaths_path]['zeroDopplerTime'].attrs['units'].decode('utf-8')
    epoch_starttime = sec_since_epoch.replace('seconds since ', '').strip()

    return RSLCRasterQA.from_h5dataset(h5_file[pol_path],
                                     band=band,
                                     freq=freq,
                                     pol=pol,
                                     dtype=np.complex64,
                                     az_spacing=az_spacing,
                                     az_start=az_start,
                                     az_stop=az_stop,
                                     range_spacing=range_spacing,
                                     rng_start=rng_start,
                                     rng_stop=rng_stop,
                                     epoch_starttime=epoch_starttime)


def process_power_images(pols, params):
    '''
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

    TODO - double check the file naming convention

    Parameters
    ----------
    pols : nested dict of RSLCRasterQA
        Nested dict of RSLCRasterQA objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRasterQA
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored 
                                       in a RSLCRasterQA object
    params : RSLCGraphsMetricsParams
        A structure containing the parameters for processing
        and outputting the power image(s).
    '''
    # Process each image in the dataset
    for band in pols:
        for freq in pols[band]:
            for pol in pols[band][freq]:
                img = pols[band][freq][pol]

                process_single_power_image(img, params)


def process_single_power_image(img, params):
    '''
    Generate the RSLC Power Image plots for the `plots_pdf` and
    corresponding browse image products for a single RSLC image.

    The browse image products will follow this naming convention:
        <prefix>_<product name>_BAND_F_PP_qqq
            <prefix>        : `browse_image_prefix`, supplied from SDS
            <product name>  : RSLC, GLSC, etc.
            BAND            : LSAR or SSAR
            F               : frequency A or B 
            PP              : polarization
            qqq             : pow (because this function is for power images)

    TODO - double check the file naming convention

    Parameters
    ----------
    img : RSLCRasterQA
        The RSLC raster to be processed
    params : RSLCGraphsMetricsParams
        A structure containing the parameters for processing
        and outputting the power image(s).
    '''

    nlooks_freqa_arg = params.nlooks_freqa
    nlooks_freqb_arg = params.nlooks_freqb

    # Get the window size for multilooking
    if (img.freq == 'A' and nlooks_freqa_arg is None) or \
        (img.freq == 'B' and nlooks_freqb_arg is None):

        nlooks = nisarqa.compute_square_pixel_nlooks(
                    img.data.shape,
                    sample_spacing=(img.az_spacing, img.range_spacing),
                    num_mpix=params.num_mpix)

    elif img.freq == 'A':
        nlooks = nlooks_freqa_arg
    elif img.freq == 'B':
        nlooks = nlooks_freqb_arg
    else:
        raise ValueError(f'freqency is {freq}, but only `A` or `B` are valid options.')

    print(f'\nMultilooking Image {img.name} with shape: {img.data.shape}')
    print('sceneCenterAlongTrackSpacing: ', img.az_spacing)
    print('sceneCenterGroundRangeSpacing: ', img.range_spacing)
    print('Beginning Multilooking with nlooks window shape: ', nlooks)

    # Multilook
    print('tile_shape: ', params.tile_shape)
    start_time = time.time()
    multilook_power_img = nisarqa.compute_multilooked_power_by_tiling(
                                            arr=img.data,
                                            nlooks=nlooks,
                                            linear_units=params.linear_units,
                                            tile_shape=params.tile_shape)
    end_time = time.time()-start_time
    print('time to multilook image (sec): ', end_time)
    print('time to multilook image (min): ', end_time/60.)

    print(f'Multilooking Complete. Multilooked shape: {multilook_power_img.shape}')
    print(f'Multilooked size: {multilook_power_img.size} Mpix.')

    # Plot and Save Power Image as Browse Image Product
    browse_img_file = get_browse_product_filename(
                                product_name='RSLC',
                                band=img.band,
                                freq=img.freq,
                                pol=img.pol,
                                quantity='pow',
                                browse_image_dir=params.browse_image_dir,
                                browse_image_prefix=params.browse_image_prefix)

    plot2png(img_arr=multilook_power_img,
                filepath=browse_img_file,
                middle_percentile=params.middle_percentile,
                highlight_inf_pixels=params.highlight_inf_pixels,
                )

    # Plot and Save Power Image to graphical summary pdf
    if params.linear_units:
        title=f'RSLC Multilooked Power (linear)\n{img.name}'
    else:
        title=f'RSLC Multilooked Power (dB)\n{img.name}'

    # Get Azimuth (y-axis) label
    az_title = f"Zero Doppler Time\n(seconds since {img.epoch_starttime})"

    # Get Range (x-axis) labels and scale
    # Convert range from meters to km
    rng_start_km = img.rng_start/1000.
    rng_stop_km = img.rng_stop/1000.
    rng_title = 'Slant Range (km)'

    plot2pdf(img_arr=multilook_power_img,
                middle_percentile=params.middle_percentile,
                title=title,
                ylim=[img.az_start, img.az_stop],
                xlim=[rng_start_km, rng_stop_km],
                ylabel=az_title,
                xlabel=rng_title,
                highlight_inf_pixels=params.highlight_inf_pixels,
                plots_pdf=params.plots_pdf
                )


def plot2png(img_arr,
             filepath,
             middle_percentile=100.0,
             highlight_inf_pixels=False
             ):
    '''
    Plot the clipped image array and save it to a browse image png.

    Parameters
    ----------
    img_arr : array_like
        Image to plot
    filepath : str
        Full filepath the browse image product.
    middle_percentile : numeric
        Defines the middle percentile range of the `img_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 100.0.
    highlight_inf_pixels : bool
        True to color pixels with an infinite value green in saved images.
        Defaults to matplotlib's default.
    '''

    # Instantiate the figure object
    # (Need to instantiate it outside of the plotting function
    # in order to later modify the plot for saving purposes.)
    f = plt.figure()

    DPI = f.get_dpi()
    H = img_arr.shape[0]
    W = img_arr.shape[1]
    f.set_size_inches(w=float(W)/float(DPI),
                        h=float(H)/float(DPI))

    # Get Plot
    plot_img_to_axis(ax=plt.gca(),
                     img_arr=img_arr,
                     middle_percentile=middle_percentile,
                     highlight_inf_pixels=highlight_inf_pixels)
 
    f.subplots_adjust(bottom=0.,left=0.,right=1.,top=1.)

    # Save plot to png (Browse Image Product)
    plt.axis('off')
    plt.savefig(filepath,
                bbox_inches='tight', pad_inches=0,
                dpi=DPI
                )

    plt.close()


def get_browse_product_filename(
        product_name,
        band,
        freq,
        pol,
        quantity,
        browse_image_dir,
        browse_image_prefix=None):
    '''
    Returns the full filename (with path) for Browse Image Product.

    The browse image products should follow this naming convention,
    (Convention modified from Phil Callahan's slides on 11 Aug 2022.)
        <prefix>_<product name>_BAND_F_PP[PP]_qqq
            <prefix>        : browse image prefix, supplied by SDS
            <product name>  : RSLC, GLSC, etc.
            BAND            : LSAR or SSAR
            F               : frequency A or B 
            PP              : polarization, e.g. 'HH' or 'HV'.
                              [PP] additional polarization for GCOV 
            qqq             : quantity: mag, phs, coh, cov, rof, aof, cnc, iph 
                                        (see product list)

    TODO - double check the file naming convention

    '''
    filename = f'{product_name.upper()}_{band}_{freq}_{pol}_{quantity}.png'
    if browse_image_prefix is not None:
        filename = f'{browse_image_prefix}_{filename}'
    filename = os.path.join(browse_image_dir, filename)

    return filename


def plot2pdf(img_arr,
             plots_pdf,
             title=None,
             xlim=None,
             ylim=None,
             xlabel=None,
             ylabel=None,
             middle_percentile=100.0,
             highlight_inf_pixels=False
             ):
    '''
    Plot the clipped image array and append it to the pdf.

    Parameters
    ----------
    img_arr : array_like
        Image to plot
    plots_pdf : PdfPages
        The output pdf file to append the power image plot to
    title : str, optional
        The full title for the plot
    xlim, ylim : sequence of numeric, optional
        Lower and upper limits for the axes ticks for the plot.
        Format: xlim=[<x-axis lower limit>, <x-axis upper limit>], 
                ylim=[<y-axis lower limit>, <y-axis upper limit>]
    xlabel, ylabel : str
        Axes labels for the x-axis and y-axis (respectively)
    middle_percentile : numeric, optional
        Defines the middle percentile range of the `img_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 100.0.
    highlight_inf_pixels : bool, optional
        True to color pixels with an infinite value green in saved images.
        False to color infinite pixels with matplotlib's default
        for infinite values. Defaults to False.
    '''

    # Instantiate the figure object
    # (Need to instantiate it outside of the plotting function
    # in order to later modify the plot for saving purposes.)
    f = plt.figure()
    ax = plt.gca()

    # Get Plot
    ax_img = plot_img_to_axis(
                     ax=ax,
                     img_arr=img_arr,
                     xlim=xlim, ylim=ylim,
                     middle_percentile=middle_percentile,
                     highlight_inf_pixels=highlight_inf_pixels)

    # Add Colorbar
    plt.colorbar(ax_img, ax=ax)

    ## Label the plot
    
    # If xlim or ylim are not provided, let matplotlib auto-assign the ticks.
    # Otherwise, dynamically calculate and set the ticks w/ labels for 
    # the x-axis and/or y-axis.
    # (Attempts to set the limits by using the `extent` argument for 
    # matplotlib.imshow() caused significantly distorted images.
    # So, compute and set the ticks w/ labels manually.)
    if xlim is not None or ylim is not None:

        img_arr_shape = np.shape(img_arr)

        # Set the density of the ticks on the figure
        ticks_per_inch = 2.5

        # Get full figure size in inches
        fig_w, fig_h = f.get_size_inches()
        W = img_arr_shape[1]
        H = img_arr_shape[0]

        # Update variables to the actual, displayed image size
        # (The actual image will have a different aspect ratio
        # than the matplotlib figure window's aspect ratio.
        # But, altering the matplotlib figure window's aspect ratio
        # will lead to inconsistently-sized pages in the output .pdf)
        if H/W >= fig_h/fig_w:
            # image will be limited by its height, so
            # it will not use the full width of the figure
            fig_w = W * (fig_h/H)
        else:
            # image will be limited by its width, so
            # it will not use the full height of the figure
            fig_h = H * (fig_w/W)

    if xlim is not None:

        # Compute num of xticks to use
        num_xticks = int(ticks_per_inch * fig_w)

        # Always have a minimum of 2 labeled ticks
        num_xticks = num_xticks if num_xticks >=2 else 2

        # Specify where we want the ticks, in pixel locations.
        xticks = np.linspace(0,img_arr_shape[1], num_xticks)
        ax.set_xticks(xticks)

        # Specify what those pixel locations correspond to in data coordinates.
        # By default, np.linspace is inclusive of the endpoint
        xticklabels = ['{:.1f}'.format(i) for i in np.linspace(start=xlim[0],
                                              stop=xlim[1],
                                              num=num_xticks)]
        ax.set_xticklabels(xticklabels)

        plt.xticks(rotation=45)

    if ylim is not None:
        
        # Compute num of yticks to use
        num_yticks = int(ticks_per_inch * fig_h)

        # Always have a minimum of 2 labeled ticks
        if num_yticks < 2:
            num_yticks = 2

        # Specify where we want the ticks, in pixel locations.
        yticks = np.linspace(0,img_arr_shape[0], num_yticks)
        ax.set_yticks(yticks)

        # Specify what those pixel locations correspond to in data coordinates.
        # By default, np.linspace is inclusive of the endpoint
        yticklabels = ['{:.1f}'.format(i) for i in np.linspace(start=ylim[0],
                                                        stop=ylim[1],
                                                        num=num_yticks)]
        ax.set_yticklabels(yticklabels)
    
    # Label the Axes
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Add title
    if title is not None:
        plt.title(title)

    # Make sure axes labels do not get cut off
    f.tight_layout()

    # Append figure to the output .pdf
    plots_pdf.savefig(plt.gcf())

    # Close the plot
    plt.close()


def plot_img_to_axis(ax,
                     img_arr,
                     highlight_inf_pixels,
                     xlim=None,
                     ylim=None,
                     middle_percentile=100.0):
    '''
    Clip and plot `img_arr` onto `ax`.

    For example, this function can be used to plot the power image
    for an RSLC product.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis to plot the image on.
    img_arr : array_like
        The image data, such as matches matplotlib.plt.imshow's
        specifications for `X`
    highlight_inf_pixels : bool
        True to color pixels with an infinite value green in saved images.
        Defaults to matplotlib's default.
    middle_percentile : numeric
        Defines the middle percentile range of the `img_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 100.0.

    Returns
    -------
    ax_img : matplotlib.image.AxesImage
        `img_arr` clipped to the `middle_percentile` and plotted on `ax`

    Notes
    -----
    1) In this function, the `img_arr` will be manually clipped
    before being passed to ax.imshow().
    While imshow() can do the clipping automatically if the
    vmin and vmax values are passed in, in practise, doing so 
    causes the resultant size of the output .pdf files that contain 
    these figures to grow from e.g. 537KB to 877MB.
    A workaround is to clip the image data in advance.
    2) The interpolation method is imshow()'s default of antialiasing.
    Setting interpolation='none' causes the size of the output
    .pdf files that contain these figures to grow from e.g. 537KB to 877MB.
    '''

    # Get vmin and vmax to set the desired range of the colorbar
    vmin, vmax = calc_vmin_vmax(img_arr, middle_percentile=middle_percentile)

    # Manually clip the image data (See `Notes` in function description)
    clipped_array = np.clip(img_arr, a_min=vmin, a_max=vmax)

    # TODO Storing the clipped image data to an array will (temporarily)
    # use another big chunk of memory. Revisit this code later if/when this
    # becomes an issue.

    # Highlight infinite pixels in green, if requested.
    cmap=plt.cm.gray
    if highlight_inf_pixels:
        cmap.set_bad('g')

    # Plot the img_arr image.
    return ax.imshow(X=clipped_array, cmap=cmap)


def calc_vmin_vmax(data_in, middle_percentile=100.0):
    '''
    Calculate the values of vmin and vmax for the 
    input array using the given middle percentile.

    For example, if `middle_percentile` is 95.0, then this will
    return the value of the 2.5th quantile and the 97.5th quantile.

    Parameters
    ----------
    data_in : array_like
        Input array
    middle_percentile : numeric
        Defines the middle percentile range of the `data_in`. 
        Must be in the range [0, 100].
        Defaults to 100.0.

    Returns
    -------
    vmin, vmax : numeric
        The lower and upper values (respectively) of the middle 
        percentile.

    '''
    nisarqa.verify_valid_percentile(middle_percentile)

    fraction = 0.5*(1.0 - middle_percentile/100.0)

    # Get the value of the e.g. 2.5th quantile and the 97.5th quantile
    vmin, vmax = np.quantile(data_in, [fraction, 1-fraction])

    return vmin, vmax
