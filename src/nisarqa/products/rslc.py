from dataclasses import dataclass, field, fields
import h5py
import os
import time

from typing import Any, Union, Iterable, Tuple
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler
from PIL import Image

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


@dataclass
class CoreQAParams:
    '''
    Data structure to hold the core parameters for the
    QA code's output plots and statistics files that are
    common to all NISAR products.

    Parameters
    ----------
    plots_pdf : PdfPages
        The output file to append the power image plot to
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    browse_image_dir : str, optional
        Path to directory to save the browse image product.
        Defaults to the current directory.
    browse_image_prefix : str, optional
        String to pre-pend to the name of the generated browse image product.
        Defaults to "".
    tile_shape : tuple of ints, optional
        Shape of each tile to be processed. If `tile_shape` is
        larger than the shape of `arr`, or if the dimensions of `arr`
        are not integer multiples of the dimensions of `tile_shape`,
        then smaller tiles may be used.
        -1 to use all rows / all columns (respectively).
        Format: (num_rows, num_cols) 
        Defaults to (1024, -1) to use all columns (i.e. full rows of data)
        and leverage Python's row-major ordering.
    '''

    # Attributes that are common to all NISAR Products
    stats_h5: h5py.File
    plots_pdf: PdfPages
    browse_image_dir: str = '.'
    browse_image_prefix: str = ""
    tile_shape: tuple = (1024,-1)

    @classmethod
    def from_parent(cls, core, **kwargs):
        '''
        Construct a child of CoreQAParams
        from an existing instance of CoreQAParams.

        Parameters
        ----------
        core : CoreQAParams
            Instance of CoreQAParams whose attributes will
            populate the new child class instance.
            Note that a only shallow copy is performed when populating
            the new instance; for parent attributes that contain
            references, the child object will reference the same
            same 
        **kwargs : optional
            Attributes specific to the child class of CoreQAParams.

        Example
        -------
        >>> parent = CoreQAParams()
        >>> @dataclass
        ... class ChildParams(CoreQAParams):
        ...     x: int
        ... 
        >>> y = ChildParams.from_parent(core=parent, x=2)
        >>> print(y)
        ChildParams(plots_pdf=<matplotlib.backends.backend_pdf.PdfPages object at 0x7fd04cab6050>,
        stats_h5=<contextlib._GeneratorContextManager object at 0x7fd04cab6690>,
        browse_image_dir='.', browse_image_prefix='', tile_shape=(1024, -1), x=2)        
        '''
        if not isinstance(core, CoreQAParams):
            raise ValueError("`core` input must be of type CoreQAParams.")

        # Create shallow copy of the dataclass into a dict.
        # (Using the asdict() method to create a deep copy throws a
        # "TypeError: cannot serialize '_io.BufferedWriter' object" 
        # exception when copying the field with the PdfPages object.)
        core_dict = dict((field.name, getattr(core, field.name)) 
                                            for field in fields(core))

        return cls(**core_dict, **kwargs)


@dataclass
class RSLCPowerImageParams(CoreQAParams):
    '''
    Data structure to hold the parameters to generate the
    RSLC Power Images.

    Use the class method .from_parent() to construct
    an instance from an existing CoreQAParams object.

    Parameters
    ----------
    **core : CoreQAParams
        All fields from the parent class CoreQAParams.
    nlooks_freqa, nlooks_freqb : int or iterable of int
        Number of looks along each axis of the input array 
        for the specified frequency.
    linear_units : bool
        True to compute power in linear units, False for decibel units.
        Defaults to True.
    num_mpix : numeric
        The approx. size (in megapixels) for the final multilooked image.
        Defaults to 4.0 MPix.
    middle_percentile : numeric
        Defines the middle percentile range of the `img_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 100.0.

    Attributes
    ----------
    pow_units : str
        Units of the power image.
        If `linear_units` is True, this will be set to 'linear'.
        If `linear_units` is False, this will be set to 'dB'.
    '''

    # Attributes for generating the Power Image(s)
    nlooks_freqa: Union[int, Iterable[int]] = 1  # No apparent multilooking
    nlooks_freqb: Union[int, Iterable[int]] = 1  # No apparent multilooking
    linear_units: bool = True
    num_mpix: float = 4.0
    middle_percentile: float = 100.0

    # Auto-generated attributes
    pow_units: str = field(init=False)

    def __post_init__(self):
        # Phase bin edges - allow for either radians or degrees
        if self.linear_units:
            self.pow_units = 'linear'
        else:
            self.pow_units = 'dB'


@dataclass
class RSLCHistogramParams(CoreQAParams):
    '''
    Data structure to hold the parameters to generate the
    RSLC Power and Phase Histograms.

    Use the class method .from_parent() to construct
    an instance from an existing CoreQAParams object.

    Parameters
    ----------
    **core : CoreQAParams
        All fields from the parent class CoreQAParams.
    decimation_ratio : pair of int
        The step size to decimate the input array for computing
        the power and phase histograms.
        For example, (2,3) means every 2nd azimuth line and
        every 3rd range line will be used to compute the histograms.
        Defaults to (1,1), i.e. no decimation will occur.
        Format: (<azimuth>, <range>)
    phs_in_radians : bool
        True to compute phase in radians units, False for degrees units.
        Defaults to True.
    pow_histogram_start : numeric, optional
        The starting value for the range of the power histogram edges.
        Defaults to -80. (units are dB)
    pow_histogram_endpoint : numeric, optional
        The endpoint value for the range of the power histogram edges.
        Defaults to 20. (units are dB)
    

    Attributes
    ----------
    pow_bin_edges : numpy.ndarray
        The bin edges (including endpoint) to use when computing
        the power histograms. Will be set to 100 uniformly-spaced bins
        in range [`pow_histogram_start`, `pow_histogram_endpoint`],
        including endpoint.
    pow_units : str
        Units of the power histograms; this will be set to 'dB'.
    phs_bin_edges : numpy.ndarray
        The bin edges (including endpoint) to use when computing
        the phase histograms.
        If `phs_in_radians` is True, this will be set to 100 
        uniformly-spaced bins in range [-pi,pi], including endpoint.
        If `phs_in_radians` is False, this will be set to 100
        uniformly-spaced bins in range [-180,180], including endpoint.
    pow_units : str
        Units of the phase histograms.
        If `phs_in_radians` is True, this will be set to 'radians'.
        If `phs_in_radians` is True, this will be set to 'degrees'.
    '''

    # Attributes for generating Power and Phase Histograms
    # User-Provided attributes:
    decimation_ratio: Tuple[int, int] = (1,1)  # no decimation
    phs_in_radians: bool = True

    pow_histogram_start: float = -80.0
    pow_histogram_endpoint: float = 20.0

    # Auto-generated attributes
    # Power Bin Edges (generated based upon
    # `pow_histogram_start` and `pow_histogram_endpoint`)
    pow_bin_edges: np.ndarray = field(init=False)
    pow_units: str = field(init=False)

    # Phase bin edges (generated based upon `phs_in_radians`)
    phs_bin_edges: np.ndarray = field(init=False)
    phs_units: str = field(init=False)

    def __post_init__(self):
        # Power Bin Edges - hardcode to be in decibels
        # 101 bin edges => 100 bins
        self.pow_bin_edges = np.linspace(self.pow_histogram_start,
                                         self.pow_histogram_endpoint,
                                         num=101,
                                         endpoint=True)
        self.pow_units = 'dB'

        # Phase bin edges - allow for either radians or degrees
        if self.phs_in_radians:
            self.phs_units = 'radians'
            start = -np.pi
            stop = np.pi
        else:
            self.phs_units = 'degrees'
            start = -180
            stop = 180

        # 101 bin edges => 100 bins
        self.phs_bin_edges = np.linspace(start, stop, num=101, endpoint=True)


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
        '''
        Read a complex float16 HDF5 dataset as a numpy.complex64 array.
        Avoids h5py/numpy dtype bugs and uses numpy float16 -> float32 conversions
        which are about 10x faster than HDF5 ones.
        '''
        # This avoids h5py exception:
        # TypeError: data type '<c4' not understood
        # Also note this syntax changed in h5py 3.0 and was deprecated in 3.6, see
        # https://docs.h5py.org/en/stable/whatsnew/3.6.html

        complex32 = np.dtype([('r', np.float16), ('i', np.float16)])
        z = ds.astype(complex32)[key]

        # Define a similar datatype for complex64 to be sure we cast safely.
        complex64 = np.dtype([('r', np.float32), ('i', np.float32)])

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
    def from_h5dataset(cls, h5dataset, name):
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
    epoch : str
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

    epoch: str


    @classmethod
    def from_h5dataset(cls,
                       h5dataset, band, freq, pol,
                       az_spacing, az_start, az_stop,
                       range_spacing, rng_start, rng_stop,
                       epoch):
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
        epoch : str
            The reference epoch for this observation,
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
        name = f'{band}_{freq}_{pol}'

        # Note the order of the positional arguments being passed;
        # the attributes of the parent class must be passed first.
        return cls(data, name, band, freq, pol, 
                   az_spacing, az_start, az_stop,
                   range_spacing, rng_start, rng_stop,
                   epoch)

    @classmethod
    def from_nisar_rslc_h5_dataset(cls,
                       h5_file, band, freq, pol):
        '''
        Initialize an RSLCRasterQA object for the given 
        band-freq-pol image in the input NISAR RSLC HDF5 file.
        
        This will store the dataset as a ComplexFloat16Decoder
        object instead of a standard Arraylike object. If the file 
        does not contain an image dataset for the given 
        band-freq-pol combination, a DatasetNotFoundError
        exception will be thrown.

        Parameters
        ----------
        h5_file : h5py.File
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

        Notes
        -----
        The `name` attribute will be populated with a string
        of the format: <band>_<freq>_<pol>
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
            raise DatasetNotFoundError

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
        epoch = sec_since_epoch.replace('seconds since ', '').strip()

        return RSLCRasterQA.from_h5dataset(h5_file[pol_path],
                                   band=band,
                                   freq=freq,
                                   pol=pol,
                                   az_spacing=az_spacing,
                                   az_start=az_start,
                                   az_stop=az_stop,
                                   range_spacing=range_spacing,
                                   rng_start=rng_start,
                                   rng_stop=rng_stop,
                                   epoch=epoch)


def get_bands_freqs_pols(h5_file):
    '''
    Locate the available bands, frequencies, and polarizations
    in the input HDF5 file.

    Parameters
    ----------
    h5_file : h5py.File
        Handle to the input product h5 file

    Returns
    -------
    bands : dict of h5py Groups
        Dict of the h5py Groups for each band in `h5_file`,
        where the keys are the available bands (i.e. 'SSAR' or 'LSAR').
        Format: bands[<band>] -> a h5py Group
        Ex: bands['LSAR'] -> the h5py Group for LSAR
    freqs : nested dict of h5py Groups
        Nested dict of the h5py Groups for each freq in `h5_file`,
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
    h5_file : h5py.File
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
    h5_file : h5py.File
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
            raise ValueError(f'Provided input file\'s band {band} does not '
                              'contain any frequency groups.')

    return freqs


def _get_pols(h5_file, freqs):
    '''
    Finds the available polarization rasters in the input file
    and stores their paths in a nested dictionary.

    Parameters
    ----------
    h5_file : h5py.File
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

                try:
                    tmp_RSLCRasterQA = \
                        RSLCRasterQA.from_nisar_rslc_h5_dataset(h5_file, band, freq, pol)
                except nisarqa.DatasetNotFoundError:
                    # RSLC Raster QA could not be created, which means that the
                    # input file did not contain am image with the current
                    # `band`, `freq`, and `pol` combination.
                    continue

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
    params : RSLCPowerImageParams
        A dataclass containing the parameters for processing
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
    params : RSLCPowerImageParams
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
    output_power_img = nisarqa.compute_multilooked_power_by_tiling(
                                            arr=img.data,
                                            nlooks=nlooks,
                                            linear_units=params.linear_units,
                                            tile_shape=params.tile_shape)
    end_time = time.time()-start_time
    print('time to multilook image (sec): ', end_time)
    print('time to multilook image (min): ', end_time/60.)

    print(f'Multilooking Complete. Multilooked shape: {output_power_img.shape}')
    print(f'Multilooked size: {output_power_img.size} Mpix.')

    # Apply image correction to the multilooked array
    output_power_img = apply_img_correction(img_arr=output_power_img,
                                       middle_percentile=params.middle_percentile)

    # Plot and Save Power Image as Browse Image Product
    browse_img_file = get_browse_product_filename(
                                product_name='RSLC',
                                band=img.band,
                                freq=img.freq,
                                pol=img.pol,
                                quantity='pow',
                                browse_image_dir=params.browse_image_dir,
                                browse_image_prefix=params.browse_image_prefix)

    plot_to_grayscale_png(img_arr=output_power_img,
                          filepath=browse_img_file)

    # Plot and Save Power Image to graphical summary pdf
    title = f'RSLC Multilooked Power ({params.pow_units})\n{img.name}'

    # Get Azimuth (y-axis) label
    az_title = f"Zero Doppler Time\n(seconds since {img.epoch})"

    # Get Range (x-axis) labels and scale
    # Convert range from meters to km
    rng_start_km = img.rng_start/1000.
    rng_stop_km = img.rng_stop/1000.
    rng_title = 'Slant Range (km)'

    plot2pdf(img_arr=output_power_img,
             title=title,
             ylim=[img.az_start, img.az_stop],
             xlim=[rng_start_km, rng_stop_km],
             ylabel=az_title,
             xlabel=rng_title,
             plots_pdf=params.plots_pdf
             )


def apply_img_correction(img_arr, middle_percentile=100.0):
    '''
    Apply image correction to the input array.

    Returns a copy of the input image with the following modifications:
        * Values outside of the range defined by `middle_percentile` clipped

    Parameters
    ----------
    img_arr : array_like
        Input array
    middle_percentile : numeric, optional
        Defines the middle percentile range of the `img_arr` 
        that the colormap covers. Must be in the range [0, 100].
        Defaults to 100.0.

    Returns
    -------
    output_img : numpy.ndarray
        The input array with the specified image correction applied.
    '''
    # Clip the image data
    vmin, vmax = calc_vmin_vmax(img_arr, middle_percentile=middle_percentile)
    output_img = np.clip(img_arr, a_min=vmin, a_max=vmax)

    return output_img


def plot_to_grayscale_png(img_arr, filepath):
    '''
    Clip and save the image array to a grayscale (1 channel)
    browse image png.

    Parameters
    ----------
    img_arr : array_like
        2D Image to plot
    filepath : str
        Full filepath the browse image product.
    '''

    # Only use 2D arrays
    if len(np.shape(img_arr)) != 2:
        raise ValueError('Input array must be 2D.')

    # Normalize array, and scale to 0-255 for unsigned int8
    arr_min = np.min(img_arr)
    arr_max = np.max(img_arr)
    img_arr = (img_arr - arr_min) / (arr_max - arr_min)
    img_arr *= 255
    img_arr = np.uint8(img_arr)

    # Save as grayscale (1-channel) image using PIL.Image
    # (Pyplot only saves png's as RGB, even if cmap=plt.cm.gray)
    im = Image.fromarray(img_arr)
    im.save(filepath)  # default = 72 dpi


def get_browse_product_filename(
        product_name,
        band,
        freq,
        pol,
        quantity,
        browse_image_dir,
        browse_image_prefix=''):
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
    if browse_image_prefix is not '':
        filename = f'{browse_image_prefix}_{filename}'
    filename = os.path.join(browse_image_dir, filename)

    return filename


def plot2pdf(img_arr,
             plots_pdf,
             title=None,
             xlim=None,
             ylim=None,
             xlabel=None,
             ylabel=None
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
    '''

    # Instantiate the figure object
    # (Need to instantiate it outside of the plotting function
    # in order to later modify the plot for saving purposes.)
    f = plt.figure()
    ax = plt.gca()

    # Plot the img_arr image.
    ax_img = ax.imshow(X=img_arr, cmap=plt.cm.gray)

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
    plt.close(f)


def calc_vmin_vmax(data_in, middle_percentile=100.0):
    '''
    Calculate the values of vmin and vmax for the 
    input array using the given middle percentile.

    For example, if `middle_percentile` is 95.0, then this will
    return the value of the 2.5th percentile and the 97.5th 
    percentile.

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

    # Get the value of the e.g. 0.025 quantile and the 0.975 quantile
    vmin, vmax = np.quantile(data_in, [fraction, 1-fraction])

    return vmin, vmax


def process_power_and_phase_histograms(pols, params):
    '''
    Generate the RSLC Power Histograms; save their plots
    to the graphical summary .pdf file and their data to the
    statistics .h5 file.

    Power histogram will be computed in decibel units.
    Phase histogram defaults to being computed in radians, 
    configurable to be computed in degrees.

    Parameters
    ----------
    pols : nested dict of RSLCRaster
        Nested dict of RSLCRaster objects, where each object represents
        a polarization dataset in `h5_file`.
        Format: pols[<band>][<freq>][<pol>] -> a RSLCRaster
        Ex: pols['LSAR']['A']['HH'] -> the HH dataset, stored 
                                       in a RSLCRaster object
    params : RSLCHistogramParams
        A structure containing the parameters for processing
        and outputting the power and phase histograms.
    '''

    # Store processing settings in stats HDF5 file
    stats_h5 = params.stats_h5
    proc_path = f'/processing/'
    stats_h5[proc_path + 'histogramEdgesPower'] = \
                params.pow_bin_edges
    stats_h5[proc_path + 'histogramEdgesPhase'] = \
                params.phs_bin_edges
    stats_h5[proc_path + 'histogramDecimationAz'] = \
                params.decimation_ratio[0]
    stats_h5[proc_path + 'histogramDecimationRange'] = \
                params.decimation_ratio[1]
    stats_h5[proc_path + 'histogramUnitsPower'] = \
                params.pow_units
    stats_h5[proc_path + 'histogramUnitsPhase'] = \
                params.phs_units

    # Generate and store the histograms
    for band in pols:
        for freq in pols[band]:
            generate_histogram_single_freq(pols[band][freq],
                                            band, freq, params)


def generate_histogram_single_freq(pol, band, freq, params):
    '''
    Generate the RSLC Power Histograms for a single frequency.
    
    The histograms' plots will be appended to the graphical
    summary file `params.plots_pdf`, and their data will be
    stored in the statistics .h5 file `params.stats_h5`.
    Power histogram will be computed in decibel units.
    Phase histogram defaults to being computed in radians, 
    configurable to be computed in degrees.

    Parameters
    ----------
    pol : dict of RSLCRaster
        dict of RSLCRaster objects for the given `band`
        and `freq`. Each key is a polarization (e.g. 'HH'
        or 'HV'), and each key's item is the corresponding
        RSLCRaster instance.
        Ex: pol['HH'] -> the HH dataset, stored 
                         in a RSLCRaster object
    band : str
        Band name for the histograms to be processed,
        e.g. 'LSAR'
    freq : str
        Frequency name for the histograms to be processed,
        e.g. 'A' or 'B'
    params : RSLCHistogramParams
        A structure containing the parameters for processing
        and outputting the power and phase histograms.
    '''

    print(f'Generating Histograms for Frequency {freq}...')

    # Open separate figures for each of the power and phase histograms.
    # Each band+frequency will have a distinct plot, with all of the
    # polarization images for that setup plotted together on the same plot.
    pow_fig, pow_ax = plt.subplots(nrows=1, ncols=1)
    phs_fig, phs_ax = plt.subplots(nrows=1, ncols=1)

    # Use custom cycler for accessibility
    pow_ax.set_prop_cycle(nisarqa.CUSTOM_CYCLER)
    phs_ax.set_prop_cycle(nisarqa.CUSTOM_CYCLER)

    for pol_name, pol_data in pol.items():
        # Get histogram probability densities
        pow_hist_density, phs_hist_density = \
                nisarqa.compute_power_and_phase_histograms_by_tiling(
                            arr=pol_data.data,
                            pow_bin_edges=params.pow_bin_edges,
                            phs_bin_edges=params.phs_bin_edges,
                            decimation_ratio=params.decimation_ratio,
                            tile_shape=params.tile_shape,
                            density=True)

        # Save to stats.h5 file
        h5_pol_grp_path = f'/data/frequency{freq}/{pol_name}/'
        params.stats_h5[h5_pol_grp_path + 'powerHistogramDensity'] = pow_hist_density
        params.stats_h5[h5_pol_grp_path + 'phaseHistogramDensity'] = phs_hist_density

        # Add these densities to the figures
        add_hist_to_axis(pow_ax,
                         counts=pow_hist_density, 
                         edges=params.pow_bin_edges,
                         label=pol_name)

        add_hist_to_axis(phs_ax,
                         counts=phs_hist_density,
                         edges=params.phs_bin_edges,
                         label=pol_name)

    # Label the Power Figure
    title = f'{band} Frequency {freq} Power Histograms'
    pow_ax.set_title(title)

    pow_ax.legend(loc='upper right')
    pow_ax.set_xlabel(f'RSLC Power ({params.pow_units})')
    pow_ax.set_ylabel(f'Density (1/{params.pow_units})')

    # Per ADT, let the top limit float for Power Spectra
    pow_ax.set_ylim(bottom=0.0)
    pow_ax.grid()

    # Label the Phase Figure
    phs_ax.set_title(f'{band} Frequency {freq} Phase Histograms')
    phs_ax.legend(loc='upper right')
    phs_ax.set_xlabel(f'RSLC Phase ({params.phs_units})')
    phs_ax.set_ylabel(f'Density (1/{params.phs_units})')
    phs_ax.set_ylim(bottom=0.0, top=0.5)
    phs_ax.grid()

    # Save complete plots to graphical summary pdf file
    params.plots_pdf.savefig(pow_fig)
    params.plots_pdf.savefig(phs_fig)

    # Close all figures
    plt.close(pow_fig)
    plt.close(phs_fig)

    print(f'Histograms for Frequency {freq} complete.')


def add_hist_to_axis(axis, counts, edges, label):
    '''Add the plot of the given counts and edges to the
    axis object. Points will be centered in each bin,
    and the plot will be denoted `label` in the legend.
    '''
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    axis.plot(bin_centers, counts, label=label)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
