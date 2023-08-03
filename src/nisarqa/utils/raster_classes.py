from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import warnings

import h5py
import numpy as np
import numpy.typing as npt

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def is_complex32(dataset: h5py.Dataset) -> bool:
    """
    Check if the input dataset is half-precision complex ("complex32").

    Parameters
    --------
    dataset : h5py.Dataset
        The input dataset.

    Returns
    -------
    bool
        True if the input dataset is complex32.

    Notes
    -----
    If a Dataset has dtype complex32, h5py < v3.8 will throw the error
    "data type '<c4' not understood" if the Dataset is accessed.
    h5py 3.8.0 adopted @bhawkins's patch which allows h5py to
    recognize the new complex32 datatype that is used for RSLC
    HDF5 Datasets in R3.2, R3.3, etc. That patch fixes the dtype attribute
    such that it returns a structured datatype if the data is complex32.
    """
    try:
        dataset.dtype
    except TypeError as e:
        # h5py < v3.8 will throw the error "data type '<c4' not understood"
        if str(e) == "data type '<c4' not understood":
            return True
        else:
            raise
    else:
        # h5py >= v3.8 recognizes the new complex32 datatype
        complex32 = np.dtype([("r", np.float16), ("i", np.float16)])
        return dataset.dtype == complex32


class ComplexFloat16Decoder(object):
    """Wrapper to read in NISAR product datasets that are '<c4' type,
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
    """

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

        complex32 = np.dtype([("r", np.float16), ("i", np.float16)])
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
class Raster:
    """
    Raster image dataset base class.

    Parameters
    ----------
    data : array_like
        Raster data to be stored. Can be a numpy.ndarray, h5py.Dataset, etc.
    name : str
        Name for the dataset
    band : str
        Name of the band for `img`, e.g. 'LSAR'
    freq : str
        Name of the frequency for `img`, e.g. 'A' or 'B'
    pol : str
        Name of the polarization for `img`, e.g. 'HH' or 'HV'
    """

    # Raster data. Could be a numpy.ndarray, h5py.Dataset, etc.
    data: npt.ArrayLike

    # identifying name of this Raster; can be used for logging
    # e.g. 'LSAR_A_HH'
    name: str

    band: str
    freq: str
    pol: str


# TODO - move to raster.py
@dataclass
class SARRaster(ABC, Raster):
    """Abstract Base Class for SAR Raster dataclasses."""

    @property
    @abstractmethod
    def y_axis_spacing(self):
        """Pixel Spacing in Y direction (azimuth for radar domain rasters)"""
        pass

    @property
    @abstractmethod
    def x_axis_spacing(self):
        """Pixel Spacing in X direction (range for radar domain rasters)"""
        pass


@dataclass
class RadarRaster(SARRaster):
    """
    A Raster with attributes specific to Radar products.

    The attributes specified here are based on the needs of the QA code
    for generating and labeling plots, etc.

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
        The start time of the observation for this RSLC Raster.
        This corresponds to the upper edge of the top pixels.
    az_stop : float
        The stopping time of the observation for this RSLC Raster.
        This corresponds to the lower side of the bottom pixels.
    range_spacing : float
        Range spacing of pixels of input array
    rng_start : float
        Start (near) distance of the range of input array
        This corresponds to the left side of the left-most pixels.
    rng_stop : float
        End (far) distance of the range of input array
        This corresponds to the right side of the right-most pixels.
    epoch : str
        The start of the epoch for this observation,
        in the format 'YYYY-MM-DD HH:MM:SS'

    Notes
    -----
    If data is NISAR HDF5 dataset, suggest initializing using
    the class method init_from_nisar_h5_product(..).
    """

    # Attributes of the input array
    az_spacing: float
    az_start: float
    az_stop: float

    range_spacing: float
    rng_start: float
    rng_stop: float

    epoch: str

    @property
    def y_axis_spacing(self):
        return self.az_spacing

    @property
    def x_axis_spacing(self):
        return self.range_spacing

    @classmethod
    def init_from_nisar_h5_product(cls, h5_file, band, freq, pol):
        """
        Initialize an RadarRaster object for the given
        band-freq-pol image in the input NISAR Radar domain HDF5 file.

        NISAR product type must be one of: 'RSLC', 'SLC', 'RIFG', 'RUNW', 'ROFF'
        If the product type is 'RSLC' or 'SLC', then the image dataset
        will be stored as a ComplexFloat16Decoder instance; this will allow
        significantly faster access to the data.

        Parameters
        ----------
        h5_file : h5py.File
            File handle to a valid NISAR product hdf5 file.
            Polarization images must be located in the h5 file in the path:
            /science/<band>/<product name>/swaths/frequency<freq>/<pol>
            or they will not be found. This is the file structure
            as determined from the NISAR Product Spec.
        band : str
            name of the band for `img`, e.g. 'LSAR'
        freq : str
            name of the frequency for `img`, e.g. 'A' or 'B'
        pol : str
            name of the polarization for `img`, e.g. 'HH' or 'HV'

        Raises
        ------
        DatasetNotFoundError
            If the file does not contain an image dataset for the given
            band-freq-pol combination, a DatasetNotFoundError
            exception will be thrown.

        Notes
        -----
        The `name` attribute will be populated with a string
        of the format: <product type>_<band>_<freq>_<pol>
        """

        product = nisarqa.get_NISAR_product_type(h5_file)

        if product not in ("RSLC", "SLC", "RIFG", "RUNW", "ROFF"):
            # self.logger.log_message(logging_base.LogFilterError, 'Invalid file structure.')
            raise nisarqa.InvalidNISARProductError

        # Hardcoded paths to various groups in the NISAR RSLC h5 file.
        # These paths are determined by the RSLC .xml product spec
        swaths_path = f"/science/{band}/{product}/swaths"
        freq_path = f"{swaths_path}/frequency{freq}"
        pol_path = f"{freq_path}/{pol}"

        swaths_group = h5_file[swaths_path]
        freq_group = h5_file[freq_path]

        if pol_path in h5_file:
            pol_group = h5_file[pol_path]
            # self.logger.log_message(logging_base.LogFilterInfo,
            #                         'Found image %s' % band_freq_pol_str)
            pass
        else:
            # self.logger.log_message(logging_base.LogFilterInfo,
            #                         'Image %s not present' % band_freq_pol_str)
            raise nisarqa.DatasetNotFoundError

        # Get dataset object
        # Most Radar Doppler NISAR products should be directly readible
        # by h5py, numpy, etc. as complex64, float, etc. The exception is RSLC.
        # RSLC Product Spec says that NISAR RSLC files should be complex32,
        # which requires special handling to read and access.
        # As of h5py 3.8.0, h5py gained the ability to read complex32
        # datasets, however numpy and other downstream packages do not
        # necessarily have that flexibility.
        if nisarqa.is_complex32(pol_group):
            # If the input RSLC product has dtype complex32, then we'll need
            # to use ComplexFloat16Decoder.
            if product == "RSLC":
                # The RSLC dataset is complex32. h5py >= 3.8 can read these
                # but numpy cannot yet. So, use the ComplexFloat16Decoder.
                dataset = ComplexFloat16Decoder(pol_group)
                print(
                    "(PASS) PASS/FAIL Check: Product raster dtype conforms"
                    " to RSLC Product Spec dtype of complex32."
                )
            else:
                raise TypeError(
                    f"Input dataset is for a {product} product and "
                    "has dtype complex32. As of R3.3, of the "
                    "radar-doppler NISAR products, only RSLC "
                    "products can have dtype complex32."
                )
        else:
            # Use h5py's standard reader
            dataset = pol_group

            if product == "SLC":
                print(
                    "(FAIL) PASS/FAIL Check: Product raster dtype conforms "
                    "to RSLC Product Spec dtype of complex32."
                )
            else:
                # TODO - for RIFG, RUNW, and ROFF, confirm that this
                # next print statement is, in fact, true.
                print(
                    "(PASS) PASS/FAIL Check: Product raster dtype conforms "
                    f"to {product} Product Spec dtype."
                )

        # From the xml Product Spec, sceneCenterAlongTrackSpacing is the
        # 'Nominal along track spacing in meters between consecutive lines
        # near mid swath of the RSLC image.'
        az_spacing = freq_group["sceneCenterAlongTrackSpacing"][...]

        # Get Azimuth (y-axis) tick range + label
        # path in h5 file: /science/LSAR/RSLC/swaths/zeroDopplerTime
        # For NISAR, radar-domain grids are referenced by the center of the
        # pixel, so +/- half the distance of the pixel's side to capture
        # the entire range.
        az_start = float(swaths_group["zeroDopplerTime"][0]) - 0.5 * az_spacing
        az_stop = float(swaths_group["zeroDopplerTime"][-1]) + 0.5 * az_spacing

        # From the xml Product Spec, sceneCenterGroundRangeSpacing is the
        # 'Nominal ground range spacing in meters between consecutive pixels
        # near mid swath of the RSLC image.'

        range_spacing = freq_group["sceneCenterGroundRangeSpacing"][...]

        # Range in meters (units are specified as meters in the product spec)
        # For NISAR, radar-domain grids are referenced by the center of the
        # pixel, so +/- half the distance of the pixel's side to capture
        # the entire range.
        rng_start = float(freq_group["slantRange"][0]) - 0.5 * range_spacing
        rng_stop = float(freq_group["slantRange"][-1]) + 0.5 * range_spacing

        # output of the next line has format: 'seconds since YYYY-MM-DD HH:MM:SS'
        sec_since_epoch = (
            swaths_group["zeroDopplerTime"].attrs["units"].decode("utf-8")
        )

        # Sanity Check
        format_data = "seconds since %Y-%m-%d %H:%M:%S"
        try:
            datetime.strptime(sec_since_epoch, format_data)
        except ValueError:
            warnings.warn(
                f"Invalid epoch format in input file: {sec_since_epoch}",
                RuntimeWarning,
            )
            # This text should appear in the REPORT.pdf to make it obvious:
            epoch = "INVALID EPOCH"
        else:
            epoch = sec_since_epoch.replace("seconds since ", "").strip()

        return cls(
            data=dataset,
            name=f"{product.upper()}_{band}_{freq}_{pol}",
            band=band,
            freq=freq,
            pol=pol,
            az_spacing=az_spacing,
            az_start=az_start,
            az_stop=az_stop,
            range_spacing=range_spacing,
            rng_start=rng_start,
            rng_stop=rng_stop,
            epoch=epoch,
        )


@dataclass
class GeoRaster(SARRaster):
    """
    A Raster with attributes specific to Geocoded products.

    The attributes specified here are based on the needs of the QA code
    for generating and labeling plots, etc.

    Parameters
    ----------
    data : array_like
        Raster data to be stored.
    name : str
        Name for the dataset
    band : str
        name of the band for `data`, e.g. 'LSAR'
    freq : str
        name of the frequency for `data`, e.g. 'A' or 'B'
    pol : str
        name of the polarization for `data`, e.g. 'HH' or 'HV'
    x_spacing : float
        X spacing of pixels of input array
    x_start : float
        The starting (West) X position of the input array
        This corresponds to the left side of the left-most pixels.
    x_stop : float
        The stopping (East) X position of the input array
        This corresponds to the right side of the right-most pixels.
    y_spacing : float
        Y spacing of pixels of input array
    y_start : float
        The starting (North) Y position of the input array
        This corresponds to the upper edge of the top pixels.
    y_stop : float
        The stopping (South) Y position of the input array
        This corresponds to the lower side of the bottom pixels.

    Notes
    -----
    If data is NISAR HDF5 dataset, suggest initializing using
    the class method init_from_nisar_h5_product(..).
    """

    # Attributes of the input array
    x_spacing: float
    x_start: float
    x_stop: float

    y_spacing: float
    y_start: float
    y_stop: float

    @property
    def y_axis_spacing(self):
        return self.y_spacing

    @property
    def x_axis_spacing(self):
        return self.x_spacing

    @classmethod
    def init_from_nisar_h5_product(cls, h5_file, band, freq, pol):
        """
        Initialize an GeoRaster object for the given
        band-freq-pol image in the input NISAR Geocoded HDF5 file.

        NISAR product type must be one of: 'GSLC', 'GCOV', 'GUNW', 'GOFF'

        Parameters
        ----------
        h5_file : h5py.File
            File handle to a valid NISAR Geocoded product hdf5 file.
            Polarization images must be located in the h5 file in the path:
            /science/<band>/<product name>/grids/frequency<freq>/<pol>
            or they will not be found. This is the file structure
            as determined from the NISAR Product Spec.
        band : str
            name of the band for `img`, e.g. 'LSAR'
        freq : str
            name of the frequency for `img`, e.g. 'A' or 'B'
        pol : str
            name of the polarization for `img`, e.g. 'HH' or 'HV'

        Raises
        ------
        DatasetNotFoundError
            If the file does not contain an image dataset for the given
            band-freq-pol combination, a DatasetNotFoundError
            exception will be thrown.

        Notes
        -----
        The `name` attribute will be populated with a string
        of the format: <product type>_<band>_<freq>_<pol>
        """

        product = nisarqa.get_NISAR_product_type(h5_file)

        if product not in ("GSLC", "GCOV", "GUNW", "GOFF"):
            # self.logger.log_message(logging_base.LogFilterError, 'Invalid file structure.')
            raise nisarqa.InvalidNISARProductError

        # Hardcoded paths to various groups in the NISAR RSLC h5 file.
        # These paths are determined by the .xml product specs
        grids_path = f"/science/{band}/{product}/grids"
        freq_path = f"{grids_path}/frequency{freq}"
        pol_path = f"{freq_path}/{pol}"

        if pol_path in h5_file:
            # self.logger.log_message(logging_base.LogFilterInfo,
            #                         'Found image %s' % band_freq_pol_str)
            pass
        else:
            # self.logger.log_message(logging_base.LogFilterInfo,
            #                         'Image %s not present' % band_freq_pol_str)
            raise nisarqa.DatasetNotFoundError

        # From the xml Product Spec, xCoordinateSpacing is the
        # 'Nominal spacing in meters between consecutive pixels'
        x_spacing = h5_file[freq_path]["xCoordinateSpacing"][...]

        # X in meters (units are specified as meters in the product spec)
        # For NISAR, geocoded grids are referenced by the upper-left corner
        # of the pixel to match GDAL conventions. So add the distance of
        # the pixel's side to far right side to get the actual stop value.
        x_start = float(h5_file[freq_path]["xCoordinates"][0])
        x_stop = float(h5_file[freq_path]["xCoordinates"][-1]) + x_spacing

        # From the xml Product Spec, yCoordinateSpacing is the
        # 'Nominal spacing in meters between consecutive lines'
        y_spacing = h5_file[freq_path]["yCoordinateSpacing"][...]

        # Y in meters (units are specified as meters in the product spec)
        # For NISAR, geocoded grids are referenced by the upper-left corner
        # of the pixel to match GDAL conventions. So add the distance of
        # the pixel's side to bottom to get the actual stop value.
        y_start = float(h5_file[freq_path]["yCoordinates"][0])
        y_stop = float(h5_file[freq_path]["yCoordinates"][-1]) + y_spacing

        # Get dataset object
        if is_complex32(h5_file[pol_path]):
            # As of R3.3 the GSLC workflow recently gained the ability
            # to generate products in complex32 format as well as complex64
            # with some bits masked out to improve compression.
            # If the input GSLC product has dtype complex32, then we'll need
            # to use ComplexFloat16Decoder.
            if product == "GSLC":
                # The GSLC dataset is complex32. h5py >= 3.8 can read these
                # but numpy cannot yet. So, use the ComplexFloat16Decoder.
                ds = ComplexFloat16Decoder(h5_file[pol_path])
                print(
                    "(FAIL) PASS/FAIL Check: Product raster dtype conforms"
                    " to Product Spec dtype of complex64."
                )
            else:
                raise TypeError(
                    f"Input dataset is for a {product} product and "
                    "has dtype complex32. As of R3.3, of the "
                    "geocoded NISAR products, only GSLC products "
                    "can have dtype complex32."
                )
        else:
            # Use h5py's standard reader
            ds = h5_file[pol_path]

            msg = "(%s) PASS/FAIL Check: Product raster dtype conforms to Product Spec dtype of %s."

            if product == "GSLC":
                pass_fail = "PASS" if (ds.dtype == np.complex64) else "FAIL"
                dtype = "complex64"

            elif product == "GCOV":
                if pol[0:2] == pol[2:4]:
                    # on-diagonal term dataset. float32 as of May 2023.
                    dtype = "float32"
                    pass_fail = "PASS" if (ds.dtype == np.float32) else "FAIL"
                else:
                    # off-diagonal term dataset. complex64 as of May 2023.
                    dtype = "complex64"
                    pass_fail = "PASS" if (ds.dtype == np.complex64) else "FAIL"
            else:
                # TODO - for GUNW, and GOFF, confirm that this
                # next print statement is, in fact, true.
                pass_fail = "PASS"
                dtype = "float32"

                if not (ds.dtype == np.float32):
                    warnings.warn(f"Double-check {product=} Product Spec dtype")

            print(msg % (pass_fail, dtype))

        return cls(
            data=ds,
            name=f"{product.upper()}_{band}_{freq}_{pol}",
            band=band,
            freq=freq,
            pol=pol,
            x_spacing=x_spacing,
            x_start=x_start,
            x_stop=x_stop,
            y_spacing=y_spacing,
            y_start=y_start,
            y_stop=y_stop,
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
