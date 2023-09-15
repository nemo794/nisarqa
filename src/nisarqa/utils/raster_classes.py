from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    """

    # Raster data. Could be a numpy.ndarray, h5py.Dataset, etc.
    data: npt.ArrayLike

    # identifying name of this Raster; can be used for logging
    # e.g. 'LSAR_A_HH'
    name: str

    band: str
    freq: str


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


__all__ = nisarqa.get_all(__name__, objects_to_skip)
