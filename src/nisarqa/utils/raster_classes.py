from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import overload

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

        Avoids h5py/numpy dtype bugs and uses numpy float16 -> float32
        conversions which are about 10x faster than HDF5 ones.
        """
        # This avoids h5py exception:
        # TypeError: data type '<c4' not understood
        # Also note this syntax changed in h5py 3.0 and was deprecated in 3.6,
        # see: https://docs.h5py.org/en/stable/whatsnew/3.6.html

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
        """Pixel Spacing in Y direction (azimuth for range-Doppler rasters)"""
        pass

    @property
    @abstractmethod
    def y_axis_limits(self) -> tuple[float, float]:
        """Min and max extents of the Y direction (azimuth for range-Doppler rasters)."""
        pass

    @property
    @abstractmethod
    def y_axis_label(self) -> str:
        """Label for the Y direction (azimuth for range-Doppler rasters)."""
        pass

    @property
    @abstractmethod
    def x_axis_spacing(self):
        """Pixel Spacing in X direction (range for range-Doppler rasters)"""
        pass

    @property
    @abstractmethod
    def x_axis_limits(self) -> tuple[float, float]:
        """Min and max extents of the X direction (range for range-Doppler rasters)."""
        pass

    @property
    @abstractmethod
    def x_axis_label(self) -> str:
        """Label for the X direction (range for range-Doppler rasters)."""
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
    ground_az_spacing : float
        Azimuth spacing of pixels of input array
        Units: meters
    az_start : float
        The start time of the observation for this RSLC Raster.
        This corresponds to the upper edge of the top pixels.
        Units: seconds since epoch
    az_stop : float
        The stopping time of the observation for this RSLC Raster.
        This corresponds to the lower side of the bottom pixels.
        Units: seconds since epoch
    ground_range_spacing : float
        Range spacing of pixels of input array.
        Units: meters
    rng_start : float
        Start (near) distance of the range of input array
        This corresponds to the left side of the left-most pixels.
        Units: meters
    rng_stop : float
        End (far) distance of the range of input array
        This corresponds to the right side of the right-most pixels.
        Units: meters
    epoch : str
        The start of the epoch for this observation,
        in the format 'YYYY-MM-DD HH:MM:SS'
    """

    # Attributes of the input array
    ground_az_spacing: float
    az_start: float
    az_stop: float

    ground_range_spacing: float
    rng_start: float
    rng_stop: float

    epoch: str

    @property
    def y_axis_spacing(self):
        return self.ground_az_spacing

    @property
    def y_axis_limits(self) -> tuple[float, float]:
        return (self.az_start, self.az_stop)

    @property
    def y_axis_label(self) -> str:
        return f"Zero Doppler Time\n(seconds since {self.epoch})"

    @property
    def x_axis_spacing(self):
        return self.ground_range_spacing

    @property
    def x_axis_limits(self) -> tuple[float, float]:
        return (nisarqa.m2km(self.rng_start), nisarqa.m2km(self.rng_stop))

    @property
    def x_axis_label(self) -> str:
        return "Slant Range (km)"


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
        X spacing of pixels (in meters) of input array.
    x_start : float
        The starting (West) X position of the input array
        This corresponds to the left side of the left-most pixels.
    x_stop : float
        The stopping (East) X position of the input array
        This corresponds to the right side of the right-most pixels.
    y_spacing : float
        Y spacing of pixels (in meters) of input array
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
    def y_axis_limits(self) -> tuple[float, float]:
        return (nisarqa.m2km(self.y_start), nisarqa.m2km(self.y_stop))

    @property
    def y_axis_label(self) -> str:
        return "Northing (km)"

    @property
    def x_axis_spacing(self):
        return self.x_spacing

    @property
    def x_axis_limits(self) -> tuple[float, float]:
        return (nisarqa.m2km(self.x_start), nisarqa.m2km(self.x_stop))

    @property
    def x_axis_label(self) -> str:
        return "Easting (km)"


@overload
def compare_raster_metadata(
    raster1: nisarqa.RadarRaster,
    raster2: nisarqa.RadarRaster,
) -> None:
    ...


@overload
def compare_raster_metadata(
    raster1: nisarqa.GeoRaster,
    raster2: nisarqa.GeoRaster,
) -> None:
    ...


def compare_raster_metadata(
    raster1,
    raster2,
):
    """
    Compare the primary metadata and shape of two *Raster instances.

    Compare metadata fields and the shape of the data for two *Raster instances.
    This is useful for checking that two rasters can be combined smoothly into
    a single image, such as combining a phase image raster and a coherence
    magnitude raster to create an HSI image.
    This function does not compare the `name` field nor the values inside the
    data arrays.
    Raises a ValueError if two fields do not match.

    Parameters
    ----------
    raster1, raster2 : nisarqa.RadarRaster | nisarqa.GeoRaster
        *Raster to compare. `raster1` and `raster2` must have the same type.

    Raises
    ------
    ValueError
        If metadata does not match
    """
    if type(raster1) != type(raster1):
        raise TypeError(
            "Input *Rasters must have same type. Type of `raster1`:"
            f" {type(raster1)}, Type of `raster2`: {type(raster2)}."
        )

    for r1, r2 in zip(fields(raster1), fields(raster2)):
        r1_val = getattr(raster1, r1.name)
        r2_val = getattr(raster2, r2.name)

        assert r1.name == r2.name

        if r1.name == "data":
            # raster data layers should have the same shape
            if np.shape(r1_val) != np.shape(r2_val):
                raise ValueError(
                    f"Values do not match: {np.shape(raster1.data)=} but"
                    f" {np.shape(raster2.data)=}."
                )
        elif r1.name == "name":
            # "name" dataclass attributes should be the same
            # except for the final layer name
            if r1_val.split("_")[:-1] != r2_val.split("_")[:-1]:
                warnings.warn(
                    f"{raster1.name=} but {raster2.name=}. Consider checking if"
                    " their band, frequency, polarization, etc. should match."
                )
        elif isinstance(r1_val, str):
            if r1_val != r2_val:
                raise ValueError(
                    f"Values do not match for `{r1.name}` field. `raster1` has"
                    f" value {r1_val}, but `raster2` has value {r2_val}."
                )
        else:
            if np.abs(r1_val - r2_val) > 1e-6:
                raise ValueError(
                    f"Values do not match for `{r1.name}` field. `raster1` has"
                    f" value {r1_val}, but `raster2` has value {r2_val}."
                )


def get_raster_array_with_square_pixels(
    raster_obj: RadarRaster | GeoRaster,
) -> np.ndarray:
    """
    Get *Raster's data array, decimated to square pixels in X and Y direction.

    Parameters
    ----------
    raster_obj : RadarRaster or GeoRaster
        *Raster object whose .data attribute will be read into memory
        and decimated along the first two dimensions to square pixels.

    Returns
    -------
    out : numpy.ndarray
        Copy of raster_obj.data array that has been decimated along the
        first two dimensions to have square pixels.
    """
    arr = raster_obj.data[...]

    ky, kx = nisarqa.compute_square_pixel_nlooks(
        img_shape=arr.shape[:2],
        sample_spacing=[
            raster_obj.y_axis_spacing,
            raster_obj.x_axis_spacing,
        ],
        # Only make square pixels. Use `max()` to not "shrink" the rasters.
        longest_side_max=max(arr.shape[:2]),
    )

    # Decimate to square pixels.
    return arr[::ky, ::kx]


__all__ = nisarqa.get_all(__name__, objects_to_skip)
