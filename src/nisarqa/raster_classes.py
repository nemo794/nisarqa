from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from functools import cached_property
from typing import Any, Optional, Sequence, overload

import h5py
import isce3
import numpy as np
import numpy.typing as npt

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)


class IsComplex(ABC):
    @property
    @abstractmethod
    def is_complex(self) -> bool:
        """True if dataset is complex-valued; False if real-valued."""
        pass


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
        return dataset.dtype == nisarqa.complex32


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

        z = ds.astype(nisarqa.complex32)[key]

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

    @property
    def attrs(self) -> Mapping[str, Any]:
        return self.dataset.attrs

    @property
    def name(self) -> str:
        return self.dataset.name

    def __repr__(self):
        original = self.dataset.__repr__()
        new = original.replace("HDF5 dataset", "ComplexFloat16Decoder")
        return new

    @property
    def chunks(self) -> Sequence[int]:
        return self.dataset.chunks


@dataclass
class RasterStats:
    """
    A set of Min/Max/Mean/STD statistics for a raster.

    Parameters
    ----------
    min_value, max_value, mean_value, std_value : float or None
        Minimum, maximum, mean, and standard deviation values
        (respectively) for a raster image.
        None if a value could not be provided.
    """

    min_value: float | None
    max_value: float | None
    mean_value: float | None
    std_value: float | None


@dataclass
class ComplexRasterStats:
    """
    Min/Max/Mean/STD statistics for a complex-valued raster.

    Parameters
    ----------
    real, imag : RasterStats
        Per ISCE3 convention, for complex-valued data, the statistics
        should be computed independently for the real component and
        for the imaginary component of the data.
        `real` contains the real component's statistics, and `imag` contains
        the imaginary component's statistics.
    """

    real: RasterStats
    imag: RasterStats

    def min_value(self, component: str) -> float:
        """
        Get the minimum of the requested component (either "real" or "imag").
        """
        return getattr(self, component).min_value

    def max_value(self, component: str) -> float:
        """
        Get the maximum of the requested component (either "real" or "imag").
        """
        return getattr(self, component).max_value

    def mean_value(self, component: str) -> float:
        """
        Get the mean value of the requested component (either "real" or "imag").
        """
        return getattr(self, component).mean_value

    def std_value(self, component: str) -> float:
        """
        Get the std. deviation of requested component (either "real" or "imag").
        """
        return getattr(self, component).std_value


@dataclass
class Raster(IsComplex):
    """
    Raster image dataset base class.

    Parameters
    ----------
    data : array_like
        Raster data to be stored. Can be a numpy.ndarray, h5py.Dataset, etc.
    units : str
        The units of the data. If `data` is numeric but unitless (e.g ratios),
        by NISAR convention please use the string "1".
    fill_value : int, float, complex, or None
        The fill value for the dataset. In general, all imagery datasets should
        have a `_FillValue` attribute. The exception might be RSLC (tbd).
    name : str
        Name for the dataset
    stats_h5_group_path : str
        Path in the STATS.h5 file for the group where all metrics and
        statistics re: this raster should be saved.
        Examples:
            RSLC/GSLC/GCOV: "/science/LSAR/QA/data/frequencyA/HH"
            RUNW/GUNW: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/alongTrackOffset"
            ROFF/GOFF: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/layer1/alongTrackOffset"
    band : str
        Name of the band for `img`, e.g. 'LSAR'
    freq : str
        Name of the frequency for `img`, e.g. 'A' or 'B'
    """

    # Raster data. Could be a numpy.ndarray, h5py.Dataset, etc.
    data: npt.ArrayLike
    units: str
    fill_value: Optional[int | float | complex]

    # identifying name of this Raster; can be used for logging
    # e.g. 'LSAR_A_HH'
    name: str

    stats_h5_group_path: str

    band: str
    freq: str

    @cached_property
    def is_complex(self) -> bool:
        """
        True if raster data is complex-valued; False otherwise.

        Returns
        -------
        is_complex_dtype : bool
            True if raster data is complex-valued; False otherwise.

        Warnings
        --------
        If `data` does not have a `dtype` attribute, it will be copied
        into a NumPy array to get the dtype. Depending on the size of the
        raster, this could be an expensive operation.
        """
        try:
            arr_dtype = self.data.dtype
        except AttributeError:
            # Convert input data to a NumPy array only if it isn't already one.
            # If the input is already an ndarray (or a subclass), it returns
            # the input as is, without making a copy.
            arr = np.asanyarray(self.data)
            arr_dtype = arr.dtype

        # Note: Need to use `np.issubdtype` instead of `np.iscomplexobj`
        # due to e.g. RSLC and GSLC datasets of type ComplexFloat16Decoder.
        return np.issubdtype(arr_dtype, np.complexfloating)


@dataclass
class CoordinateGrid(ABC):
    """Abstract Base Class for raster grid dataclasses."""

    @property
    @abstractmethod
    def x_posting(self):
        """
        Posting in X direction of raster grid.

        Notes
        -----
        "Spacing" refers to the (positive-valued) width of a pixel while
        "posting" refers to the (positive- or negative-valued) stride between
        points in a grid. (The spacing is the absolute value of the posting.)
        """
        pass

    @property
    def x_spacing(self):
        """
        Spacing in X direction of raster grid.

        Notes
        -----
        "Spacing" refers to the (positive-valued) width of a pixel while
        "posting" refers to the (positive- or negative-valued) stride between
        points in a grid. (The spacing is the absolute value of the posting.)
        """
        return abs(self.x_posting)

    @property
    @abstractmethod
    def x_pixel_centers(self):
        """
        1D vector of the raster grid's pixel center locations in X direction.
        """
        pass

    @property
    @abstractmethod
    def y_posting(self):
        """
        Posting in Y direction of raster grid.

        Notes
        -----
        "Spacing" refers to the (positive-valued) width of a pixel while
        "posting" refers to the (positive- or negative-valued) stride between
        points in a grid. (The spacing is the absolute value of the posting.)

        For NISAR L2 products, the y-coordinate posting of the
        coordinate grid is negative (the positive y-axis points up in QA plots).
        For NISAR L1 products (i.e. radar grids), the y-coordinate posting
        is positive (the positive y-axis points down in QA plots).
        """
        pass

    @property
    def y_spacing(self):
        """
        Spacing in Y direction of raster grid.

        Notes
        -----
        "Spacing" refers to the (positive-valued) width of a pixel while
        "posting" refers to the (positive- or negative-valued) stride between
        points in a grid. (The spacing is the absolute value of the posting.)
        """
        return abs(self.y_posting)

    @property
    @abstractmethod
    def y_pixel_centers(self):
        """
        1D vector of the raster grid's pixel center locations in Y direction.
        """
        pass

    def _downsample(
        self, y_stride: int, x_stride: int, mode: str
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Downsample this CoordinateGrid's properties by given strides and mode.

        Parameters
        ----------
        y_stride, x_stride : int
            Stride for downsampling along the Y (azimuth) or X (slant range)
            axis, respectively.
        mode : str, optional
            Downsampling algorithm. One of:
                "decimate" : (default) Pure decimation. For example, if
                    `y_stride` is 3 and `x_stride` is 4, then rows 0, 3, ...,
                    and columns 0, 4, ... will be extracted to form
                    the downsampled grid.
                "multilook" : Naive, unweighted multilooking. For example, if
                    `y_stride` is 3 and `x_stride` is 4,
                    then every 3-by-4 window (12 pixels total) will be averaged
                    to form the output pixel.
                    Note that if any of those 12 input pixels is NaN, then the
                    output pixel will be NaN.

        Returns
        -------
        y_coords, x_coords, y_posting, x_posting :
            Downsampled copies of the instance's properties.

        Notes
        -----
        - Coordinates whose index is greater than the largest integer
        multiple of the Y and X stride (along their respective axis)
        will be truncated/ignored.
        """
        if y_stride < 1:
            raise ValueError(f"{y_stride=}, must be >= 1.")
        if x_stride < 1:
            raise ValueError(f"{x_stride=}, must be >= 1.")

        # Downsample to the correct size along the X and Y directions.
        if mode == "decimate":
            y_coords = self.y_pixel_centers[::y_stride]
            x_coords = self.x_pixel_centers[::x_stride]

        elif mode == "multilook":
            y_coords = _get_multilooked_center_coordinates(
                coords=self.y_pixel_centers, nlooks=y_stride
            )
            x_coords = _get_multilooked_center_coordinates(
                coords=self.x_pixel_centers, nlooks=x_stride
            )
        else:
            raise ValueError(
                f"`{mode=}`, only 'decimate' and 'multilook' supported."
            )

        y_post = self.y_posting * y_stride
        assert np.isclose(y_post, y_coords[1] - y_coords[0])

        x_post = self.x_posting * x_stride
        assert np.isclose(x_post, x_coords[1] - x_coords[0])

        return y_coords, x_coords, y_post, x_post


@dataclass
class SARRaster(Raster, ABC):
    """Abstract Base Class for SAR Raster dataclasses."""

    grid: CoordinateGrid

    @property
    def x_posting(self):
        return self.grid.x_posting

    @property
    def x_spacing(self):
        return self.grid.x_spacing

    @property
    def x_pixel_centers(self):
        return self.grid.x_pixel_centers

    @property
    def y_posting(self):
        return self.grid.y_posting

    @property
    def y_spacing(self):
        return self.grid.y_spacing

    @property
    def y_pixel_centers(self):
        return self.grid.y_pixel_centers

    @property
    @abstractmethod
    def y_ground_spacing(self):
        """
        Nominal ground spacing in Y direction, in meters.

        Nominal ground spacing (in meters) in Y direction between
        consecutive pixels near mid swath.
        """
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
    def x_ground_spacing(self):
        """
        Nominal ground spacing in X direction, in meters.

        Nominal ground spacing (in meters) in X direction between
        consecutive pixels near mid swath.
        """
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


def _get_multilooked_center_coordinates(coords: ArrayLike, nlooks: int):
    """
    Get the vector of center coordinates for a multilooked array.

    For odd nlooks, the center falls on a pixel. For even nlooks, the center
    falls between two pixels and requires interpolation.

    Parameters
    ----------
    coords : array_like
        1D array of coordinate values (e.g., x_coordinates, slant_range)
        where values correspond to pixel center.
    nlooks : int
        Number of looks (multilook window size).

    Returns
    -------
    numpy.ndarray
        Coordinate values at the centers of the multilooked blocks
    """
    coords = coords.copy()

    # truncate the coords array
    truncation_amount = len(coords) % nlooks
    if truncation_amount > 0:
        trunc_coords = coords[:-truncation_amount]
    else:
        trunc_coords = coords

    if nlooks % 2 == 1:
        # Odd nlooks: center falls exactly on a pixel
        center_idx = nlooks // 2
        decimated = trunc_coords[center_idx::nlooks]
    else:
        # Even nlooks: center falls between two pixels, need to interpolate
        left_idx = nlooks // 2 - 1
        right_idx = nlooks // 2
        left_coords = trunc_coords[left_idx::nlooks]
        right_coords = trunc_coords[right_idx::nlooks]
        decimated = (left_coords + right_coords) / 2.0

    return decimated


@dataclass
class RadarGrid(CoordinateGrid):
    """
    Uniformly-spaced 2D grid in radar (azimuth time, slant range) coordinates.

    The attributes specified here are based on the needs of the QA code
    for generating and labeling plots, etc.

    Parameters
    ----------
    zero_doppler_time : numpy.ndarray
        1D vector of zero Doppler azimuth times (in seconds) measured relative
        to a UTC epoch. These correspond to the center of each pixel
        of the raster grid in the Y direction.
    zero_doppler_time_spacing : float
        Time interval in the along-track direction of the raster, in seconds.
        This is same as the spacing between consecutive entries in the
        `zero_doppler_time` array.
    slant_range : numpy.ndarray
        1D vector of the slant range values (in meters), corresponding to
        the center of each pixel of the raster grid in the X direction.
    slant_range_spacing : float
        Slant range spacing of grid, in meters. Same as difference between
        consecutive samples in `slant_range` array.
    ground_az_spacing : float
        Scene center azimuth spacing of pixels of the grid, in meters.
    ground_range_spacing : float
        Scene center ground range spacing of pixels of the grid, in meters.
    epoch : str
        The reference epoch for time coordinates in the grid,
        in the format 'YYYY-MM-DDTHH:MM:SS'.

    Attributes
    ----------
    az_start : float
        The start time of the radar grid.
        This corresponds to the upper edge of the top pixels.
        Units: seconds since epoch
    az_stop : float
        The stopping time of the radar grid.
        This corresponds to the lower side of the bottom pixels.
        Units: seconds since epoch
    rng_start : float
        Start (near) range of the radar grid.
        This corresponds to the left side of the left-most pixels.
        Units: meters
    rng_stop : float
        End (far) range of the radar grid.
        This corresponds to the right side of the right-most pixels.
        Units: meters

    Notes
    -----
    Provided initialization parameters will also be stored as attributes.
    """

    # Attributes of the input array
    zero_doppler_time: np.ndarray
    zero_doppler_time_spacing: float

    slant_range: np.ndarray
    slant_range_spacing: float

    ground_az_spacing: float
    ground_range_spacing: float

    epoch: str

    az_start: float = field(init=False)
    az_stop: float = field(init=False)
    rng_start: float = field(init=False)
    rng_stop: float = field(init=False)

    def __post_init__(self):

        self.zero_doppler_time_spacing = float(self.zero_doppler_time_spacing)
        self.slant_range_spacing = float(self.slant_range_spacing)
        self.ground_az_spacing = float(self.ground_az_spacing)
        self.ground_range_spacing = float(self.ground_range_spacing)

        # Infer start and stop values

        # For NISAR, radar-domain grids are referenced by the center of the
        # pixel, so +/- half the distance of the pixel's side to capture
        # the entire range.
        self.az_start = (
            float(self.zero_doppler_time[0])
            - 0.5 * self.zero_doppler_time_spacing
        )
        self.az_stop = (
            float(self.zero_doppler_time[-1])
            + 0.5 * self.zero_doppler_time_spacing
        )

        # For NISAR, radar-domain grids are referenced by the center of the
        # pixel, so +/- half the distance of the pixel's side to capture
        # the entire range.
        self.rng_start = (
            float(self.slant_range[0]) - 0.5 * self.slant_range_spacing
        )
        self.rng_stop = (
            float(self.slant_range[-1]) + 0.5 * self.slant_range_spacing
        )

    @property
    def x_posting(self):
        return self.slant_range_spacing

    @property
    def x_pixel_centers(self):
        return self.slant_range

    @property
    def y_posting(self):
        return self.zero_doppler_time_spacing

    @property
    def y_pixel_centers(self):
        return self.zero_doppler_time

    def get_isce3_radar_grid_parameters(
        self,
        wavelength: float,
        look_side: isce3.core.LookSide | str,
    ) -> isce3.product.RadarGridParameters:
        """
        Generate ISCE3 RadarGridParameters for this RadarRaster.

        This method constructs an isce3.product.RadarGridParameters object
        from the radar coordinate information stored in this RadarRaster.

        Parameters
        ----------
        wavelength : float
            The radar central wavelength, in meters.
        look_side : isce3.core.LookSide or {'left', 'right'}
            The look direction of the radar (left-looking or right-looking).

        Returns
        -------
        isce3.product.RadarGridParameters
            ISCE3 radar grid parameters object representing this raster's
            coordinate system.
        """
        # Parse the epoch string to create an ISCE3 DateTime object
        ref_epoch = isce3.core.DateTime(self.epoch)

        # Convert look_side to ISCE3 LookSide enum if it's a string
        # TODO - check if it is ok to skip conversion and pass the string
        if isinstance(look_side, str):
            if look_side.lower() == "left":
                look_side = isce3.core.LookSide.Left
            elif look_side.lower() == "right":
                look_side = isce3.core.LookSide.Right
            else:
                raise ValueError(f"Invalid look_side string: {look_side}")

        # Compute PRF from zero doppler time spacing
        prf = 1.0 / self.zero_doppler_time_spacing

        # Create the RadarGridParameters object
        radar_grid = isce3.product.RadarGridParameters(
            sensing_start=float(self.zero_doppler_time[0]),
            wavelength=float(wavelength),
            prf=float(prf),
            starting_range=float(self.slant_range[0]),
            range_pixel_spacing=float(self.slant_range_spacing),
            lookside=look_side,
            length=len(self.zero_doppler_time),
            width=len(self.slant_range),
            ref_epoch=ref_epoch,
        )

        return radar_grid

    def downsample(self, y_stride: int, x_stride: int, mode: str) -> RadarGrid:
        """
        Downsample grid by the given strides and mode.

        Parameters
        ----------
        y_stride, x_stride : int
            Stride for downsampling along the Y (azimuth) or X (slant range)
            axis, respectively.
        mode : str, optional
            Downsampling algorithm. One of:
                "decimate" : (default) Pure decimation. For example, if
                    `y_stride` is 3 and `x_stride` is 4, then rows 0, 3, ...,
                    and columns 0, 4, ... will be extracted to form
                    the downsampled grid.
                "multilook" : Naive, unweighted multilooking. For example, if
                    `y_stride` is 3 and `x_stride` is 4,
                    then every 3-by-4 window (12 pixels total) will be averaged
                    to form the output pixel.
                    Note that if any of those 12 input pixels is NaN, then the
                    output pixel will be NaN.

        Returns
        -------
        downsampled_grid : RadarGrid
            Copy of this RadarGrid instance which has been downsampled.

        Notes
        -----
        - Coordinates whose index is greater than the largest integer
        multiple of the Y and X stride (along their respective axis)
        will be truncated/ignored.
        """

        zdt_coords, sr_coords, zdt_spacing, sr_spacing = self._downsample(
            y_stride=y_stride, x_stride=x_stride, mode=mode
        )

        grd_az_spacing = self.ground_az_spacing * y_stride
        grd_rg_spacing = self.ground_range_spacing * x_stride

        return RadarGrid(
            zero_doppler_time=zdt_coords,
            zero_doppler_time_spacing=zdt_spacing,
            slant_range=sr_coords,
            slant_range_spacing=sr_spacing,
            ground_az_spacing=grd_az_spacing,
            ground_range_spacing=grd_rg_spacing,
            epoch=self.epoch,
        )

    def get_latlonquad(
        self, orbit, wavelength, look_side, dem_file
    ) -> nisarqa.LatLonQuad:
        """
        Get the LatLonQuad for this RadarGrid.
        """
        return nisarqa.compute_latlonquad_from_radar_coords(
            slant_range=self.slant_range,
            zero_doppler_time=self.zero_doppler_time,
            orbit=orbit,
            wavelength=wavelength,
            look_side=look_side,
            dem_file=dem_file,
        )

    def save_kml(
        self,
        *,
        orbit,
        wavelength,
        look_side,
        dem_file,
        output_dir,
        kml_filename,
        png_filename,
    ) -> None:
        """
        Save a KML with a lonlatquad corresponding to this RadarGrid.
        """

        llq = self.get_latlonquad(
            orbit=orbit,
            wavelength=wavelength,
            look_side=look_side,
            dem_file=dem_file,
        )

        nisarqa.write_latlonquad_to_kml(
            llq=llq,
            output_dir=output_dir,
            kml_filename=kml_filename,
            png_filename=png_filename,
        )


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
    units : str
        The units of the data. If `data` is numeric but unitless (e.g ratios),
        by NISAR convention please use the string "1".
    fill_value : int, float, complex, or None
        The fill value for the dataset. In general, all imagery datasets should
        have a `_FillValue` attribute. The exception might be RSLC (tbd).
    name : str
        Name for the dataset
    stats_h5_group_path : str
        Path in the STATS.h5 file for the group where all metrics and
        statistics re: this raster should be saved.
        Examples:
            RSLC/GSLC/GCOV: "/science/LSAR/QA/data/frequencyA/HH"
            RUNW/GUNW: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/alongTrackOffset"
            ROFF/GOFF: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/layer1/alongTrackOffset"
    band : str
        name of the band for `img`, e.g. 'LSAR'
    freq : str
        name of the frequency for `img`, e.g. 'A' or 'B'
    grid : RadarGrid
        The radar coordinate grid for this raster. This contains all the
        radar-specific coordinate information (zero_doppler_time, slant_range,
        spacing values, epoch, etc.).
    """

    grid: RadarGrid

    @property
    def epoch(self) -> str:
        return self.grid.epoch

    @property
    def y_ground_spacing(self) -> float:
        return self.grid.ground_az_spacing

    @property
    def y_axis_limits(self) -> tuple[float, float]:
        return (self.grid.az_start, self.grid.az_stop)

    @property
    def y_axis_label(self) -> str:
        return f"Zero Doppler Time\n(seconds since {self.grid.epoch})"

    @property
    def x_ground_spacing(self) -> float:
        return self.grid.ground_range_spacing

    @property
    def x_axis_limits(self) -> tuple[float, float]:
        return (
            nisarqa.m2km(self.grid.rng_start),
            nisarqa.m2km(self.grid.rng_stop),
        )

    @property
    def x_axis_label(self) -> str:
        return "Slant Range (km)"


@dataclass
class GeoGrid(CoordinateGrid):
    """
    Uniformly-spaced 2D grid in geocoded (e.g. projected) coordinates.

    The attributes specified here are based on the needs of the QA code
    for generating and labeling plots, etc.

    Parameters
    ----------
    epsg : int
        The EPSG code of the coordinate system.
    x_axis_posting : float
        X posting of pixels of the grid, in units matching `x_coordinates`.
    x_coordinates : numpy.ndarray
        1D vector of the coordinate values of the center of each pixel
        of the raster grid in the X direction, in the units of the
        coordinate reference system represented by `epsg`.
    y_axis_posting : float
        Y posting of pixels of the grid, in units matching `y_coordinates`.
        Note: For NISAR L2 products, the y-coordinate posting of the
        coordinate grid is negative (the positive y-axis points up in QA plots).
    y_coordinates : numpy.ndarray
        1D vector of the coordinate values of the center of each pixel
        of the raster grid in the Y direction, in the units of the
        coordinate reference system represented by `epsg`.

    Attributes
    ----------
    x_start : float
        The starting (typically West) X position of the grid.
        This corresponds to the left side of the left-most pixels.
    x_stop : float
        The stopping (typically East) X position of the grid.
        This corresponds to the right side of the right-most pixels.
    y_start : float
        The starting (typically North) Y position of the grid.
        This corresponds to the upper edge of the top pixels.
    y_stop : float
        The stopping (typically South) Y position of the grid.
        This corresponds to the lower side of the bottom pixels.

    Notes
    -----
    Provided initialization parameters will also be stored as attributes.
    """

    epsg: int

    # Call this `x_axis_posting` instead of simply `x_posting` to avoid
    # name collisions with the property in the base class.
    x_axis_posting: float
    x_coordinates: np.ndarray

    # Call this `y_axis_posting` per reasoning above
    y_axis_posting: float
    y_coordinates: np.ndarray

    x_start: float = field(init=False)
    x_stop: float = field(init=False)
    y_start: float = field(init=False)
    y_stop: float = field(init=False)

    def __post_init__(self):
        self.epsg = int(self.epsg)
        self.x_axis_posting = float(self.x_axis_posting)
        self.y_axis_posting = float(self.y_axis_posting)

        # Infer start and stop values

        # For NISAR, geocoded grids are referenced by the center
        # of the pixel (different from GDAL conventions!). So add half
        # of the pixels' spacing to get the grid's far edges.
        # Note: `isce3.product.GeoGridParameters` adopts the GDAL convention
        # of referencing by the upper-left corner of the pixel, but the
        # NISAR L2 product writers construct `xCoordinates` and `yCoordinates`
        # to refer to the location of the center of the pixels.
        self.x_start = float(self.x_coordinates[0] - 0.5 * self.x_posting)
        self.x_stop = float(self.x_coordinates[-1] + 0.5 * self.x_posting)

        # Use the posting (not the spacing) to capture the
        # negative-valued stride of the y-coordinates for NISAR L2 products.
        self.y_start = float(self.y_coordinates[0] - 0.5 * self.y_posting)
        self.y_stop = float(self.y_coordinates[-1] + 0.5 * self.y_posting)

    @property
    def x_posting(self):
        return self.x_axis_posting

    @property
    def x_pixel_centers(self):
        return self.x_coordinates

    @property
    def y_posting(self):
        return self.y_axis_posting

    @property
    def y_pixel_centers(self):
        return self.y_coordinates

    def to_isce3_geo_grid_parameters(self) -> isce3.product.GeoGridParameters:
        """
        Generate ISCE3 GeoGridParameters for this GeoRaster.

        This method constructs an isce3.product.GeoGridParameters object
        from the geocoded coordinate information stored in this GeoRaster.

        Returns
        -------
        isce3.product.GeoGridParameters
            ISCE3 geo grid parameters object representing this raster's
            coordinate system.

        Notes
        -----
        ISCE3's GeoGridParameters uses the GDAL convention where coordinate
        values reference the upper-left corner of pixels, while NISAR products
        reference pixel centers.
        """

        # ISCE3 GeoGridParameters expects the starting coordinates to represent
        # the upper-left corner of the first pixel (GDAL convention).
        geogrid = isce3.product.GeoGridParameters(
            start_x=self.x_start,
            start_y=self.y_start,
            spacing_x=float(self.x_posting),
            spacing_y=float(self.y_posting),
            width=len(self.x_coordinates),
            length=len(self.y_coordinates),
            epsg=int(self.epsg),
        )

        return geogrid

    @classmethod
    def from_isce3_geo_grid(
        cls, isce3_geogrid: isce3.product.GeoGridParameters
    ) -> GeoGrid:
        """
        Construct a nisarqa.GeoGrid from a ISCE3 GeoGridParameters instance.

        ISCE3 GeoGridParameters uses upper-left corner convention (like GDAL),
        but nisarqa.GeoGrid uses pixel center convention.

        Parameters
        ----------
        isce3_geogrid : isce3.product.GeoGridParameters
            An ISCE3 GeoGridParameters instance, which uses the upper-left
            corner convention (like GDAL).

        Returns
        -------
        qa_geogrid : nisarqa.GeoGrid
            A nisarqa GeoGridParameters instance, which uses the pixel center
            convention.
        """
        # Convert from corner to center coordinates:
        # geogrid.start_x = x-coordinate of upper-left corner of upper-left pixel
        # geogrid.start_y = y-coordinate of upper-left corner of upper-left pixel
        # geogrid.spacing_x = pixel width (x spacing)
        # geogrid.spacing_y = pixel height (y spacing, negative for north-up)

        # Calculate pixel center coordinates
        x_coords = (
            isce3_geogrid.start_x
            + (np.arange(isce3_geogrid.width) + 0.5) * isce3_geogrid.spacing_x
        )
        y_coords = (
            isce3_geogrid.start_y
            + (np.arange(isce3_geogrid.length) + 0.5) * isce3_geogrid.spacing_y
        )

        epsg = str(isce3_geogrid.epsg)

        return cls(
            epsg=epsg,
            x_axis_posting=isce3_geogrid.spacing_x,
            x_coordinates=x_coords,
            y_axis_posting=abs(isce3_geogrid.spacing_y),
            y_coordinates=y_coords,
        )

    @classmethod
    def from_coordinates(
        cls,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        epsg: int,
    ) -> GeoGrid:
        """
        Construct a GeoGrid from coordinate vectors.

        The provided arguements should all be consistent with each other for a
        given raster on a geocoded grid (e.g. NISAR Level-2 products).

        Parameters
        ----------
        x_coords : numpy.ndarray
            1D array of X coordinate values (in units corresponding to `epsg`)
            for the given raster. Values should correspond to pixel centers.
        y_coords : numpy.ndarray
            1D array of Y coordinate values (in units corresponding to `epsg`)
            for the given raster. Values should correspond to pixel centers.
        epsg : int
            The EPSG code of the projected coordinate system for the given raster
            (e.g., 32610 for UTM Zone 10N, or 3413 for NSIDC Sea Ice Polar
            Stereographic North).

        Returns
        -------
        LatLonQuad
            A LatLonQuad object for the given raster, with longitude normalization
            applied for proper antimeridian handling.

        Returns
        -------
        qa_geogrid : nisarqa.GeoGrid
            A nisarqa GeoGridParameters instance, which uses the pixel center
            convention.
        """

        return cls(
            epsg=epsg,
            x_axis_posting=x_coords[1] - x_coords[0],
            x_coordinates=x_coords,
            y_axis_posting=y_coords[1] - y_coords[0],
            y_coordinates=y_coords,
        )

    def downsample(self, y_stride: int, x_stride: int, mode: str) -> GeoGrid:
        """
        Downsample grid by the given strides and mode.

        Parameters
        ----------
        y_stride, x_stride : int
            Stride for downsampling along the Y (azimuth) or X (slant range)
            axis, respectively.
        mode : str, optional
            Downsampling algorithm. One of:
                "decimate" : (default) Pure decimation. For example, if
                    `x_stride` is 3 and `y_stride` is 4, then starting
                    with and including row 0 and column 0, every 3rd row and
                    4th column will be extracted to form the downsampled image.
                "multilook" : Naive, unweighted multilooking. For example, if
                    `x_stride` is 3 and `y_stride` is 4,
                    then every 3-by-4 window (12 pixels total) will be averaged
                    to form the output pixel.
                    Note that if any of those 12 input pixels is NaN, then the
                    output pixel will be NaN.

        Returns
        -------
        downsampled_grid : GeoGrid
            Copy of this GeoGrid instance which has been downsampled.
        """

        y_coords, x_coords, y_posting, x_posting = self._downsample(
            y_stride=y_stride, x_stride=x_stride, mode=mode
        )

        return GeoGrid(
            epsg=self.epsg,
            x_axis_posting=x_posting,
            x_coordinates=x_coords,
            y_axis_posting=y_posting,
            y_coordinates=y_coords,
        )

    def get_latlonquad(self) -> nisarqa.LatLonQuad:
        """
        Get the LatLonQuad for this Geogrid.
        """
        return nisarqa.compute_latlonquad_from_geo_coords(
            x_coords=self.x_coordinates,
            y_coords=self.y_coordinates,
            epsg=self.epsg,
        )

    def save_kml(
        self,
        *,
        output_dir,
        kml_filename,
        png_filename,
    ) -> None:
        """
        Save a KML with a lonlatquad corresponding to this GeoGrid.
        """

        nisarqa.write_latlonquad_to_kml(
            llq=self.get_latlonquad(),
            output_dir=output_dir,
            kml_filename=kml_filename,
            png_filename=png_filename,
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
        Raster data to be stored, aka the input array.
    units : str
        The units of the data. If `data` is numeric but unitless (e.g ratios),
        by NISAR convention please use the string "1".
    fill_value : int, float, complex, or None
        The fill value for the dataset. In general, all imagery datasets should
        have a `_FillValue` attribute. The exception might be RSLC (tbd).
    name : str
        Name for the dataset
    stats_h5_group_path : str
        Path in the STATS.h5 file for the group where all metrics and
        statistics re: this raster should be saved.
        Examples:
            RSLC/GSLC/GCOV: "/science/LSAR/QA/data/frequencyA/HH"
            RUNW/GUNW: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/alongTrackOffset"
            ROFF/GOFF: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/layer1/alongTrackOffset"
    band : str
        name of the band for `data`, e.g. 'LSAR'
    freq : str
        name of the frequency for `data`, e.g. 'A' or 'B'
    grid : GeoGrid
        The geocoded coordinate grid for this raster. This contains all the
        geo-specific coordinate information (epsg, x_coordinates, y_coordinates,
        posting values, etc.).
    """

    grid: GeoGrid

    @property
    def epsg(self) -> int:
        return self.grid.epsg

    # SARRaster-specific properties
    @property
    def y_ground_spacing(self) -> float:
        return self.grid.y_spacing

    @property
    def y_axis_limits(self) -> tuple[float, float]:
        return (nisarqa.m2km(self.grid.y_start), nisarqa.m2km(self.grid.y_stop))

    @property
    def y_axis_label(self) -> str:
        return f"Y Coordinate, EPSG:{self.grid.epsg} (km)"

    @property
    def x_ground_spacing(self) -> float:
        return self.grid.x_spacing

    @property
    def x_axis_limits(self) -> tuple[float, float]:
        return (nisarqa.m2km(self.grid.x_start), nisarqa.m2km(self.grid.x_stop))

    @property
    def x_axis_label(self) -> str:
        return f"X Coordinate, EPSG:{self.grid.epsg} (km)"


@overload
def compare_raster_metadata(
    raster1: nisarqa.RadarRaster,
    raster2: nisarqa.RadarRaster,
    almost_identical: bool,
) -> None: ...


@overload
def compare_raster_metadata(
    raster1: nisarqa.GeoRaster,
    raster2: nisarqa.GeoRaster,
    almost_identical: bool,
) -> None: ...


def compare_raster_metadata(raster1, raster2, almost_identical=True):
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
        *Raster to compare. `raster1` and `raster2` must either both be
        instances of RadarRasters or both be instances of GeoRasters.
    almost_identical : bool, optional
        True if the two inputs rasters are expected to have identical metadata
        (except for the layer name).
            Ex 1: RSLC's frequency A HH Raster vs frequency A VV Raster
            Ex 2: RIFG's freq A HH alongTrackOffset Raster vs
                  freq A HH slantRangeOffset Raster
        False if the two input rasters are expected to have more differences in their
        metadata, and should only have matching shape, spacing, etc.
        In practice, this suppresses warnings about expected dissimilar
        fields, such as `units`.
            Ex: RIFG's freq A HH wrappedInterferogram Raster vs freq A HH
                coherenceMagnitude Raster
        Defaults to True (more verbose).

    Raises
    ------
    ValueError
        If metadata does not match
    """
    # This function only compares fields in instances of the RadarRaster or
    # GeoRaster base classes. Child classes have additional fields, which we
    # need to ignore when comparing the metadata of the two input *Rasters.
    r_raster = nisarqa.RadarRaster
    g_raster = nisarqa.GeoRaster
    if isinstance(raster1, r_raster) and isinstance(raster2, r_raster):
        raster_class = r_raster
    elif isinstance(raster1, g_raster) and isinstance(raster2, g_raster):
        raster_class = g_raster
    else:
        raise TypeError(
            f"{type(raster1)=} and {type(raster2)=}, must both be instances"
            " of either RadarRaster or of GeoRaster."
        )

    for f in fields(raster_class):
        field_name = f.name
        r1_val = getattr(raster1, field_name)
        r2_val = getattr(raster2, field_name)

        log = nisarqa.get_logger()

        if field_name == "data":
            # raster data layers should have the same shape
            if np.shape(r1_val) != np.shape(r2_val):
                raise ValueError(
                    f"Values do not match: {np.shape(raster1.data)=} but"
                    f" {np.shape(raster2.data)=}."
                )
        elif field_name == "units":
            if not almost_identical:
                if raster1.units != raster2.units:
                    log.warning(
                        f"Layer `{raster1.name}` has units attribute of"
                        f" `{raster1.units}`, and is being compared to layer"
                        f" `{raster2.name}` which has units attribute of"
                        f" `{raster2.units}`. Please confirm these two rasters"
                        " are ok to have different units."
                    )
        elif field_name == "name":
            # "name" dataclass attributes should be the same
            # except for the final layer name
            if r1_val.split("_")[:-1] != r2_val.split("_")[:-1]:
                log.warning(
                    f"{raster1.name=} but {raster2.name=}. Consider checking if"
                    " their band, frequency, polarization, etc. should match."
                )
        elif field_name == "stats_h5_group_path":
            # "stats_h5_group_path" dataclass attributes should be the same
            # except for the final layer name
            if r1_val.split("/")[:-1] != r2_val.split("/")[:-1]:
                log.warning(
                    f"{raster1.stats_h5_group_path=} but"
                    f" {raster2.stats_h5_group_path=}. Consider checking if"
                    " these base paths should match."
                )
        elif isinstance(r1_val, str):
            if r1_val != r2_val:
                raise ValueError(
                    f"Values do not match for `{field_name}` field. `raster1`"
                    f" has value {r1_val}, but `raster2` has value {r2_val}."
                )
        elif isinstance(r1_val, nisarqa.CoordinateGrid):
            for grid_f in fields(r1_val):
                field_name = grid_f.name
                grid1_val = getattr(r1_val, field_name)
                grid2_val = getattr(r2_val, field_name)
            if grid1_val != grid2_val:
                raise ValueError(
                    f"Values do not match for Coordinate Grid's `{field_name}`"
                    f" field. `raster1`'s grid has value {r1_val},"
                    f" but `raster2`'s grid has value {r2_val}."
                )
        else:
            if np.any(np.abs(r1_val - r2_val) > 1e-6):
                raise ValueError(
                    f"Values do not match for `{field_name}` field. `raster1`"
                    f" has value {r1_val}, but `raster2` has value {r2_val}."
                )


def decimate_raster_array_to_square_pixels(
    raster_obj: RadarRaster | GeoRaster,
) -> np.ndarray:
    """
    Decimate *Raster's data array to approx. square pixels in X and Y direction.

    Parameters
    ----------
    raster_obj : RadarRaster or GeoRaster
        *Raster object whose .data attribute will be read into memory
        and decimated along the first two dimensions to approx. square pixels.

    Returns
    -------
    out : numpy.ndarray
        Copy of raster_obj.data array that has been decimated along the
        first two dimensions to have approx. square pixels.
    """
    # Decimate to square pixels.
    return decimate_array_to_square_pixels(
        arr=np.array(raster_obj.data, copy=True),
        y_axis_spacing=raster_obj.y_ground_spacing,
        x_axis_spacing=raster_obj.x_ground_spacing,
    )


def decimate_raster_array_to_square_pixels_with_strides(
    raster_obj: RadarRaster | GeoRaster,
) -> tuple[np.ndarray, int, int]:
    """
    Decimate *Raster's data array to square pixels and also return strides.

    Pixels will be approx. square in the X and Y direction.

    Parameters
    ----------
    raster_obj : RadarRaster or GeoRaster
        *Raster object whose .data attribute will be read into memory
        and decimated along the first two dimensions to approx. square pixels.

    Returns
    -------
    out : numpy.ndarray
        Copy of raster_obj.data array that has been decimated along the
        first two dimensions to have approx. square pixels.
    ky, kx : int
        The stride used for performing decimation in the Y and X directions,
        respectively.
    """
    # Decimate to square pixels.
    return decimate_array_to_square_pixels_with_strides(
        arr=np.array(raster_obj.data, copy=True),
        y_axis_spacing=raster_obj.y_ground_spacing,
        x_axis_spacing=raster_obj.x_ground_spacing,
    )


def decimate_array_to_square_pixels(
    arr: np.ndarray, y_axis_spacing: float, x_axis_spacing: float
) -> np.ndarray:
    """
    Decimate array to approx. square pixels in X and Y direction.

    Parameters
    ----------
    arr : np.ndarray
        Input array to be decimated along the first two dimensions to
        approx. square pixels.
    y_axis_spacing : float
        Pixel Spacing in Y direction (azimuth for range-Doppler rasters).
        Must be in same units as `x_axis_spacing`.
    x_axis_spacing : float
        Pixel Spacing in X direction (range for range-Doppler rasters).
        Must be in same units as `y_axis_spacing`.

    Returns
    -------
    out : numpy.ndarray
        View of `arr` that has been decimated along the first two
        dimensions to have approx. square pixels.
    """
    out, _, _ = decimate_array_to_square_pixels_with_strides(
        arr=arr, y_axis_spacing=y_axis_spacing, x_axis_spacing=x_axis_spacing
    )

    return out


def decimate_array_to_square_pixels_with_strides(
    arr: np.ndarray, y_axis_spacing: float, x_axis_spacing: float
) -> tuple[np.ndarray, int, int]:
    """
    Decimate array to approx. square pixels in X and Y direction.

    Parameters
    ----------
    arr : np.ndarray
        Input array to be decimated along the first two dimensions to
        approx. square pixels.
    y_axis_spacing : float
        Pixel Spacing in Y direction (azimuth for range-Doppler rasters).
        Must be in same units as `x_axis_spacing`.
    x_axis_spacing : float
        Pixel Spacing in X direction (range for range-Doppler rasters).
        Must be in same units as `y_axis_spacing`.

    Returns
    -------
    out : numpy.ndarray
        View of `arr` that has been decimated along the first two
        dimensions to have approx. square pixels.
    ky, kx : int
        The stride used for performing decimation in the Y and X directions,
        respectively.
    """

    ky, kx = nisarqa.compute_square_pixel_nlooks(
        img_shape=arr.shape[:2],
        sample_spacing=[
            y_axis_spacing,
            x_axis_spacing,
        ],
        # Only make square pixels. Use `max()` to not "shrink" the rasters.
        longest_side_max=max(arr.shape[:2]),
    )

    # Decimate to square pixels.
    return arr[::ky, ::kx], ky, kx


@dataclass
class StatsMixin(IsComplex):
    """
    A mixin type providing access to dataset statistics (Min/Max/Mean/STD).

    Parameters
    ----------
    stats : nisarqa.RasterStats or nisarqa.ComplexRasterStats
        Statistics of the `data` array.
    """

    stats: RasterStats | ComplexRasterStats

    def __post_init__(self) -> None:
        if self.is_complex:
            if not isinstance(self.stats, ComplexRasterStats):
                raise TypeError(
                    f"`data` provided is complex-valued, so `stats` must"
                    f" be an instance of ComplexRasterStats. Dataset: {self.name}"
                )
        else:
            if not isinstance(self.stats, RasterStats):
                raise TypeError(
                    f"`data` provided is real-valued, so `stats` must"
                    " be an instance of RasterStats."
                )

    def get_stat(self, stat: str, component: str | None = None) -> float:
        """
        Return value for a min/max/mean/std metric.

        Parameters
        ----------
        stat : str
            One of {"min", "max", "mean", "std"}.
        component : str or None, optional
            One of "real", "imag", or None.
            Per ISCE3 convention, statistics for complex-valued data should be
            computed independently for the real and imaginary components.
            If the raster is real-valued, set this to None.
            If the raster is complex-valued, set to "real" or "imag" for the
            real or imaginery component's metric (respectively).
            Defaults to None.

        Returns
        -------
        statistic : float
            The value of the requested statistic.
        """
        stat_opts = ("min", "max", "mean", "std")
        if stat not in stat_opts:
            raise ValueError(f"{stat=!r}, must be one of {stat_opts}.")

        if self.is_complex:
            if component not in ("real", "imag"):
                raise ValueError(
                    f"{component=!r}, but Raster is complex-valued. Set to"
                    " either 'real' or 'imag'."
                )
            return getattr(getattr(self.stats, component), f"{stat}_value")

        else:
            if component is not None:
                raise ValueError(
                    f"{component=!r}, but Raster is real-valued. Set to None."
                )
            return getattr(self.stats, f"{stat}_value")

    def get_stat_val_name_descr(
        self, stat: str, component: str | None = None
    ) -> tuple[float, str, str]:
        """
        Return value, name, and description for a min/max/mean/std metric.

        The name and description returned follow the NISAR conventions for
        saving min/max/mean/std metrics in NISAR L1/L2 HDF5 metadata.

        Parameters
        ----------
        stat : str
            One of {"min", "max", "mean", "std"}.
        component : str or None, optional
            One of "real", "imag", or None.
            Per ISCE3 convention, statistics for complex-valued data should be
            computed independently for the real and imaginary components.
            If the raster is real-valued, set this to None.
            If the raster is complex-valued, set to "real" or "imag" for the
            real or imaginery component's metric (respectively).
            Defaults to None.

        Returns
        -------
        val, name, descr : float, str, str
            The value, name, and description of the requested statistic.
        """

        val = self.get_stat(stat=stat, component=component)
        name, descr = nisarqa.get_stats_name_descr(
            stat=stat, component=component
        )
        return val, name, descr


@dataclass
class RadarRasterWithStats(RadarRaster, StatsMixin):
    """
    A RadarRaster with statistics.

    Parameters
    ----------
    data : array_like
        Raster data to be stored. Can be a numpy.ndarray, h5py.Dataset, etc.
    units : str
        The units of the data. If `data` is numeric but unitless (e.g ratios),
        by NISAR convention please use the string "1".
    fill_value : int, float, complex, or None
        The fill value for the dataset. In general, all imagery datasets should
        have a `_FillValue` attribute. The exception might be RSLC (tbd).
    name : str
        Name for the dataset
    stats_h5_group_path : str
        Path in the STATS.h5 file for the group where all metrics and
        statistics re: this raster should be saved.
        Examples:
            RSLC/GSLC/GCOV: "/science/LSAR/QA/data/frequencyA/HH"
            RUNW/GUNW: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/alongTrackOffset"
            ROFF/GOFF: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/layer1/alongTrackOffset"
    band : str
        Name of the band for `img`, e.g. 'LSAR'
    freq : str
        Name of the frequency for `img`, e.g. 'A' or 'B'
    grid : RadarGrid
        The radar coordinate grid for this raster. This contains all the
        radar-specific coordinate information (zero_doppler_time, slant_range,
        spacing values, epoch, etc.).
    stats : nisarqa.RasterStats or nisarqa.ComplexRasterStats
        Statistics of the `data` array.
    """

    def __post_init__(self):
        # Call post init of parent classes
        # (RadarRaster does not have a __post_init__)
        StatsMixin.__post_init__(self)


@dataclass
class GeoRasterWithStats(GeoRaster, StatsMixin):
    """
    A GeoRaster with statistics.

    Parameters
    ----------
    data : array_like
        Raster data to be stored. Can be a numpy.ndarray, h5py.Dataset, etc.
    units : str
        The units of the data. If `data` is numeric but unitless (e.g ratios),
        by NISAR convention please use the string "1".
    fill_value : int, float, complex, or None
        The fill value for the dataset. In general, all imagery datasets should
        have a `_FillValue` attribute. The exception might be RSLC (tbd).
    name : str
        Name for the dataset
    stats_h5_group_path : str
        Path in the STATS.h5 file for the group where all metrics and
        statistics re: this raster should be saved.
        Examples:
            RSLC/GSLC/GCOV: "/science/LSAR/QA/data/frequencyA/HH"
            RUNW/GUNW: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/alongTrackOffset"
            ROFF/GOFF: "/science/LSAR/QA/data/frequencyA/pixelOffsets/HH/layer1/alongTrackOffset"
    band : str
        Name of the band for `img`, e.g. 'LSAR'
    freq : str
        Name of the frequency for `img`, e.g. 'A' or 'B'
    grid : GeoGrid
        The geocoded coordinate grid for this raster. This contains all the
        geo-specific coordinate information (epsg, x_coordinates, y_coordinates,
        posting values, etc.).
    stats : nisarqa.RasterStats or nisarqa.ComplexRasterStats
        Statistics of the `data` array.
    """

    def __post_init__(self):
        # Call post init of parent classes
        # (GeoRaster does not have a __post_init__)
        StatsMixin.__post_init__(self)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
