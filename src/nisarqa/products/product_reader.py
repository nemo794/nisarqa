from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property, lru_cache
from typing import Optional

import h5py
import isce3
import nisar
import numpy as np
import shapely
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def _get_path_to_nearest_dataset(
    h5_file: h5py.File, starting_path: str, dataset_to_find: str
) -> str:
    """
    Get path to the occurrence of `dataset_to_find` nearest to `starting_path`.

    Walking up each parent directory from right to left, this function
    searches each parent directory in `starting_path` (non-recursive)
    to find the first occurrence of `dataset_to_find`, and then returns
    the full path to that dataset.
    By design, this is not a recursive nor exhaustive search of
    the input file. In many NISAR products, `dataset_to_find` can occur
    in multiple groups, and the goal of this function
    is to find the occurrence of `dataset_to_find` that is in closest
    proximity to the dataset located at `starting_path`.

    Parameters
    ----------
    h5_file : h5py.File
        Handle to the input product h5 file.
    starting_path : str
        Path to the starting dataset. This function will iterate up through
        each successive parent directory in `starting_path` to find the first
        occurrence of `dataset_to_find`.
    dataset_to_find : str
        Base name of the dataset to locate in `h5_file`.

    Returns
    -------
    path : str
        Path inside `h5_file` to the requested `dataset_to_find`.

    Raises
    ------
    nisarqa.DatasetNotFoundError
        If `dataset_to_find` is not found.

    Examples
    --------
    Example setup, using an InSAR product

    >>> import h5py
    >>> in_file = h5py.File("RIFG_product.h5", "r")

    Initial path is to a specific raster layer (dataset).

    >>> path = "/science/LSAR/RIFG/swaths/frequencyA/interferogram/HH/wrappedInterferogram"

    First, find the coherence magnitude layer that (likely) corresponds to
    that wrapped interferogram dataset. Since this function searches the
    dataset's parent directory first, it locates the coherence magnitude
    layer inside the same directory as our provided path.

    >>> name = "coherenceMagnitude"
    >>> nisarqa.get_path_to_nearest_dataset(in_file, path, name)
    '/science/LSAR/RIFG/swaths/frequencyA/interferogram/HH/coherenceMagnitude'

    Now, find the "zeroDopplerTime" dataset that corresponds to that layer.
    Note: The located dataset will be one level higher in the directory tree.

    >>> name = "zeroDopplerTime"
    >>> _get_path_to_nearest_dataset(in_file, path, name)
    '/science/LSAR/RIFG/swaths/frequencyA/interferogram/zeroDopplerTime'

    If we provide a path to a different raster in the pixel offsets group,
    but give it the same dataset name to find, the function locates the
    "zeroDopplerTime" dataset which (likely) corresponds to that other layer.

    >>> path2 = "/science/LSAR/RIFG/swaths/frequencyA/pixelOffsets/HH/alongTrackOffset"
    >>> _get_path_to_nearest_dataset(in_file, path2, name)
    '/science/LSAR/RIFG/swaths/frequencyA/pixelOffsets/zeroDopplerTime'

    If a dataset is not found, an error is raised.

    >>> name2 = "zeroDopplerTimeFake"
    >>> path3 = _get_path_to_nearest_dataset(in_file, path2, name2)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File ".../product_reader.py", line 98, in _get_path_to_nearest_dataset
        raise nisarqa.DatasetNotFoundError
    nisarqa.utils.utils.DatasetNotFoundError: Dataset not found.
    """
    path_list = starting_path.split("/")

    # Iterate through each parent directory of the dataset (raster image),
    # starting with the directory that the dataset (raster) itself is inside.
    for i in reversed(range(len(path_list))):
        parent = "/".join(path_list[:i])

        path = f"{parent}/{dataset_to_find}"
        if path in h5_file:
            return path
    else:
        raise nisarqa.DatasetNotFoundError(
            f"`{dataset_to_find}` not found in any of the parent directories of"
            f" {starting_path}."
        )


def _get_paths_in_h5(h5_file: h5py.File, name: str) -> list[str]:
    """
    Return the path(s) to the `name` group or dataset in the input file.

    Parameters
    ----------
    h5_file : h5py.File, h5py.Group
        Handle to HDF5 input file or group to be searched.
    name : str
        Base Name of a h5py.Dataset or h5py.Group to be located.

    Returns
    -------
    paths : list of str
        List containing the paths to each dataset or group with the
        name `name` in the input file. For example, if `name` is
        "identification", this function could return
        "science/LSAR/identification"].
        If no occurrences are found, returns an empty list.
    """
    paths = []

    def get_path(path: str) -> None:
        """Find all paths to the group or dataset specified."""
        if path.split("/")[-1] == name:
            paths.append(path)

    h5_file.visit(get_path)

    return paths


@dataclass
class NisarProduct(ABC):
    """
    Base class for NISAR product readers.

    Parameters
    ----------
    filepath : path-like
        Filepath to the input product.
    """

    # Full file path to the input product file.
    filepath: str

    def __post_init__(self):
        # Verify that the input product contained a product spec version.
        self.product_spec_version

        self._check_product_type()
        self._check_root_path()
        self._check_is_geocoded()
        self._check_data_group_path()

    @abstractproperty
    def product_type(self) -> str:
        """Product type (e.g. "RSLC" or "GCOV")."""
        pass

    @abstractproperty
    def is_geocoded(self) -> bool:
        """True if product is geocoded, False if range Doppler grid."""
        pass

    @abstractmethod
    def get_browse_latlonquad(self) -> nisarqa.LatLonQuad:
        """
        Create a LatLonQuad for the corners of the input product.

        Returns
        -------
        llq : LatLonQuad
            A LatLonQuad object containing the four corner coordinates for this
            product's browse image.
        """
        pass

    @cached_property
    def bounding_polygon(self) -> str:
        """Bounding polygon WKT string."""
        id_group = self.identification_path
        with h5py.File(self.filepath) as f:
            wkt = f[id_group]["boundingPolygon"][()]
            return nisarqa.byte_string_to_python_str(wkt)

    @cached_property
    def identification_path(self) -> str:
        """
        Return the path to the identification group in the input file.

        Assumes that the input file contains only one path whose base name
        is "identification". If multiple such paths exist, or if no paths
        have that base name, then an exception is raised.

        Returns
        -------
        id_group : str
            The path to the identification group in the input file.

        Raises
        ------
        nisarqa.InvalidNISARProductError
            If "identification" is not found exactly one time in the input file.
        """
        with h5py.File(self.filepath) as f:
            paths = _get_paths_in_h5(h5_file=f, name="identification")

        if len(paths) != 1:
            raise nisarqa.InvalidNISARProductError(
                "Input product must contain exactly one group named"
                f" 'identification'. Found: {paths}"
            )

        return paths[0]

    @cached_property
    def product_spec_version(self) -> str:
        """
        Return the product specification version.

        Returns
        -------
        spec_version : str
            The value of "productSpecificationVersion" in the "identification"
            group in the input file. If this field is missing (such as with
            older test datasets), "0.0.0" is returned.
        """
        id_group = self.identification_path
        with h5py.File(self.filepath) as f:
            if "productSpecificationVersion" in f[id_group]:
                spec_version = f[id_group]["productSpecificationVersion"][...]
                spec_version = nisarqa.byte_string_to_python_str(spec_version)
            else:
                # spec for very old test datasets.
                # `productSpecificationVersion` metadata was added after this.
                spec_version = "0.0.0"

            # Sanity check for if QA has been tested with this product spec
            if spec_version not in ("0.0.0", "0.9.0"):
                warnings.warn(
                    f"QA for product specification version {spec_version}"
                    " not tested."
                )

            return spec_version

    @cached_property
    def _root_path(self) -> str:
        """
        Get the path to the group which is the root for the primary groups.

        Returns
        -------
        root : str
            Path to the directory where the product data is stored.
                Standard Format: "/science/<band>/<product_type>
                Example:
                    "/science/LSAR/RSLC"

        See Also
        --------
        _data_group_path : Constructs the path to the data group.

        Notes
        -----
        In products up to and including the R3.4 delivery (which used product
        spec 0.9.0), this will be something like "/science/LSAR".
        This structure is set up like e.g.:
            /science/LSAR/RSLC/swaths/frequencyX/...
            /science/LSAR/RSLC/metadata/...
            /science/LSAR/identification
        In subsequent deliveries, that path will likely be truncated. All
        products would follow a consistent structure, something like:
            /data/frequencyX/...
            /metadata/...
            /identification
        In that case, the empty string "" should be returned by this function.
        """
        id_group = self.identification_path
        # The `identification` group is found at e.g.:
        #   "/science/LSAR/identification"
        # Remove identification.
        root = id_group.replace("/identification", "")
        return root

    def _check_root_path(self) -> None:
        """Sanity check that `self._root_path` is valid."""

        with h5py.File(self.filepath) as f:
            # Conditional branch for datasets <=R3.4 (TBD possibly beyond)
            if self._root_path not in f:
                raise ValueError(
                    f"self._root_path determined to be {self._root_path},"
                    " but this is not a valid path in the input file."
                )

    @cached_property
    def band(self) -> str:
        """
        Get the frequency-band ("L" or "S") of the input file.

        Assumption by QA SAS: each input product contains only one band.

        Returns
        -------
        band : str
            The input file's band. (One of "L" or "S").
        """
        log = nisarqa.get_logger()

        id_group = self.identification_path
        band = None

        with h5py.File(self.filepath) as f:
            try:
                band = f[id_group]["radarBand"][...]
            except KeyError:
                log.warn("`radarBand` missing from `identification` group.")
            else:
                band = nisarqa.byte_string_to_python_str(band)

        # TODO - remove the below code once all test data sets are updated to
        # new product spec
        # Attempt to determine the band from the frequency group path
        if band is None:
            # WLOG, get the path to one of the frequency groups
            path = self.get_freq_path(freq=self.freqs[0])
            for b in ("L", "S"):
                if f"{b}SAR" in path:
                    band = b
                    break
            else:
                # Product updated to not include the band in the
                # "/science/LSAR/..." directory structure
                raise ValueError("Cannot determine band from product.")

        return band

    @cached_property
    def freqs(self) -> tuple[str, ...]:
        """
        The available frequencies in the input file.

        Returns
        -------
        found_freqs : tuple of str
            The available frequencies in the input file. Will be a subset of
            ("A", "B").

        Raises
        ------
        nisarqa.InvalidNISARProductError
            If no frequency groups are found.

        Notes
        -----
        "frequency" in this context is different than the typical
        meaning of e.g. "L-band" or "S-band" frequencies.
        In QA SAS, the convention is for e.g. L band or S band
        to be referred to as "bands", while the section of the band used
        for the raster data is referred to as "frequency."
        Most satellites are single band instruments, meaning that "band" and
        "frequency" are one and the same, so this distinction is redundant.
        However, the NISAR mission is capable of dual frequency data collection,
        (meaning that two frequencies of data can be collected within a single
        band), hence the reason for this distinction. For NISAR, these
        are referred to as "Frequency A" and "Freqency B".
        """
        log = nisarqa.get_logger()

        # Get paths to the frequency groups
        found_freqs = []
        for freq in ("A", "B"):
            try:
                path = self.get_freq_path(freq=freq)
            except nisarqa.DatasetNotFoundError:
                log.info(f"Frequency{freq} group not found.")
            else:
                found_freqs.append(freq)
                log.info(f"Found Frequency {freq} group: {path}")

        if not found_freqs:
            errmsg = "Input product does not contain any frequency groups."
            print(errmsg)
            raise nisarqa.InvalidNISARProductError(errmsg)

        return tuple(found_freqs)

    def get_freq_path(self, freq: str) -> str:
        """
        Return the path inside the input file to the specified frequency group.

        Parameters
        ----------
        freq : str
            Must be either "A" or "B".

        Returns
        -------
        path : str
            Path inside the input file to the requested frequency group.

        Raises
        ------
        nisarqa.DatasetNotFoundError
            If frequency not found in the input file.
        """
        if freq not in ("A", "B"):
            raise ValueError(f"{freq=}, must be either 'A' or 'B'")

        # We cannot use functools.lru_cache() on a instance method due to
        # the `self` parameter, so use an inner function to cache the results.
        @lru_cache
        def _freq_path(freq):
            log = nisarqa.get_logger()
            path = self._data_group_path + f"/frequency{freq}"
            with h5py.File(self.filepath) as f:
                if path in f:
                    return path
                else:
                    errmsg = (
                        f"Input file does not contain frequency {freq} group at"
                        f" path: {path}"
                    )
                    log.info(errmsg)
                    raise nisarqa.DatasetNotFoundError(errmsg)

        return _freq_path(freq)

    @cached_property
    def science_freq(self) -> str:
        """
        The science frequency (primary frequency) of the input product.

        Returns
        -------
        freq : str
            The science frequency for the input product. One of: "A" or "B".
        """
        return "A" if "A" in self.freqs else "B"

    def _check_product_type(self) -> None:
        """
        Sanity check for `self.product_type`.

        Ensures that `self.product_type` returns a value
        that matches the `identification > productType` dataset in the
        input file.
        """
        id_group = self.identification_path

        with h5py.File(self.filepath) as f:
            in_file_prod_type = f[id_group]["productType"][...]
            in_file_prod_type = nisarqa.byte_string_to_python_str(in_file_prod_type)

        if self.product_type != in_file_prod_type:
            raise ValueError(
                f"QA requested for {self.product_type}, but input file's"
                f" `productType` field is {in_file_prod_type}."
            )

    def _check_is_geocoded(self) -> None:
        """Sanity check for `self.is_geocoded`."""
        log = nisarqa.get_logger()

        id_group = self.identification_path

        with h5py.File(self.filepath) as f:
            if "isGeocoded" in f[id_group]:
                # Check that `isGeocoded` is set correctly, i.e. that it is
                # False in range Doppler products, and True in Geocoded products
                ds_handle = f[id_group]["isGeocoded"]
                if self.is_geocoded is not bool(ds_handle[...]):
                    warnings.warn(
                        "WARNING `/identification/isGeocoded` field has value"
                        f" {ds_handle[...]}, which is inconsistent with"
                        f" product type of {self.product_type}."
                    )
                # Check that the value has the correct dtype and formatting
                nisarqa.verify_isce3_boolean(ds_handle)
            else:
                # The `isGeocoded` field is not necessary for successful
                # completion QA SAS: whether a product is geocoded
                # or not can be determined by the product type (e.g. RSLC vs.
                # GSLC). So let's simply raise a warning and let QA continue.
                # However, it's part of QA's job to check these fields for
                # product robustness and to warn the user of faulty products.
                log.warn("Product missing `identification > isGeocoded` field")

    def _check_data_group_path(self) -> None:
        """Sanity check to ensure the grid path exists in the input file."""
        grid_path = self._data_group_path
        with h5py.File(self.filepath) as f:
            if grid_path not in f:
                errmsg = f"Input file is missing the path: {grid_path}"
                print(errmsg)
                raise nisarqa.DatasetNotFoundError(errmsg)

    @abstractmethod
    def _data_group_path(self) -> str:
        """
        Return the path to the group containing the primary datasets.

        For range Doppler products, this should be e.g.:
            'science/LSAR/RIFG/swaths'
        For geocoded products, this should be e.g.:
            'science/LSAR/RIFG/grids'

        Returns
        -------
        root : str
            Path to the directory where the product data is stored.
                Standard Format:
                    "science/<band>/<product_type>/<'swaths' OR 'grids'>"
                Example:
                    "science/LSAR/RSLC/swaths"

        See Also
        --------
        _root_path : Constructs the path to the primary root group.

        Notes
        -----
        A common implementation for this would be e.g.:
            return "/".join([self._root_path, self.product_type, "swaths"])

        In products up to and including the R3.4 delivery (which used product
        spec 0.9.0), the L1/L2 product structure is set up like e.g.:
            /science/LSAR/RSLC/swaths/frequencyX/...
            /science/LSAR/RSLC/metadata/...
            /science/LSAR/identification
        In which case "/science/LSAR/RSLC/swaths" should be returned.

        In subsequent deliveries, that path will likely be truncated. All
        products would follow a consistent structure, something like:
            /data/frequencyX/...
            /metadata/...
            /identification
        In that case, the string "data" should be returned by this function.
        """
        pass

    def _metadata_group_path(self) -> str:
        """
        Get the path to the metadata group.

        Returns
        -------
        root : str
            Path to the metadata directory.
                Standard Format: "/science/<band>/<product_type>/metadata
                Example:
                    "/science/LSAR/RSLC/metadata"

        Notes
        -----
        In products up to and including the R3.4 delivery (which used product
        spec 0.9.0), this input product structure is set up like e.g.:
            /science/LSAR/RSLC/swaths/frequencyX/...
            /science/LSAR/RSLC/metadata/...
            /science/LSAR/identification
        In subsequent deliveries, that path will likely be truncated. All
        products would follow a consistent structure, something like:
            /data/frequencyX/...
            /metadata/...
            /identification
        When that change is implemented, this function will need to be updated
        to accommodate the new spec.
        """
        return "/".join([self._root_path, self.product_type, "metadata"])

    @abstractmethod
    def _get_raster_from_path(
        self, h5_file: h5py.File, raster_path: str
    ) -> nisarqa.RadarRaster | nisarqa.GeoRaster:
        """
        Generate a *Raster for the raster at `raster_path`.

        Parameters
        ----------
        h5_file : h5py.File
            Open file handle for the input file containing the raster.
        raster_path : str
            Full path in `h5_file` to the desired raster dataset
            Examples:
                "/science/LSAR/RSLC/swaths/frequencyA/HH"
                "/science/LSAR/RIFG/swaths/frequencyA/interferogram/HH/wrappedInterferogram"

        Returns
        -------
        raster : nisarqa.RadarRaster | nisarqa.GeoRaster
            *Raster of the given dataset.

        Raises
        ------
        DatasetNotFoundError
            If the file does not contain a raster dataset at `raster_path`,
            or if any of the corresponding metadata for that raster are
            not located.
        """
        pass


@dataclass
class NisarRadarProduct(NisarProduct):
    @property
    def is_geocoded(self) -> bool:
        return False

    @cached_property
    def _data_group_path(self) -> str:
        return "/".join([self._root_path, self.product_type, "swaths"])

    def get_browse_latlonquad(self) -> nisarqa.LatLonQuad:
        # Shapely boundary coords is a tuple of coordinate lists of form ([x...], [y...])
        coords = shapely.from_wkt(self.bounding_polygon).boundary.coords
        # Rezip the coordinates to a list of (x, y) tuples,
        # and convert to radians for the internal LonLat class
        coords = [
            nisarqa.LonLat(np.deg2rad(c[0]), np.deg2rad(c[1]))
            for c in zip(*coords.xy)
        ]

        # Workaround for bug in ISCE3 generated products. We expect 41 points
        # (10 along each side + endpoint same as start point), but there
        # are 42 points (41 points in expected order + duplicated endpoint).
        # So if the endpoint is duplicated, we drop it here to handle this bug.
        # (See https://github-fn.jpl.nasa.gov/isce-3/isce/issues/1486)
        if coords[-1] == coords[-2]:
            coords = coords[:-1]

        # Drop last (same as first) coordinate
        if coords[-1] != coords[0]:
            msg = (
                "Input product's boundingPolygon is not closed"
                " (endpoint does not match start point)"
            )
            warnings.warn(msg, RuntimeWarning)
        coords = coords[:-1]

        if len(coords) < 4:
            raise ValueError("Not enough coordinates for bounding polygon")
        if len(coords) % 4 != 0:
            raise ValueError("Bounding polygon requires evenly spaced corners")

        # Corners are assumed to start at the 0th index and be evenly spaced
        clockwise_corners = [coords[len(coords) // 4 * i] for i in range(4)]

        # The boundingPolygon is specified in clockwise order, starting at the
        # upper-left of the image. Here we reorder them for the LatLonQuad,
        # constructor, which expects them in left-to-right top-to-bottom order.
        geo_corners = (
            clockwise_corners[0],
            clockwise_corners[1],
            clockwise_corners[3],
            clockwise_corners[2],
        )
        return nisarqa.LatLonQuad(*geo_corners)

    @abstractmethod
    def _get_raster_name(self, raster_path: str) -> str:
        """
        Get the name for the raster, e.g. 'RSLC_LSAR_A_HH'.

        Parameters
        ----------
        raster_path : str
            Full path in `h5_file` to the desired raster dataset
            Examples:
                "/science/LSAR/GSLC/grids/frequencyA/HH"
                "/science/LSAR/GUNW/grids/frequencyA/interferogram/HH/unwrappedPhase"

        Returns
        -------
        name : str
            The human-understandable name that is derived from the dataset.
            Examples:
                "GSLC_L_A_HH"
                "GUNW_L_A_HH_unwrappedPhase"
        """
        pass

    def _get_raster_from_path(
        self, h5_file: h5py.File, raster_path: str
    ) -> nisarqa.RadarRaster:
        """
        Generate a RadarRaster for the raster at `raster_path`.

        NISAR product type must be one of: 'RSLC', 'SLC', 'RIFG', 'RUNW', 'ROFF'
        If the product type is 'RSLC' or 'SLC', then the image dataset
        will be stored as a ComplexFloat16Decoder instance; this will allow
        significantly faster access to the data.

        Parameters
        ----------
        h5_file : h5py.File
            Open file handle for the input file containing the raster.
        raster_path : str
            Full path in `h5_file` to the desired raster dataset.
            Examples:
                "/science/LSAR/RSLC/swaths/frequencyA/HH"
                "/science/LSAR/RIFG/swaths/frequencyA/interferogram/HH/wrappedInterferogram"

        Returns
        -------
        raster : nisarqa.RadarRaster
            RadarRaster of the given dataset.

        Raises
        ------
        DatasetNotFoundError
            If the file does not contain a raster dataset at `raster_path`,
            or if any of the corresponding metadata for that raster are
            not located.

        Notes
        -----
        The `name` attribute will be populated per `self.get_name()`.
        As of 6/22/2023, these additional datasets which correspond to
        `raster_path` will be parsed via
        `_get_path_to_nearest_dataset(..., raster_path, <dataset name>) :
            sceneCenterAlongTrackSpacing
            sceneCenterGroundRangeSpacing
            slantRange
            zeroDopplerTime
        """
        if raster_path not in h5_file:
            errmsg = f"Input file does not contain raster {raster_path}"
            print(errmsg)
            raise nisarqa.DatasetNotFoundError(errmsg)

        # Get dataset object and check for correct dtype
        dataset = self._get_dataset_handle(h5_file, raster_path)

        # From the xml Product Spec, sceneCenterAlongTrackSpacing is the
        # 'Nominal along track spacing in meters between consecutive lines
        # near mid swath of the RSLC image.'
        path = _get_path_to_nearest_dataset(
            h5_file=h5_file,
            starting_path=raster_path,
            dataset_to_find="sceneCenterAlongTrackSpacing",
        )
        ground_az_spacing = h5_file[path][...]

        # Get Azimuth (y-axis) tick range + label
        # path in h5 file: /science/LSAR/RSLC/swaths/zeroDopplerTime
        # For NISAR, radar-domain grids are referenced by the center of the
        # pixel, so +/- half the distance of the pixel's side to capture
        # the entire range.
        path = _get_path_to_nearest_dataset(
            h5_file=h5_file,
            starting_path=raster_path,
            dataset_to_find="zeroDopplerTime",
        )
        az_start = float(h5_file[path][0]) - 0.5 * ground_az_spacing
        az_stop = float(h5_file[path][-1]) + 0.5 * ground_az_spacing

        # Use zeroDopplerTime's units attribute to read the epoch
        # output of the next line will have the format:
        #       'seconds since YYYY-MM-DD HH:MM:SS'
        sec_since_epoch = h5_file[path].attrs["units"].decode("utf-8")
        epoch = sec_since_epoch.replace("seconds since ", "").strip()

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

        # From the xml Product Spec, sceneCenterGroundRangeSpacing is the
        # 'Nominal ground range spacing in meters between consecutive pixels
        # near mid swath of the RSLC image.'
        path = _get_path_to_nearest_dataset(
            h5_file=h5_file,
            starting_path=raster_path,
            dataset_to_find="sceneCenterGroundRangeSpacing",
        )
        ground_range_spacing = h5_file[path][...]

        # Range in meters (units are specified as meters in the product spec)
        # For NISAR, radar-domain grids are referenced by the center of the
        # pixel, so +/- half the distance of the pixel's side to capture
        # the entire range.
        path = _get_path_to_nearest_dataset(
            h5_file=h5_file,
            starting_path=raster_path,
            dataset_to_find="slantRange",
        )
        rng_start = float(h5_file[path][0]) - 0.5 * ground_range_spacing
        rng_stop = float(h5_file[path][-1]) + 0.5 * ground_range_spacing

        # Construct Name
        name = self._get_raster_name(raster_path)

        return nisarqa.RadarRaster(
            data=dataset,
            name=name,
            band=self.band,
            freq="A" if "frequencyA" in raster_path else "B",
            ground_az_spacing=ground_az_spacing,
            az_start=az_start,
            az_stop=az_stop,
            ground_range_spacing=ground_range_spacing,
            rng_start=rng_start,
            rng_stop=rng_stop,
            epoch=epoch,
        )

    @abstractmethod
    def _get_dataset_handle(
        self, h5_file: h5py.File, raster_path: str
    ) -> h5py.Dataset:
        """
        Return a handle to the requested dataset.

        When implemented, it is recommended that validation checks on the
        dataset's dtype, etc. are performed before returning the handle.
        This function is a useful abstraction in case we need to e.g. use
        ComplexFloat16DataDecoder to access RSLC data.

        Parameters
        ----------
        h5_file : h5py.File
            File handle for the input file.
        raster_path : str
            Path in the input file to the desired dataset.
            Examples:
                "/science/LSAR/RSLC/swaths/frequencyA/HH"
                "/science/LSAR/RIFG/swaths/frequencyA/interferogram/HH/wrappedInterferogram"

        Returns
        -------
        dataset : h5py.Dataset
            Handle to the requested dataset.
        """
        pass


@dataclass
class NisarGeoProduct(NisarProduct):
    @property
    def is_geocoded(self) -> bool:
        return True

    def get_browse_latlonquad(self) -> nisarqa.LatLonQuad:
        epsg = self.epsg
        proj = isce3.core.make_projection(epsg)

        geo_corners = ()
        for y in self.browse_y_range:
            for x in self.browse_x_range:
                # Use a dummy height value in computing the inverse projection.
                # isce3 projections are always 2-D transformations -- the height
                # has no effect on lon/lat
                lon, lat, _ = proj.inverse([x, y, 0])
                geo_corners += (nisarqa.LonLat(lon, lat),)
        return nisarqa.LatLonQuad(*geo_corners)

    @cached_property
    def _data_group_path(self) -> str:
        return "/".join([self._root_path, self.product_type, "grids"])

    @cached_property
    def epsg(self) -> str:
        """EPSG code for input product."""
        with h5py.File(self.filepath) as f:
            # EPSG code is consistent for both frequencies. WLOG pick the
            # science frequency.
            freq_path = self.get_freq_path(freq=self.science_freq)

            # Get the sub path to an occurrence of a `projection` dataset
            # for the chosen frequency. (Again, they're all the same.)
            proj_path = _get_paths_in_h5(
                h5_file=f[freq_path], name="projection"
            )
            try:
                proj_path = "/".join([freq_path, proj_path[0]])
            except IndexError as exc:
                raise nisarqa.DatasetNotFoundError(
                    "no projection path found"
                ) from exc

            return f[proj_path][...]

    @abstractproperty
    def browse_x_range(self) -> tuple(float, float):
        """
        Get the x range coordinates for the browse image.

        Returns
        -------
        x_range : tuple of float
            The range of the x coordinates for the browse image.
            Format:
                (<x_start>, <x_stop>)
        """
        pass

    @abstractproperty
    def browse_y_range(self) -> tuple(float, float):
        """
        Get the y range coordinates for the browse image.

        Returns
        -------
        y_range : tuple of float
            The range of the y coordinates for the browse image.
            Format:
                (<y_start>, <y_stop>)
        """
        pass

    @abstractmethod
    def _get_raster_name(self, raster_path: str) -> str:
        """
        Return a name for the raster, e.g. 'RSLC_LSAR_A_HH'.

        Parameters
        ----------
        raster_path : str
            Full path in `h5_file` to the desired raster dataset.
            Examples:
                "/science/LSAR/GSLC/grids/frequencyA/HH"
                "/science/LSAR/GUNW/grids/frequencyA/interferogram/HH/unwrappedPhase"

        Returns
        -------
        name : str
            The human-understandable name that is derived from the dataset.
            Examples: "GSLC_L_A_HH" or "GUNW_L_A_HH_unwrappedPhase".
        """
        pass

    def _get_raster_from_path(
        self, h5_file: h5py.File, raster_path: str
    ) -> nisarqa.GeoRaster:
        """
        Get the GeoRaster for the raster at `raster_path`.

        NISAR product type must be one of: 'GSLC', 'GCOV', 'GUNW', 'GOFF'.

        Parameters
        ----------
        h5_file : h5py.File
            Open file handle for the input file containing the raster.
        raster_path : str
            Full path in `h5_file` to the desired raster dataset.
            Examples:
                "/science/LSAR/GSLC/grids/frequencyA/HH"
                "/science/LSAR/GUNW/grids/frequencyA/interferogram/HH/unwrappedPhase"

        Returns
        -------
        raster : nisarqa.GeoRaster
            GeoRaster of the given dataset.

        Raises
        ------
        DatasetNotFoundError
            If the file does not contain a raster dataset at `raster_path`,
            or if any of the corresponding metadata for that raster are
            not located.

        Notes
        -----
        The `name` attribute will be populated per `self.get_name()`.
        As of Aug 7, 2023, these additional datasets which correspond to
        `raster_path` will be parsed via
        `_get_path_to_nearest_dataset(..., raster_path, <dataset name>) :
            xCoordinateSpacing
            xCoordinates
            yCoordinateSpacing
            yCoordinates
        """
        if raster_path not in h5_file:
            errmsg = f"Input file does not contain raster {raster_path}"
            print(errmsg)
            raise nisarqa.DatasetNotFoundError(errmsg)

        # Get dataset object and check for correct dtype
        dataset = self._get_dataset_handle(h5_file, raster_path)

        # From the xml Product Spec, xCoordinateSpacing is the
        # 'Nominal spacing in meters between consecutive pixels'
        path = _get_path_to_nearest_dataset(
            h5_file=h5_file,
            starting_path=raster_path,
            dataset_to_find="xCoordinateSpacing",
        )
        x_spacing = float(h5_file[path][...])

        # X in meters (units are specified as meters in the product spec)
        # For NISAR, geocoded grids are referenced by the upper-left corner
        # of the pixel to match GDAL conventions. So add the distance of
        # the pixel's side to far right side to get the actual stop value.
        path = _get_path_to_nearest_dataset(
            h5_file=h5_file,
            starting_path=raster_path,
            dataset_to_find="xCoordinates",
        )
        x_start = float(h5_file[path][0])
        x_stop = float(h5_file[path][-1]) + x_spacing

        # From the xml Product Spec, yCoordinateSpacing is the
        # 'Nominal spacing in meters between consecutive lines'
        path = _get_path_to_nearest_dataset(
            h5_file=h5_file,
            starting_path=raster_path,
            dataset_to_find="yCoordinateSpacing",
        )
        y_spacing = float(h5_file[path][...])

        # Y in meters (units are specified as meters in the product spec)
        # For NISAR, geocoded grids are referenced by the upper-left corner
        # of the pixel to match GDAL conventions. So add the distance of
        # the pixel's side to bottom to get the actual stop value.
        path = _get_path_to_nearest_dataset(
            h5_file=h5_file,
            starting_path=raster_path,
            dataset_to_find="yCoordinates",
        )
        y_start = float(h5_file[path][0])
        y_stop = float(h5_file[path][-1]) + y_spacing

        # Construct Name
        name = self._get_raster_name(raster_path)

        return nisarqa.GeoRaster(
            data=dataset,
            name=name,
            band=self.band,
            freq="A" if ("frequencyA" in raster_path) else "B",
            x_spacing=x_spacing,
            x_start=x_start,
            x_stop=x_stop,
            y_spacing=y_spacing,
            y_start=y_start,
            y_stop=y_stop,
        )

    @abstractmethod
    def _get_dataset_handle(
        self, h5_file: h5py.File, raster_path: str
    ) -> h5py.Dataset:
        """
        Return a handle to the requested dataset.

        It is recommended that the implementation perform validation
        checks on the dataset's dtype, etc. before returning the handle.
        For example, RSLC might need to use ComplexFloat16Decoder.

        Parameters
        ----------
        h5_file : h5py.File
            File handle for the input file.
        raster_path : str
            Path in the input file to the desired dataset.
            Examples:
                "/science/LSAR/RSLC/grids/frequencyA/HH"
                "/science/LSAR/RIFG/grids/frequencyA/wrappedInterferogram/HH/wrappedInterferogram"

        Returns
        -------
        dataset : h5py.Dataset
            Handle to the requested dataset.
        """
        pass


@dataclass
class NonInsarProduct(NisarProduct):
    """Common functionality for RSLC, GLSC, and GCOV products."""

    @abstractmethod
    def get_layers_for_browse(self) -> dict[str, list[str]]:
        """
        Determine which polarization layers to use to generate the browse image.

        Returns
        -------
        layers_for_browse : dict
            Dict of the polarizations layers to use. Its structure is:
                layers_for_browse[<freq>] : List of the polarization(s)
                                            e.g. ['HH','HV'] for SLCs,
                                            or ["HHHH", "HVHV"] for GCOV
        """
        pass

    @staticmethod
    @abstractmethod
    def save_browse(
        pol_imgs: dict[str, np.ndarray], filepath: str | os.PathLike
    ) -> None:
        """
        Save given polarization images to a RGB or Grayscale PNG.

        Dimensions of the output PNG (in pixels) will be the same as the
        dimensions of the input polarization image array(s). (No scaling will
        occur.) Non-finite values will be made transparent.

        Parameters
        ----------
        pol_imgs : dict of numpy.ndarray
            Dictionary of 2D array(s) that will be mapped to specific color
            channel(s) for the output browse PNG.
            If there are multiple image arrays, they must have identical shape.
            Format of dictionary:
                pol_imgs[<polarization>] : <2D numpy.ndarray image>
            Example:
                pol_imgs['HHHH'] : <2D numpy.ndarray image>
                pol_imgs['VVVV'] : <2D numpy.ndarray image>
        filepath : path-like
            Full filepath for where to save the browse image PNG.

        Notes
        -----
        Provided image array(s) must previously be image-corrected. This
        function will take the image array(s) as-is and will not apply
        additional image correction processing to them. This function directly
        combines the image(s) into a single browse image.
        If there are multiple input images, they must be thoughtfully prepared
        and standardized relative to each other prior to use by this function.
        For example, trying to combine a Freq A 20 MHz image
        and a Freq B 5 MHz image into the same output browse image might not go
        well, unless the image arrays were properly prepared and standardized
        in advance.
        """
        pass

    @contextmanager
    def get_raster(
        self, freq: str, pol: str
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Context Manager for a RadarRaster or GeoRaster for the specified raster.

        Parameters
        ----------
        freq : str
            One of the frequencies returned by `self.freqs()`.
        pol : str
            The desired polarization.
              For RSLC and GSLC, use e.g. "HH", "HV",
              For GCOV, use e.g. "HHVV"

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
            Warning: the input NISAR file cannot be opened by ISCE3 until
            this context manager is exited.

        Raises
        ------
        DatasetNotFoundError
            If the file does not contain the requested raster dataset,
            or if any of the corresponding metadata for that raster are
            not located.
        """
        if freq not in self.freqs:
            raise ValueError(f"{freq=}, must be one of: {self.freqs=}")

        poss_pols = nisarqa.get_possible_pols(
            product_type=self.product_type.lower()
        )
        if pol not in poss_pols:
            raise ValueError(f"{pol=}, must be one of: {poss_pols}.")

        path = self._layers[freq][pol]

        with h5py.File(self.filepath, "r") as in_file:
            if path not in in_file:
                errmsg = f"Input file does not contain raster {path}"
                print(errmsg)
                raise nisarqa.DatasetNotFoundError(errmsg)
            yield self._get_raster_from_path(h5_file=in_file, raster_path=path)

    def _get_raster_name(self, raster_path: str) -> str:
        """
        Return a name for the raster, e.g. 'RSLC_LSAR_A_HH'.

        Parameters
        ----------
        raster_path : str
            Full path in `h5_file` to the desired raster dataset.
            Example:
                "/science/LSAR/GSLC/grids/frequencyA/HH"

        Returns
        -------
        name : str
            The human-understandable name that is derived from the dataset.
            Example:
                "GSLC_L_A_HH"
        """
        freq = "A" if ("frequencyA" in raster_path) else "B"
        pol = raster_path.split("/")[-1]
        name = f"{self.product_type}_{self.band}_{freq}_{pol}"

        return name

    @cached_property
    def _layers(self) -> dict[str, dict[str, str]]:
        """
        Locate available bands, frequencies, and polarizations in the product.

        Returns
        -------
        layers : nested dict of str
            Nested dict of paths to input file raster datasets.
            Format: layers[<freq>][<pol>] -> str
            Example:
                layers["A"]["HH"] -> "/science/LSAR/RSLC/swaths/frequencyA/HH"
        """
        # Discover images in input file and populate the `pols` dictionary
        with h5py.File(self.filepath) as h5_file:
            layers = {}
            for freq in self.freqs:
                path = self.get_freq_path(freq=freq)
                layers[freq] = {}

                for pol in nisarqa.get_possible_pols(self.product_type.lower()):
                    raster_path = f"{path}/{pol}"
                    if raster_path in h5_file:
                        layers[freq][pol] = raster_path

        # Sanity Check - if a band/freq does not have any polarizations,
        # this is a validation error. This check should be handled during
        # the validation process before this function was called,
        # not the quality process, so raise an error.
        # In the future, this step might be moved earlier in the
        # processing, and changed to be handled via: 'log the error
        # and remove the band from the dictionary'
        for freq in layers:
            # Empty dictionaries evaluate to False in Python
            if not layers[freq]:
                raise ValueError(
                    "Provided input file does not have any polarizations"
                    f" included under band {self.band}, frequency {freq}."
                )

        return layers

    def get_pols(self, freq: str) -> tuple[str, ...]:
        """
        Get a tuple of the available polarizations for frequency `freq`.

        Parameters
        ----------
        freq : str
            One of the frequencies in `self.freqs`.

        Returns
        -------
        pols : tuple of str
            An tuple of the available polarizations for the requested
            frequency.
                Example for RSLC or GSLC: ("HH", "VV")
                Example for GCOV: ("HHHH", "HVHH", "VVVV")
        """
        if freq not in self.freqs:
            raise ValueError(
                f"Requested frequency {freq}, but product only contains"
                f" frequencies {self.freqs}."
            )

        layers = self._layers
        return tuple(layers[freq].keys())


@dataclass
class SLC(NonInsarProduct):
    def get_layers_for_browse(self) -> dict[str, list[str]]:
        """
        Assign polarizations to grayscale or RGBA channels for the Browse Image.

        See `Notes` for details on  possible NISAR modes and assigned channels
        for LSAR band.
        SSAR is currently only minimally supported, so only a grayscale image
        will be created. Prioritization order to select the freq/pol to use:
            For frequency: Freq A then Freq B.
            For polarization: 'HH', then 'VV', then first polarization found.

        Returns
        -------
        layers_for_browse : dict
            A dictionary containing the frequency+polarization combinations
            that will be used to generate the browse image. Its structure is:
            layers_for_browse['A']     : list of str, optional
                                            List of the Freq A polarization(s)
                                            required to create the browse image.
                                            A subset of:
                                            ['HH','HV','VV','RH','RV','LH','LV']
            layers_for_browse['B']     : list of str, optional
                                            List of the Freq B polarizations
                                            required to create the browse image.
                                            A subset of ['HH','VV']

        Notes
        -----
        Possible modes for L-Band, as of Feb 2023:
            Single Pol      SP HH:      20+5, 40+5, 77
            Single Pol      SP VV:      5, 40
            Dual Pol        DP HH/HV:   77, 40+5, 20+5
            Dual Pol        DP VV/VH:   5, 77, 20+5, 40+5
            Quasi Quad Pol  QQ:         20+20, 20+5, 40+5, 5+5
            Quad Pol        QP:         20+5, 40+5
            Quasi Dual Pol  QD HH/VV:   5+5
            Compact Pol     CP RH/RV:   20+20           # an experimental mode
        Single Pol (SP) Assignment:
            - Freq A CoPol
            else:
            - Freq B CoPol
        DP and QQ Assignment:
            - Freq A: Red=HH, Green=HV, Blue=HH
        QP Assignment:
            - Freq A: Red=HH, Green=HV, Blue=VV
        QD Assignment:
            - Freq A: Red=HH, Blue=HH
            - Freq B: Green=VV
        CP Assignment:
            - Freq A: Grayscale of one pol image, with
                    Prioritization order: ['RH','RV','LH','LV']
        """
        layers_for_browse = {}

        # Get the frequency sub-band containing science mode data.
        freq = self.science_freq
        science_pols = self.get_pols(freq=freq)

        # SSAR is not fully supported by QA, so just make a simple grayscale
        if self.band == "S":
            # Prioritize Co-Pol
            if "HH" in science_pols:
                layers_for_browse[freq] = ["HH"]
            elif "VV" in science_pols:
                layers_for_browse[freq] = ["VV"]
            else:
                # Take the first available Cross-Pol
                layers_for_browse[freq] = self.get_raster(
                    freq=freq, pol=science_pols[0]
                )

            return layers_for_browse

        # The input file contains LSAR data. Will need to make
        # grayscale/RGB channel assignments

        n_pols = len(science_pols)

        if freq == "B":
            # Only Freq B has data; this only occurs in Single Pol case.
            if n_pols > 1:
                raise ValueError(
                    "When only Freq B is present, then only "
                    f"single-pol mode supported. Freq{freq}: {science_pols}"
                )

            layers_for_browse["B"] = science_pols

        else:  # freq A exists
            if science_pols[0].startswith("R") or science_pols[0].startswith(
                "L"
            ):
                # Compact Pol. This is not a planned mode for LSAR,
                # and there is no test data, so simply make a grayscale image.

                # Per the Prioritization Order, use first available polarization
                for pol in ["RH", "RV", "LH", "LV"]:
                    if pol in science_pols:
                        layers_for_browse["A"] = [pol]
                        break

                assert len(layers_for_browse["A"]) == 1

            elif n_pols == 1:  # Horizontal/Vertical transmit
                if "B" in self.freqs:
                    b_pols = self.get_pols(freq="B")
                    # Freq A has one pol image, and Freq B exists.
                    if set(science_pols) == set(b_pols):
                        # A's polarization image is identical to B's pol image,
                        # which means that this is a single-pol observation mode
                        # where both frequency bands were active
                        layers_for_browse["A"] = science_pols

                    elif len(b_pols) == 1:
                        # Quasi Dual Pol -- Freq A has HH, Freq B has VV
                        assert "HH" in self.get_pols(freq="A")
                        assert "VV" in b_pols

                        layers_for_browse["A"] = science_pols
                        layers_for_browse["B"] = ["VV"]

                    else:
                        # There is/are polarization image(s) for both A and B.
                        # But, they are not representative of any of the current
                        # observation modes for NISAR.
                        raise ValueError(
                            f"Freq A contains 1 polarization {science_pols},"
                            f" but Freq B contains polarization(s) {b_pols}."
                            " This setup does not match any known NISAR"
                            " observation mode."
                        )
                else:
                    # Single Pol
                    layers_for_browse["A"] = science_pols

            elif n_pols in (2, 4):  # Horizontal/Vertical transmit
                # dual-pol, quad-pol, or Quasi-Quad pol

                # HH has priority over VV
                if "HH" in science_pols and "HV" in science_pols:
                    layers_for_browse["A"] = ["HH", "HV"]
                    if n_pols == 4:
                        # quad pol
                        layers_for_browse["A"].append("VV")

                elif "VV" in science_pols and "VH" in science_pols:
                    # If there is only 'VV', then this granule must be dual-pol
                    assert n_pols == 2
                    layers_for_browse["A"] = ["VV", "VH"]

                else:
                    raise ValueError(
                        "For dual-pol, quad-pol, and quasi-quad modes, "
                        "the input product must contain at least one "
                        "of HH+HV and/or VV+VH channels. Instead got: "
                        f"{science_pols}"
                    )
            else:
                raise ValueError(
                    f"Input product's frequnecy {freq} contains {n_pols} "
                    "polarization images, but only 1, 2, or 4 "
                    "are supported."
                )

        # Sanity Check
        if ("A" not in layers_for_browse) and ("B" not in layers_for_browse):
            raise ValueError(
                "Current Mode (configuration) of the NISAR input file"
                " not supported for browse image."
            )

        return layers_for_browse

    @staticmethod
    def save_browse(
        pol_imgs: dict[str, np.ndarray], filepath: str | os.PathLike
    ) -> None:
        """
        Save images in `pol_imgs` to a RGB or Grayscale PNG with transparency.

        Dimensions of the output PNG (in pixels) will be the same as the
        dimensions of the input polarization image array(s). (No scaling will
        occur.) Non-finite values will be made transparent.
        Color Channels will be assigned per the following pseudocode:
            If pol_imgs.keys() contains only one image, then:
                grayscale = <that image>
            If pol_imgs.keys() is ['HH','HV','VV'], then:
                red = 'HH'
                green = 'HV'
                blue = 'VV'
            If pol_imgs.keys() is ['HH','HV'], then:
                red = 'HH'
                green = 'HV'
                blue = 'HH'
            If pol_imgs.keys() is ['HH','VV'], then:
                red = 'HH'
                green = 'VV'
                blue = 'HH'
            If pol_imgs.keys() is ['VV','VH'], then:
                red = 'VV'
                green = 'VH'
                blue = 'VV'
            Otherwise, one image in `pol_imgs` will be output as grayscale.

        Parameters
        ----------
        pol_imgs : dict of numpy.ndarray
            Dictionary of 2D array(s) that will be mapped to specific color
            channel(s) for the output browse PNG.
            If there are multiple image arrays, they must have identical shape.
            Format of dictionary:
                pol_imgs[<polarization>] : <2D numpy.ndarray image>, where
                    <polarization> must be a subset of: 'HH', 'HV', 'VV', 'VH',
                                                        'RH', 'RV', 'LV', 'LH',
            Example:
                pol_imgs['HH'] : <2D numpy.ndarray image>
                pol_imgs['VV'] : <2D numpy.ndarray image>
        filepath : path-like
            Full filepath for where to save the browse image PNG.

        Notes
        -----
        Provided image array(s) must previously be image-corrected. This
        function will take  image array(s) as-is and will not apply additional
        image correction processing to them. This function directly combines
        the image(s) into a single browse image.
        If there are multiple input images, they must be thoughtfully prepared
        and standardized relative to each other prior to use by this function.
        For example, trying to combine a Freq A 20 MHz image
        and a Freq B 5 MHz image into the same output browse image might not go
        well, unless the image arrays were properly prepared and standardized
        in advance.
        """
        # WLOG, get the shape of the image arrays
        # They should all be the same shape; the check for this is below.
        arbitrary_img = next(iter(pol_imgs.values()))
        img_2D_shape = np.shape(arbitrary_img)
        for img in pol_imgs.values():
            # Input validation check
            if np.shape(img) != img_2D_shape:
                raise ValueError(
                    "All image arrays in `pol_imgs` must have the same shape."
                )

        # Assign color channels
        set_of_pol_imgs = set(pol_imgs)

        if set_of_pol_imgs == {"HH", "HV", "VV"}:
            # Quad Pol
            red = pol_imgs["HH"]
            green = pol_imgs["HV"]
            blue = pol_imgs["VV"]
        elif set_of_pol_imgs == {"HH", "HV"}:
            # dual pol horizontal transmit, or quasi-quad
            red = pol_imgs["HH"]
            green = pol_imgs["HV"]
            blue = pol_imgs["HH"]
        elif set_of_pol_imgs == {"HH", "VV"}:
            # quasi-dual mode
            red = pol_imgs["HH"]
            green = pol_imgs["VV"]
            blue = pol_imgs["HH"]
        elif set_of_pol_imgs == {"VV", "VH"}:
            # dual-pol only, vertical transmit
            red = pol_imgs["VV"]
            green = pol_imgs["VH"]
            blue = pol_imgs["VV"]
        else:
            # If we get into this "else" statement, then
            # either there is only one image provided (e.g. single pol),
            # or the images provided are not one of the expected cases.
            # Either way, WLOG plot one of the image(s) in `pol_imgs`.
            gray_img = pol_imgs.popitem()[1]
            nisarqa.rslc.plot_to_grayscale_png(
                img_arr=gray_img, filepath=filepath
            )

            # This `else` is a catch-all clause. Return early, so that
            # we do not try to plot to RGB
            return

        nisarqa.rslc.plot_to_rgb_png(
            red=red, green=green, blue=blue, filepath=filepath
        )


@dataclass
class NonInsarGeoProduct(NonInsarProduct, NisarGeoProduct):
    @cached_property
    def browse_x_range(self) -> tuple(float, float):
        # All rasters used for the browse should have the same grid specs
        # So, WLOG parse the specs from the first one of them.
        layers = self.get_layers_for_browse()
        freq = next(iter(layers.keys()))
        pol = layers[freq][0]

        with self.get_raster(freq=freq, pol=pol) as img:
            x_start = img.x_start
            x_stop = img.x_stop

        return (x_start, x_stop)

    @cached_property
    def browse_y_range(self) -> tuple(float, float):
        # All rasters used for the browse should have the same grid specs
        # So, WLOG parse the specs from the first one of them.
        layers = self.get_layers_for_browse()
        freq = next(iter(layers.keys()))
        pol = layers[freq][0]

        with self.get_raster(freq=freq, pol=pol) as img:
            y_start = img.y_start
            y_stop = img.y_stop

        return (y_start, y_stop)


@dataclass
class RSLC(SLC, NisarRadarProduct):
    @property
    def product_type(self) -> str:
        return "RSLC"

    @cached_property
    def _data_group_path(self) -> str:
        path = super()._data_group_path

        # Special handling for old UAVSAR test datasets that have paths
        # like "/science/LSAR/SLC/..."
        with h5py.File(self.filepath) as f:
            if path in f:
                return path
            elif self.product_spec_version == "0.0.0":
                slc_path = path.replace("RSLC", "SLC")
                if slc_path in f:
                    return slc_path
                else:
                    raise ValueError(
                        "Could not determine the path to the group containing"
                        " the primary datasets."
                    )
            else:
                raise ValueError(
                    f"self._data_group_path determined to be {path}, but this"
                    " is not a valid path in the input file."
                )

    @cached_property
    def _metadata_group_path(self) -> str:
        path = super()._metadata_group_path

        # Special handling for old UAVSAR test datasets that have paths
        # like "/science/LSAR/SLC/metadata..."
        with h5py.File(self.filepath) as f:
            if path in f:
                return path
            elif self.product_spec_version == "0.0.0":
                slc_path = path.replace("RSLC", "SLC")
                if slc_path in f:
                    return slc_path
                else:
                    raise ValueError(
                        "Could not determine the path to the group containing"
                        " the product metadata."
                    )
            else:
                raise ValueError(
                    f"self._metadata_group_path determined to be {path}, but"
                    " this is not a valid path in the input file."
                )

    def _get_dataset_handle(
        self, h5_file: h5py.File, raster_path: str
    ) -> h5py.Dataset:
        
        log = nisarqa.get_logger()

        # RSLC Product Spec says that NISAR RSLC rasters should be complex32,
        # which requires special handling to read and access.
        # As of h5py 3.8.0, h5py gained the ability to read complex32
        # datasets, however numpy and other downstream packages do not
        # necessarily have that flexibility.
        if nisarqa.is_complex32(h5_file[raster_path]):
            # The RSLC dataset is complex32, as desired. Use the
            # ComplexFloat16Decoder so that numpy et al can read the datasets.
            dataset = nisarqa.ComplexFloat16Decoder(h5_file[raster_path])
            pass_fail = "PASS"
            logger = log.info
        else:
            # Use h5py's standard reader
            dataset = h5_file[raster_path]
            pass_fail = "FAIL"
            logger = log.warning

        logger(
            f"({pass_fail}) PASS/FAIL Check: Product raster dtype conforms"
            " to RSLC Product Spec dtype of complex32."
        )

        return dataset

    def get_scene_center_along_track_spacing(self, freq: str) -> float:
        """
        Get along-track spacing at mid-swath.

        Get the along-track spacing at the center of the swath, in meters, of
        the radar grid corresponding to the specified frequency sub-band.

        Parameters
        ----------
        freq : {'A', 'B'}
            The frequency sub-band. Must be a valid sub-band in the product.

        Returns
        -------
        az_spacing : float
            The mid-swath along-track sample spacing, in meters.
        """

        # Future Alert!
        # The name of the dataset accessed via this function is also
        # accessed in NisarRadarProduct._get_raster_from_path().
        # It is not ideal to have the hardcoded path in two places.
        # It is also not ideal to need to fully initialize an entire
        # RadarRaster just to access a single dataset.
        # Also, only RSLC can access `sceneCenterAlongTrackSpacing` with only
        # `freq` as an input; RIFG/RUNW/ROFF also need to know the layer group.
        # For simplicity, let's hard-code this path in two places,
        # and update later if/when a better design pattern is needed.
        # See: RSLC.get_slant_range_spacing() for a related issue.

        @lru_cache
        def _get_scene_center_along_track_spacing(freq: str) -> float:
            path = f"{self.get_freq_path(freq)}/sceneCenterAlongTrackSpacing"
            with h5py.File(self.filepath) as f:
                try:
                    return f[path][()]
                except KeyError as e:
                    raise nisarqa.DatasetNotFoundError from e

        return _get_scene_center_along_track_spacing(freq)

    def get_slant_range_spacing(self, freq: str) -> float:
        """
        Get slant range spacing.

        Get the slant range spacing, in meters, of the radar grid corresponding
        to the specified frequency sub-band.

        Parameters
        ----------
        freq : {'A', 'B'}
            The frequency sub-band. Must be a valid sub-band in the product.

        Returns
        -------
        rg_spacing : float
            The slant range sample spacing, in meters.
        """

        # Future Alert!
        # It makes sense for this dataset to be an attribute in RadarRaster.
        # However, for now, only RSLC() products need access to this dataset,
        # and we should not clutter RIFG/RUNW/ROFF RadarRasters.
        # Additionally, like RSLC.get_scene_center_along_track_spacing(),
        # it is not ideal to need to fully initialize an entire
        # RadarRaster just to access a single dataset.
        # For simplicity, let's use this design pattern for now,
        # and update later if/when a better design pattern is needed.

        @lru_cache
        def _get_slant_range_spacing(freq: str) -> float:
            path = f"{self.get_freq_path(freq)}/slantRangeSpacing"
            with h5py.File(self.filepath) as f:
                try:
                    return f[path][()]
                except KeyError as e:
                    raise nisarqa.DatasetNotFoundError from e

        return _get_slant_range_spacing(freq)

    def get_processed_center_frequency(self, freq: str) -> float:
        """
        Get processed center frequency.

        Get the processed center frequency, in Hz, of the radar signal
        corresponding to the specified frequency sub-band. (It is assumed
        that the input product's processed center frequency is in Hz.)

        Parameters
        ----------
        freq : {'A', 'B'}
            The frequency sub-band. Must be a valid sub-band in the product.

        Returns
        -------
        proc_center_freq : float
            The processed center frequency, in Hz.
        """

        @lru_cache
        def _get_proc_center_freq(freq: str) -> float:
            path = f"{self.get_freq_path(freq)}/processedCenterFrequency"
            with h5py.File(self.filepath) as f:
                try:
                    proc_center_freq = f[path][()]
                except KeyError as e:
                    raise nisarqa.DatasetNotFoundError from e

                # As of R3.4, units for `processedCenterFrequency` are "Hz",
                # not MHz. Do a soft check that this the units are correct.
                try:
                    units = f[path].attrs["units"]
                except KeyError:
                    errmsg = (
                        "`processedCenterFrequency` missing 'units' attribute."
                    )
                    print(errmsg)
                    warnings.warn(errmsg)

                units = nisarqa.byte_string_to_python_str(units)
                # units should be either "hz" or "hertz", and not MHz
                if (units[0].lower() != "h") or (units[-1].lower() != "z"):
                    errmsg = (
                        "Input product's `processedCenterFrequency` dataset"
                        f" has units of {units}, but should be in Hertz."
                    )
                    print(errmsg)
                    warnings.warn(errmsg)

            return proc_center_freq

        return _get_proc_center_freq(freq)


@dataclass
class GSLC(SLC, NonInsarGeoProduct):
    @property
    def product_type(self) -> str:
        return "GSLC"

    def _get_dataset_handle(
        self, h5_file: h5py.File, raster_path: str
    ) -> nisarqa.GeoRaster:
        # As of R3.3 the GSLC workflow recently gained the ability
        # to generate products in complex32 format as well as complex64
        # with some bits masked out to improve compression.
        # If the input GSLC product has dtype complex32, then we'll need
        # to use ComplexFloat16Decoder.
        if nisarqa.is_complex32(h5_file[raster_path]):
            # The GSLC dataset is complex32. Use the
            # ComplexFloat16Decoder so that numpy et al can read the datasets.
            dataset = nisarqa.ComplexFloat16Decoder(h5_file[raster_path])
            pass_fail = "FAIL"
        else:
            # Use h5py's standard reader
            dataset = h5_file[raster_path]
            pass_fail = "PASS"

        print(
            f"({pass_fail}) PASS/FAIL Check: Product raster dtype conforms"
            " to GSLC Product Spec dtype of complex64."
        )

        return dataset


@dataclass
class GCOV(NonInsarGeoProduct):
    @property
    def product_type(self) -> str:
        return "GCOV"

    def _get_dataset_handle(
        self, h5_file: h5py.File, raster_path: str
    ) -> nisarqa.GeoRaster:
        # Use h5py's standard reader
        dataset = h5_file[raster_path]

        # Check the dataset dtype
        pol = raster_path.split("/")[-1]
        if pol[0:2] == pol[2:4]:
            # on-diagonal term dataset. These are float32 as of May 2023.
            spec_dtype = np.float32
        else:
            # off-diagonal term dataset. These are complex64 as of May 2023.
            spec_dtype = np.complex64

        raster_dtype = dataset.dtype
        pass_fail = "PASS" if (raster_dtype == spec_dtype) else "FAIL"

        print(
            f"({pass_fail}) PASS/FAIL Check: Product raster {raster_dtype}"
            f" conforms to GCOV Product Spec dtype of {spec_dtype}."
        )

        return dataset

    def get_layers_for_browse(self) -> dict[str, list[str]]:
        """
        Assign polarizations to grayscale or RGBA channels for the Browse Image.

        Only on-diagonal terms will be used to create the browse image.
        See `Notes` for details on the possible NISAR modes and assigned
        channels for LSAR band.
        SSAR is currently only minimally supported, so only a grayscale image
        will be created. Prioritization order to select the freq/pol to use:
            For frequency: Freq A then Freq B.
            For polarization: 'HHHH', then 'VVVV', then first pol found.

        Returns
        -------
        layers_for_browse : dict
            A dictionary containing the frequency+polarization combinations
            that will be used to generate the browse image.
            For GCOV, either `layers_for_browse['A']` or
            `layers_for_browse['B']` will exist, but not both.
            Its structure is:
            layers_for_browse['A']  : list of str, optional
                                         List of the Freq A polarization(s)
                                         required to create the browse image.
                                         Warning: Only on-diag terms supported.
            layers_for_browse['B']  : list of str, optional
                                         List of the Freq B polarizations
                                         required to create the browse image.
                                         Warning: Only on-diag terms supported.

        See Also
        --------
        save_browse : Assigns color channels and generates the browse PNG.

        Notes
        -----
        Unlike RSLC products, the polarizations contained within a GCOV product
        do not map to the NISAR mode table. For GCOV, The user selects a subset
        of polarizations of the RSLC to process. With that subset, the GCOV
        SAS workflow verifies if it should symmetrize the cross-polarimetric
        channels (HV and VH) into a single cross-polarimetric channel (HV),
        and also verifies if it should generate the full covariance or only
        the diagonal terms.
        Usually polarimetric symmetrization is applied; symmetrization
        joins HV and VH into a single polarimetric channel HV.
        Layer selection for LSAR GCOV Browse:
        - Frequency A is used if available. Otherwise, Frequency B.
        - If only one polarization is available, or if the images are cross-pol,
        make one layer into grayscale. This function selects that layer.
        - Otherwise, generate an RGB color composition, per the algorithm
        described in `save_gcov_browse_img()`. This function will gather the
        largest subset of: {HHHH, VVVV, (HVHV or VHVH)}, in prep for that
        function.
        GCOV and RTC-S1 pixels are square on the ground, so the multilooking
        factor is the same in both directions, depending only in the expected
        output dimensions.
        """
        layers_for_browse = {}

        # Get the frequency sub-band containing science mode data.
        freq = self.science_freq
        science_pols = self.get_pols(freq=freq)

        # SSAR is not fully supported by QA, so just make a simple grayscale
        if self.band == "S":
            # Prioritize Co-Pol
            if "HHHH" in science_pols:
                layers_for_browse[freq] = ["HHHH"]
            elif "VVVV" in science_pols:
                layers_for_browse[freq] = ["VVVV"]
            else:
                # Take the first available on-diagonal term
                for pol in science_pols:
                    if pol[0:2] == pol[2:4]:
                        layers_for_browse[freq] = [pol]
                        break
                else:
                    # Take first available pol, even if it is an off-diag term
                    layers_for_browse[freq] = [science_pols[0]]

            return layers_for_browse

        # The input file contains LSAR data. Will need to make
        # grayscale/RGB channel assignments

        # Keep only the on-diagonal polarizations
        # (On-diag terms have the same first two letters as second two letters,
        # e.g. HVHV or VVVV.)
        science_pols = [p for p in science_pols if (p[0:2] == p[2:4])]
        n_pols = len(science_pols)

        # Sanity check: There should always be on-diag pols for GCOV
        if n_pols == 0:
            raise ValueError(
                "No on-diagonal polarizations found in input GCOV."
            )

        elif n_pols == 1:
            # Only one image; it will be grayscale
            layers_for_browse[freq] = science_pols

        elif all(p.startswith(("R", "L")) for p in science_pols):
            # Only compact pol(s) are available. Create grayscale.
            # Per the Prioritization Order, use first available polarization
            for pol in ("RHRH", "RVRV", "LHLH", "LVLV"):
                if pol in science_pols:
                    layers_for_browse[freq] = [pol]
                    break
            else:
                # Use first available pol
                layers_for_browse[freq] = [science_pols[0]]

            assert len(layers_for_browse[freq]) == 1

        else:
            # Only keep "HHHH", "HVHV", "VHVH", "VVVV".
            keep = [
                p
                for p in science_pols
                if (p in ("HHHH", "HVHV", "VHVH", "VVVV"))
            ]

            # Sanity Check
            assert len(keep) >= 1

            # If both cross-pol terms are available, only keep one
            if ("HVHV" in keep) and ("VHVH" in keep):
                if ("VVVV" in keep) and not ("HHHH" in keep):
                    # Only VVVV is in keep, and not HHHH. So, prioritize
                    # keeping VHVH with VVVV.
                    keep.remove("HVHV")
                else:
                    # prioritize keeping "HVHV"
                    keep.remove("VHVH")

            layers_for_browse[freq] = keep

        # Sanity Checks
        if ("A" not in layers_for_browse) and ("B" not in layers_for_browse):
            raise ValueError(
                "Input file must contain either Frequency A or Frequency B"
                " iamges."
            )

        if len(layers_for_browse[freq]) == 0:
            raise ValueError(
                f"The input file's Frequency {freq} group does not contain "
                "the expected polarization names."
            )

        return layers_for_browse

    @staticmethod
    def save_browse(
        pol_imgs: dict[str, np.ndarray], filepath: str | os.PathLike
    ) -> None:
        """
        Save the given polarization images to a RGB or Grayscale PNG.

        Dimensions of the output PNG (in pixels) will be the same as the
        dimensions of the input polarization image array(s). (No scaling will
        occur.) Non-finite values will be made transparent.
        Color Channels will be assigned per the following pseudocode:
            If pol_imgs.keys() contains only one image, then:
                grayscale = <that image>
            Else:
                Red: first available co-pol of the list [HHHH, VVVV]
                Green: first of the list [HVHV, VHVH, VVVV]
                if Green is VVVV:
                    Blue: HHHH
                else:
                    Blue: first co-pol of the list [VVVV, HHHH]

        Parameters
        ----------
        pol_imgs : dict of numpy.ndarray
            Dictionary of 2D array(s) that will be mapped to specific color
            channel(s) for the output browse PNG.
            If there are multiple image arrays, they must have identical shape.
            Format of dictionary:
                pol_imgs[<polarization>] : <2D numpy.ndarray image>, where
                    <polarization> is a subset of:
                                            'HHHH', 'HVHV', 'VVVV', 'VHVH',
                                            'RHRH', 'RVRV', 'LVLV', 'LHLH'
            Example:
                pol_imgs['HHHH'] : <2D numpy.ndarray image>
                pol_imgs['VVVV'] : <2D numpy.ndarray image>
        filepath : path-like
            Full filepath for where to save the browse image PNG.

        See Also
        --------
        select_layers_for_browse : Function to select the layers.

        Notes
        -----
        Provided image array(s) must previously be image-corrected. This
        function will take the image array(s) as-is and will not apply
        additional image correction processing to them. This function
        directly combines the image(s) into a single browse image.
        If there are multiple input images, they must be thoughtfully prepared
        and standardized relative to each other prior to use by this function.
        For example, trying to combine a Freq A 20 MHz image
        and a Freq B 5 MHz image into the same output browse image might not go
        well, unless the image arrays were properly prepared and standardized
        in advance.
        """
        # WLOG, get the shape of the image arrays
        # They should all be the same shape; the check for this is below.
        first_img = next(iter(pol_imgs.values()))
        img_2D_shape = np.shape(first_img)
        for img in pol_imgs.values():
            # Input validation check
            if np.shape(img) != img_2D_shape:
                raise ValueError(
                    "All image arrays in `pol_imgs` must have the same shape."
                )

        # Only on-diagonal terms are supported.
        if not set(pol_imgs.keys()).issubset(set(nisarqa.GCOV_DIAG_POLS)):
            raise ValueError(
                f"{pol_imgs.keys()=}, must be a subset of"
                f" {nisarqa.GCOV_DIAG_POLS}"
            )

        # Assign channels

        if len(pol_imgs) == 1:
            # Single pol. Make a grayscale image.
            nisarqa.products.rslc.plot_to_grayscale_png(
                img_arr=first_img, filepath=filepath
            )

            # Return early, so that we do not try to plot to RGB
            return

        # Initialize variables. Later, check to ensure they were all used.
        red = None
        blue = None
        green = None

        for pol in ["HHHH", "VVVV"]:
            if pol in pol_imgs:
                red = pol_imgs[pol]
                break

        # There should only be one cross-pol in the input
        if ("HVHV" in pol_imgs) and ("VHVH" in pol_imgs):
            raise ValueError(
                "`pol_imgs` should only contain one cross-pol image."
                f"It contains {pol_imgs.keys()}. Please update logic in "
                "`_select_layers_for_gcov_browse()`"
            )

        for pol in ["HVHV", "VHVH", "VVVV"]:
            if pol in pol_imgs:
                green = pol_imgs[pol]

                if pol == "VVVV":
                    # If we get here, this means two things:
                    #   1: no cross-pol images were available
                    #   2: only HHHH and VVVV are available
                    # So, final assignment should be R: HHHH, G: VVVV, B: HHHH
                    blue = pol_imgs["HHHH"]
                else:
                    for pol2 in ["VVVV", "HHHH"]:
                        if pol2 in pol_imgs:
                            blue = pol_imgs[pol2]
                            break
                break

        # Sanity Check, and catch-all logic to make a browse image
        if any(arr is None for arr in (red, green, blue)):
            # If we get here, then the images provided are not one of the
            # expected cases. WLOG plot one of the image(s) in `pol_imgs`.
            warnings.warn(
                "The images provided are not one of the expected cases to form"
                " the GCOV browse image. Grayscale image will be created by"
                " default."
            )

            for gray_img in pol_imgs.values():
                nisarqa.products.rslc.plot_to_grayscale_png(
                    img_arr=gray_img, filepath=filepath
                )

        else:
            # Output the RGB Browse Image
            nisarqa.products.rslc.plot_to_rgb_png(
                red=red, green=green, blue=blue, filepath=filepath
            )


@dataclass
class InsarProduct(NisarProduct):
    def get_browse_freq_pol(self) -> tuple[str, str]:
        """
        Return the frequency and polarization for the browse image.

        Returns
        -------
        freq, pol : pair of str
            The frequency and polarization to use for the browse image.
        """
        for freq in ("A", "B"):
            if freq not in self.freqs:
                continue

            for pol in ("HH", "VV", "HV", "VH"):
                if pol not in self.get_pols(freq=freq):
                    continue

                return freq, pol

        # The input product does not contain the expected frequencies and/or
        # polarization combinations
        raise nisarqa.InvalidNISARProductError

    def _get_raster_name(self, raster_path: str) -> str:
        """
        Return a name for the raster, e.g. 'RSLC_LSAR_A_HH'.

        Parameters
        ----------
        raster_path : str
            Full path in `h5_file` to the desired raster dataset.
            Example:
                "/science/LSAR/GUNW/grids/frequencyA/interferogram/HH/unwrappedPhase"

        Returns
        -------
        name : str
            The human-understandable name that is derived from the dataset.
            Example:
                "GUNW_L_A_HH_unwrappedPhase"
        """
        # InSAR product. Example `raster_path` to parse:
        # "/science/LSAR/RIFG/swaths/frequencyA/pixelOffsets/HH/alongTrackOffset"
        band = self.band
        freq = "A" if ("frequencyA" in raster_path) else "B"
        path = raster_path.split("/")
        group = path[-3]
        pol = path[-2]
        layer = path[-1]

        # Sanity check
        assert pol in nisarqa.get_possible_pols(
            product_type=self.product_type.lower()
        )

        name = (
            f"{self.product_type.upper()}_{band}_{freq}_{group}_{pol}_{layer}"
        )
        return name

    def _get_dataset_handle(
        self, h5_file: h5py.File, raster_path: str
    ) -> h5py.Dataset:
        """
        Return a handle to the requested dataset.

        Parameters
        ----------
        h5_file : h5py.File
            File handle for the input file.
        raster_path : str
            Path in the input file to the desired dataset.
            Examples:
                "/science/LSAR/RSLC/swaths/frequencyA/HH"
                "/science/LSAR/RIFG/swaths/frequencyA/interferogram/HH/wrappedInterferogram"

        Returns
        -------
        dataset : h5py.Dataset
            Handle to the requested dataset.
        """
        # Unlike RSLC, GSLC, and GCOV, the product readers for
        # insar products do their dtype checking when the paths are fetched
        # (e.g. when calling get_along_track_offset(...)).
        # So, treat this as a wrapper function.
        return h5_file[raster_path]

    def _check_dtype(self, path: str, expected_dtype: np.dtype) -> None:
        """
        Check that the dataset found at `path` has the correct dtype.

        Parameters
        ----------
        path : str
            Path to a dataset inside the input product.
        expected_dtype : np.dtype
            The expected dtype for the dataset, e.g. np.complex64.
        """

        # Use lru_cache to minimize the amount of (slow) file i/o
        @lru_cache
        def _check_dtype_inner(path: str, expected_dtype: np.dtype) -> None:
            with h5py.File(self.filepath) as f:
                try:
                    dataset_handle = f[path]
                except KeyError:
                    raise nisarqa.DatasetNotFoundError

                product_dtype = dataset_handle.dtype

            # dataset.dtype returns e.g. "<f4" for NISAR products.
            # Use .base to convert to equivalent native numpy dtype.
            pass_fail = (
                "PASS" if (product_dtype.base == expected_dtype) else "FAIL"
            )

            print(
                f"({pass_fail}) PASS/FAIL Check: Input file's dataset has"
                f" type {product_dtype} which conforms to expected dtype"
                f" {expected_dtype}. Dataset: {path}"
            )

        _check_dtype_inner(path, expected_dtype)

    @abstractmethod
    def _get_path_containing_freq_pol(self, freq: str, pol: str) -> str:
        """
        Return a potential valid path in input HDF5 containing `freq` and `pol`.

        This is a helper method (for e.g. `get_pols()`) which provides a
        generic way to get the path to a group in the product containing
        a particular frequency+polarization.
        The returned path may or may not exist in the NISAR input product.

        Parameters
        ----------
        freq : str
            Frequency of interest. Must be one of "A" or "B".
        pol : str
            Polarization of interest. Examples: "HH" or "HV".

        Returns
        -------
        path : str
            A potential valid path in the dataset which incorporates the
            requested freq and pol.

        See Also
        --------
        get_pols : Returns the polarizations for the requested frequency.
        """
        pass

    def get_pols(self, freq: str) -> tuple[str, ...]:
        """
        Get the polarizations for the given frequency.

        Parameters
        ----------
        freq : str
            Either "A" or "B".

        Returns
        -------
        pols : tuple[str]
            Tuple of the available polarizations in the input product
            for the requested frequency.

        Raises
        ------
        DatasetNotFoundError
            If no polarizations were found for this frequency.
        InvalidNISARProductError
            If the polarizations found are inconsistent with the polarizations
            listed in product's `listOfPolarizations` dataset for this freq.
        """
        if freq not in ("A", "B"):
            raise ValueError(f"{freq=}, must be one of 'A' or 'B'.")

        @lru_cache
        def _get_pols(freq):
            pols = []
            with h5py.File(self.filepath) as f:
                for pol in nisarqa.get_possible_pols(self.product_type.lower()):
                    pol_path = self._get_path_containing_freq_pol(freq, pol)
                    try:
                        f[pol_path]
                    except KeyError:
                        print(
                            f"Did not locate polarization group at: {pol_path}"
                        )
                        pass
                    else:
                        print(f"Located polarization group at: {pol_path}")
                        pols.append(pol)

                # The product contains a list of expected polarizations.
                # While file is open, grab it to use in a sanity check later.
                path = f"{self.get_freq_path(freq)}/listOfPolarizations"
                list_of_pols_ds = f[path][...].tolist()

                nisarqa.verify_str_meets_isce3_conventions(f[path])

                # convert from byte strings to str
                list_of_pols_ds = [
                    bytes.decode(b, "utf-8") for b in list_of_pols_ds
                ]

            # Sanity checks
            if set(pols) != set(list_of_pols_ds):
                errmsg = (
                    f"Frequency {freq} contains polarizations {pols}, but"
                    f" `listOfPolarizations` says {list_of_pols_ds}"
                    " should be available."
                )
                print(errmsg)
                raise nisarqa.InvalidNISARProductError(errmsg)

            if not pols:
                # No polarizations were found for this frequency
                errmsg = f"No polarizations were found for frequency {freq}"
                print(errmsg)
                raise nisarqa.DatasetNotFoundError(errmsg)

            return pols

        return tuple(_get_pols(freq))

    def save_qa_metadata_to_h5(self, stats_h5: h5py.File) -> None:
        """
        Populate `stats_h5` file with a list of each available polarization.

        If the input file contains Frequency A, then this dataset will
        be created in `stats_h5`:
            /science/<band>/QA/data/frequencyA/listOfPolarizations

        If the input file contains Frequency B, then this dataset will
        be created in `stats_h5`:
            /science/<band>/QA/data/frequencyB/listOfPolarizations

        * Note: The paths are pulled from nisarqa.STATS_H5_QA_FREQ_GROUP.
        If the value of that global changes, then the path for the
        `listOfPolarizations` dataset(s) will change accordingly.

        Parameters
        ----------
        stats_h5 : h5py.File
            Handle to an h5 file where the list(s) of polarizations
            should be saved.
        """
        band = self.band

        for freq in self.freqs:
            list_of_pols = self.get_pols(freq=freq)

            nisarqa.create_dataset_in_h5group(
                h5_file=stats_h5,
                grp_path=nisarqa.STATS_H5_QA_FREQ_GROUP % (band, freq),
                ds_name="listOfPolarizations",
                ds_data=list_of_pols,
                ds_description=f"Polarizations for Frequency {freq}.",
            )


class WrappedGroup(InsarProduct):
    """
    Contains common functionality for products with a wrapped igram data group.

    As of Sept. 2023, this is only for RIFG and GUNW products. RUNW products
    do not contain this group.
    """

    @staticmethod
    @abstractmethod
    def _wrapped_group_path(freq: str, pol: str) -> str:
        """Path in input file to wrapped interferogram group."""
        pass

    def _get_path_containing_freq_pol(self, freq: str, pol: str) -> str:
        return self._wrapped_group_path(freq, pol)

    @contextmanager
    def get_wrapped_igram(
        self, freq: str, pol: str
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the complex-valued wrapped interferogram image *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._wrapped_group_path(freq=freq, pol=pol)
        path = f"{parent_path}/wrappedInterferogram"
        self._check_dtype(path=path, expected_dtype=np.complex64)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)

    @contextmanager
    def get_wrapped_coh_mag(
        self, freq: str, pol: str
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the wrapped coherence magnitude *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._wrapped_group_path(freq=freq, pol=pol)
        path = f"{parent_path}/coherenceMagnitude"
        self._check_dtype(path=path, expected_dtype=np.float32)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)


class UnwrappedGroup(InsarProduct):
    """
    Contains common functionality for products with unwrapped phase data group.

    As of Sept. 2023, this is only for RUNW and GUNW products. RIFG products
    do not contain this group.
    """

    @staticmethod
    @abstractmethod
    def _unwrapped_group_path(freq: str, pol: str) -> str:
        """Path in input file to unwrapped interferogram group.."""
        pass

    def _get_path_containing_freq_pol(self, freq: str, pol: str) -> str:
        return self._unwrapped_group_path(freq, pol)

    @contextmanager
    def get_unwrapped_phase(
        self, freq: str, pol: str
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the unwrapped phase image *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._unwrapped_group_path(freq=freq, pol=pol)
        path = f"{parent_path}/unwrappedPhase"
        self._check_dtype(path=path, expected_dtype=np.float32)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)

    @contextmanager
    def get_unwrapped_coh_mag(
        self, freq: str, pol: str
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the unwrapped coherence magnitude *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._unwrapped_group_path(freq=freq, pol=pol)
        path = f"{parent_path}/coherenceMagnitude"
        self._check_dtype(path=path, expected_dtype=np.float32)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)

    @contextmanager
    def get_ionosphere_phase_screen(
        self, freq: str, pol: str
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the ionosphere phase screen *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._unwrapped_group_path(freq=freq, pol=pol)
        path = f"{parent_path}/ionospherePhaseScreen"
        self._check_dtype(path=path, expected_dtype=np.float32)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)

    @contextmanager
    def get_ionosphere_phase_screen_uncertainty(
        self, freq: str, pol: str
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the ionosphere phase screen uncertainty *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._unwrapped_group_path(freq=freq, pol=pol)
        path = f"{parent_path}/ionospherePhaseScreenUncertainty"
        self._check_dtype(path=path, expected_dtype=np.float32)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)


class IgramOffsetsGroup(InsarProduct):
    """
    InSAR product where pixel offsets datasets only need freq and pol to locate.

    This includes RIFG, RUNW, GUNW products, but not ROFF and GOFF products.

    RIFG, RUNW, and GUNW's structure follow a pattern like:
        .../frequencyA/pixelOffsets/HH/<data set>
    ROFF and GOFF follow a pattern like:
        .../frequencyA/pixelOffsets/pixelOffsets/HH/layer2/<data set>

    Note that RIFG/RUNW/GUNW only require freq and pol to correctly locate the
    desired dataset, while ROFF and GOFF additionally require a layer number.

    See Also
    --------
    OffsetProduct :
        Class that handles the pixel offsets group for ROFF and GOFF products.
    """

    def _igram_offsets_group_path(self, freq: str, pol: str) -> str:
        """Path in input file to the pixel offsets group."""
        return f"{self.get_freq_path(freq)}/pixelOffsets/{pol}"

    def _get_path_containing_freq_pol(self, freq: str, pol: str) -> str:
        return self._igram_offsets_group_path(freq, pol)

    @contextmanager
    def get_along_track_offset(
        self, freq: str, pol: str
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the along track offsets image *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._igram_offsets_group_path(freq=freq, pol=pol)
        path = f"{parent_path}/alongTrackOffset"
        self._check_dtype(path=path, expected_dtype=np.float32)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)

    @contextmanager
    def get_slant_range_offset(
        self, freq: str, pol: str
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the slant range offsets image *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._igram_offsets_group_path(freq=freq, pol=pol)
        path = f"{parent_path}/slantRangeOffset"
        self._check_dtype(path=path, expected_dtype=np.float32)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)


@dataclass
class RIFG(WrappedGroup, IgramOffsetsGroup, NisarRadarProduct):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Make sure that all groups contain the same polarizations
        for freq in self.freqs:
            wrapped_pols = super(WrappedGroup, self).get_pols(freq)
            offset_pols = super(IgramOffsetsGroup, self).get_pols(freq)
            if set(wrapped_pols) != set(offset_pols):
                warnings.warn(
                    f"Wrapped interferogram group contains {wrapped_pols},"
                    f" but the pixel offsets group contains {offset_pols}."
                )

    @property
    def product_type(self) -> str:
        return "RIFG"

    def _wrapped_group_path(self, freq: str, pol: str) -> str:
        return f"{self.get_freq_path(freq)}/interferogram/{pol}"


@dataclass
class RUNW(UnwrappedGroup, IgramOffsetsGroup, NisarRadarProduct):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Make sure that all groups contain the same polarizations
        for freq in self.freqs:
            unwrapped_pols = super(UnwrappedGroup, self).get_pols(freq)
            offset_pols = super(IgramOffsetsGroup, self).get_pols(freq)
            if set(unwrapped_pols) != set(offset_pols):
                warnings.warn(
                    f"Unwrapped interferogram group contains {unwrapped_pols},"
                    f" but the pixel offsets group contains {offset_pols}."
                )

    @property
    def product_type(self) -> str:
        return "RUNW"

    def _unwrapped_group_path(self, freq: str, pol: str) -> str:
        return f"{self.get_freq_path(freq)}/interferogram/{pol}"


@dataclass
class GUNW(
    WrappedGroup,
    UnwrappedGroup,
    IgramOffsetsGroup,
    NisarGeoProduct,
):
    def __post_init__(self) -> None:
        super().__post_init__()

        # Make sure that all groups contain the same polarizations
        for freq in self.freqs:
            wrapped_pols = super(WrappedGroup, self).get_pols(freq)
            unwrapped_pols = super(UnwrappedGroup, self).get_pols(freq)
            offset_pols = super(IgramOffsetsGroup, self).get_pols(freq)

            if set(wrapped_pols) != set(unwrapped_pols):
                warnings.warn(
                    f"Wrapped interferogram group contains {wrapped_pols},"
                    " but the unwrapped phase image group contains "
                    f" {unwrapped_pols}."
                )
            if set(wrapped_pols) != set(offset_pols):
                warnings.warn(
                    f"Wrapped interferogram group contains {wrapped_pols},"
                    f" but the pixel offsets group contains {offset_pols}."
                )

    @property
    def product_type(self) -> str:
        return "GUNW"

    def _wrapped_group_path(self, freq, pol) -> str:
        return f"{self.get_freq_path(freq)}/wrappedInterferogram/{pol}"

    def _unwrapped_group_path(self, freq, pol) -> str:
        if self.product_spec_version == "0.0.0":
            return f"{self.get_freq_path(freq)}/interferogram/{pol}"
        else:
            # Path for product spec v0.9.0 (...and maybe subsequent versions?)
            return f"{self.get_freq_path(freq)}/unwrappedInterferogram/{pol}"

    @cached_property
    def browse_x_range(self) -> tuple[float, float]:
        freq, pol = self.get_browse_freq_pol()

        with self.get_unwrapped_phase(freq, pol) as img:
            x_start = img.x_start
            x_stop = img.x_stop

        return (x_start, x_stop)

    @cached_property
    def browse_y_range(self) -> tuple[float, float]:
        freq, pol = self.get_browse_freq_pol()

        with self.get_unwrapped_phase(freq, pol) as img:
            y_start = img.y_start
            y_stop = img.y_stop

        return (y_start, y_stop)


class OffsetProduct(InsarProduct):
    """
    InSAR product where datasets need freq, pol, and layer number to locate.

    This includes ROFF and GOFF products, but not RIFG, RUNW, GUNW products.

    RIFG, RUNW, and GUNW's structure follow a pattern like:
        .../frequencyA/pixelOffsets/HH/<data set>
    ROFF and GOFF follow a pattern like:
        .../frequencyA/pixelOffsets/pixelOffsets/HH/layer2/<data set>

    Note that RIFG/RUNW/GUNW only require freq and pol to correctly locate a
    specific dataset, while ROFF and GOFF additionally require a layer number.

    See Also
    --------
    IgramOffsetsGroup :
        Class that handles the pixel offsets group for RIFG, RUNW, and
        GUNW products.
    """

    def _get_raster_name(self, raster_path: str) -> str:
        """
        Return name for the raster, e.g. 'ROFF_L_A_HH_Layer1_alongTrackOffset'.

        Parameters
        ----------
        raster_path : str
            Full path in `h5_file` to the desired raster dataset.
            Example:
                "/science/LSAR/ROFF/swaths/frequencyA/pixelOffsets/HH/layer1/alongTrackOffset"

        Returns
        -------
        name : str
            The human-understandable name that is derived from the dataset.
            Example:
                "ROFF_L_A_HH_Layer1_alongTrackOffset"
        """
        # We have a ROFF or GOFF product. Example `raster_path` to parse:
        # "/science/LSAR/ROFF/swaths/frequencyA/pixelOffsets/HH/layer1/alongTrackOffset"
        band = self.band
        freq = "A" if ("frequencyA" in raster_path) else "B"
        path = raster_path.split("/")
        pol = path[-3]
        layer_num = path[-2]
        layer_name = path[-1]

        # Sanity check
        assert pol in nisarqa.get_possible_pols(
            product_type=self.product_type.lower()
        )

        name = f"{self.product_type.upper()}_{band}_{freq}_{pol}_{layer_num}_{layer_name}"
        return name

    @cached_property
    def available_layer_numbers(self) -> tuple[int, ...]:
        """
        The numbers of the available layers in the input product.

        Here, a "numbered layer" refers to the layer groups:
            /science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer1/...
            /science/LSAR/GOFF/grids/frequencyA/pixelOffsets/HH/layer2/...

        Each layer group in a product contains an indentical set of
        raster and metadata datasets (although the values in those datasets
        are unique).

        As of Sept 2023, ISCE3 insar.yaml runconfig file allows users the
        option to produce layer1, layer2, layer3, and/or layer4. (all optional)
        For nominal NISAR products, up to layers 1, 2, and 3 will be processed.
        (Not necessarily all 3.) Urgent response will only process layer1 and
        layer2. NISAR Science Team is itching to use up to 7 layers.
        Each layer is processed with a unique algorithm combination, which
        strikes a unique balance between the amount of noise and
        the coarseness of the granularity.

        Returns
        -------
        layers : tuple[int, ...]
            Tuple of the available layers.
        """

        def _get_available_layer_numbers(freq: str) -> tuple[int, ...]:
            golden_layers = []
            with h5py.File(self.filepath) as f:
                # if multiple pols, make sure they contain the same layers
                possible_pols = nisarqa.get_possible_pols(
                    self.product_type.lower()
                )
                for pol in possible_pols:
                    # As of Sept 2023, ISCE3 only supports up to 4 layers,
                    # however NISAR Science Team is asking for up to 7 layers.
                    layers_tmp = []
                    for l_num in range(1, 8):
                        path = self._numbered_layer_group_path(freq, pol, l_num)
                        try:
                            f[path]
                        except KeyError:
                            pass
                        else:
                            layers_tmp.append(l_num)

                    if not layers_tmp:
                        # no layers were located in this polarization
                        continue

                    # Sanity Checks that each polarization contains the
                    # same layer groups
                    if not golden_layers:
                        # First polarization found in product
                        golden_layers = layers_tmp.copy()
                        golden_pol = pol
                    elif set(golden_layers) != set(layers_tmp):
                        warnings.warn(
                            f"Freq {freq} Pol {golden_pol} contains layers"
                            f" {golden_layers}, but Freq {freq} Pol {pol}"
                            f" contains layers {layers_tmp}."
                        )

            if not golden_layers:
                warnings.warn(f"No layer groups found for Freq {freq}.")
                raise nisarqa.DatasetNotFoundError

            return golden_layers

        # self.freqs is a property containing only confirmed frequencies
        # in the input product. If a DatasetNotFoundError is raised, then
        # the input product is incorrectly formed. Let the error propogate up.
        layers_1 = _get_available_layer_numbers(self.freqs[0])

        # If multiple frequencies, ensure they contain the same layers
        if len(self.freqs) == 2:
            layers_2 = _get_available_layer_numbers(self.freqs[1])
            if set(layers_1) != set(layers_2):
                warnings.warn(
                    f"Frequency {self.freqs[0]} contains layers {layers_1}, but"
                    f" Frequency {self.freqs[1]} contains layers {layers_2}."
                )

        return layers_1

    def get_browse_freq_pol_layer(self) -> tuple[str, str, int]:
        """
        Return the frequency, polarization, and layer number for browse image.

        Returns
        -------
        freq : str
            The frequency to use for the browse image.
        pol : str
            The polarization to use for the browse image.
        layer_num : int
            The layer number to use for the browse image.
        """
        freq, pol = self.get_browse_freq_pol()

        # Prioritization order, as determined by insar product lead (Sept 2023).
        # Layer 3 should be nicest-looking for the browse image; compared
        # to the other layers, it has coarser granularity but is less noisy.
        priority_order = (3, 2, 1, 4, 5, 6, 7)

        for layer_num in priority_order:
            if layer_num in self.available_layer_numbers:
                return freq, pol, layer_num
        else:
            errmsg = (
                f"Prioritization order of layer groups is {priority_order}, but"
                " the product only contains layers"
                f" {self.available_layer_numbers}."
            )
            print(errmsg)
            raise nisarqa.InvalidNISARProductError(errmsg)

    def _get_path_containing_freq_pol(self, freq: str, pol: str) -> str:
        # Each polarization should contain the same layer numbers.
        # WLOG, use the first available layer number.
        return self._numbered_layer_group_path(
            freq=freq, pol=pol, layer_num=self.available_layer_numbers[0]
        )

    def _numbered_layer_group_path(
        self, freq: str, pol: str, layer_num: int
    ) -> str:
        """Get path in input file to the group for this numbered layer group."""
        return f"{self.get_freq_path(freq)}/pixelOffsets/{pol}/layer{layer_num}"

    @contextmanager
    def get_along_track_offset(
        self, freq: str, pol: str, layer_num: int
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the along track offset *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.
        layer_num : int
            Layer number of the desired raster. For example, to get the
            requested raster from the `layer1` group, set `layer_num` to `1`.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._numbered_layer_group_path(freq, pol, layer_num)
        path = f"{parent_path}/alongTrackOffset"
        self._check_dtype(path=path, expected_dtype=np.float32)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)

    @contextmanager
    def get_slant_range_offset(
        self, freq: str, pol: str, layer_num: int
    ) -> Iterator[nisarqa.RadarRaster | nisarqa.GeoRaster]:
        """
        Get the slant range offset *Raster.

        Parameters
        ----------
        freq, pol : str
            Frequency and polarization (respectively) for the desired raster.
        layer_num : int
            Layer number of the desired raster. For example, to get the
            requested raster from the `layer1` group, set `layer_num` to `1`.

        Yields
        ------
        raster : RadarRaster or GeoRaster
            Generated *Raster for the requested dataset.
        """
        parent_path = self._numbered_layer_group_path(freq, pol, layer_num)
        path = f"{parent_path}/slantRangeOffset"
        self._check_dtype(path=path, expected_dtype=np.float32)

        with h5py.File(self.filepath) as f:
            yield self._get_raster_from_path(h5_file=f, raster_path=path)


@dataclass
class ROFF(OffsetProduct, NisarRadarProduct):
    @property
    def product_type(self) -> str:
        return "ROFF"


@dataclass
class GOFF(OffsetProduct, NisarGeoProduct):
    @property
    def product_type(self) -> str:
        return "GOFF"

    @cached_property
    def browse_x_range(self) -> tuple[float, float]:
        freq, pol, layer = self.get_browse_freq_pol_layer()

        with self.get_along_track_offset(
            freq=freq, pol=pol, layer_num=layer
        ) as raster:
            x_start = raster.x_start
            x_stop = raster.x_stop

        return (x_start, x_stop)

    @cached_property
    def browse_y_range(self) -> tuple[float, float]:
        freq, pol, layer = self.get_browse_freq_pol_layer()

        with self.get_along_track_offset(
            freq=freq, pol=pol, layer_num=layer
        ) as raster:
            y_start = raster.y_start
            y_stop = raster.y_stop

        return (y_start, y_stop)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
