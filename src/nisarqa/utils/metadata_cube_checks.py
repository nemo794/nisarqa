from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from osgeo import gdal, osr

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)


@dataclass
class MetadataCube1D:
    """
    1D metadata cube Dataset.

    Parameters
    ----------
    data : np.ndarray
        Metadata cube with shape (X,). Can be a numpy.ndarray,
        h5py.Dataset, etc.
    name : str
        Name for this Dataset. If source data is from an HDF5 file, suggest
        using the full path to the Dataset for `name`.
    x_coord_vector : np.ndarray
        1D vector with shape (X,) containing the coordinate values for the
        x axis of the datacube.
        For L1 products, this is the `slantRange` corresponding to `data`.
        For L2 products, this is the `xCoordinates` corresponding to `data`.
    """

    data: np.ndarray
    name: str
    x_coord_vector: np.ndarray

    def __post_init__(self):

        # Metadata cubes are, by design, very small in size. So, coerce these
        # input ArrayLike objects to NumPy arrays during init.
        # Otherwise, they may be repeatedly converted to (temporary) NumPy
        # arrays as we pass them to NumPy functions, matplotlib, etc.
        self.data = np.asanyarray(self.data)
        self.x_coord_vector = np.asanyarray(self.x_coord_vector)

        if self.shape[-1] != len(self.x_coord_vector):
            raise ValueError(
                f"Dataset {self.name} has x dimension {self.shape[-1]},"
                f" which should match the length of the cooresponding x-axis"
                f" coordinate vector which is {len(self.x_coord_vector)}."
            )

    @property
    def shape(self) -> tuple[int]:
        """Shape of the metadata cube."""
        return self.data.shape


@dataclass
class MetadataCube2D(MetadataCube1D):
    """
    2D metadata cube Dataset.

    Parameters
    ----------
    data : np.ndarray
        Metadata cube with shape (Y, X). Can be a numpy.ndarray,
        h5py.Dataset, etc.
    name : str
        Name for this Dataset. If data is from an HDF5 file, suggest using
        the full path to the Dataset for `name`.
    x_coord_vector : np.ndarray
        1D vector with shape (X,) containing the coordinate values for the
        x axis of the datacube.
        For L1 products, this is the `slantRange` corresponding to `data`.
        For L2 products, this is the `xCoordinates` corresponding to `data`.
    y_coord_vector : np.ndarray
        1D vector with shape (Y,) containing the coordinate values for the
        y axis of the datacube.
        For L1 products, this is the `zeroDopplerTime` corresponding to `data`.
        For L2 products, this is the `yCoordinates` corresponding to `data`.
    """

    y_coord_vector: np.ndarray

    def __post_init__(self):

        super().__post_init__()

        self.y_coord_vector = np.asanyarray(self.y_coord_vector)

        if self.shape[-2] != len(self.y_coord_vector):
            raise ValueError(
                f"Dataset {self.name} has y dimension {self.shape[-2]},"
                f" which should match the length of the cooresponding y-axis"
                f" coordinate vector which is {len(self.y_coord_vector)}."
            )


@dataclass
class MetadataCube3D(MetadataCube2D):
    """
    3D metadata cube Dataset.

    Parameters
    ----------
    data : np.ndarray
        Metadata cube with shape (Z, Y, X). Can be a numpy.ndarray,
        h5py.Dataset, etc.
    name : str
        Name for this Dataset. If data is from an HDF5 file, suggest using
        the full path to the Dataset for `name`. If `name` ends with 'Baseline',
        the dataset will be assumed to be a parallel baseline or perpendicular
        baseline dataset, which may have a z-dimension of 2.
    x_coord_vector : np.ndarray
        1D vector with shape (X,) containing the coordinate values for the
        x axis of the datacube.
        For L1 products, this is the `slantRange` corresponding to `data`.
        For L2 products, this is the `xCoordinates` corresponding to `data`.
    y_coord_vector : np.ndarray
        1D vector with shape (Y,) containing the coordinate values for the
        y axis of the datacube.
        For L1 products, this is the `zeroDopplerTime` corresponding to `data`.
        For L2 products, this is the `yCoordinates` corresponding to `data`.
    z_coord_vector : np.ndarray
        1D vector with shape (Z,) containing the coordinate values for the
        z axis of the datacube.
        For NISAR, this is `heightAboveEllipsoid` corresponding to `data`.
    """

    z_coord_vector: np.ndarray

    def __post_init__(self):

        super().__post_init__()
        self.z_coord_vector = np.asanyarray(self.z_coord_vector)

        len_z = len(self.z_coord_vector)
        if self.shape[-3] != len_z:
            if self.name.endswith("Baseline"):
                # `parallelBaseline` and `perpendicularBaseline` Datasets
                # either have a height of 2 or the length of the z coordinates
                if self.shape[-3] != 2:
                    raise nisarqa.InvalidRasterError(
                        f"Dataset {self.name} has z dimension {self.shape[-3]},"
                        " which should either be 2 or match the length of the"
                        " cooresponding z-axis coordinate vector which is"
                        f" {len_z}."
                    )
                else:
                    len_z = 2
            else:
                raise nisarqa.InvalidRasterError(
                    f"Dataset {self.name} has z dimension {self.shape[-3]},"
                    f" which should match the length of the cooresponding"
                    f" z-axis coordinate vector which is {len_z}."
                )


def verify_metadata_cubes(
    product: nisarqa.NisarProduct, fail_if_all_nan: bool = True
) -> None:
    """
    Verify if the input product's metadata cubes are valid.

    Parameters
    ----------
    product : nisarqa.NisarProduct
        Instance of the input product. Note: there will be additional checks
    fail_if_all_nan : bool, optional
        True to raise an exception if one of the metadata cubes contains
        all non-finite (e.g. Nan, +/- Inf) values, or if one of the
        z-dimension height layers in a 3D cube has all non-finite values.
        False to quiet the exception, although it will still be logged.
        Defaults to True.

    Raises
    ------
    nisarqa.InvalidRasterError
        If one of the metadata cubes contains all non-finite (e.g. Nan, +/- Inf)
        values, or if one of the z-dimension height layers in a 3D cube has
        all non-finite values.
    """

    # Flag indicating if metadata cubes pass all verification checks; used
    # for Summary CSV reporting
    passes = True
    has_finite = True

    # helper function to check if the cube is GDAL-friendly.
    # no-op if the cube is not geocoded and/or not an h5py.Dataset
    def _check_gdal(c: nisarqa.MetadataCube2D | nisarqa.MetadataCube3D) -> bool:
        if product.is_geocoded and isinstance(c.data, h5py.Dataset):
            return is_gdal_friendly(
                input_filepath=product.filepath, ds_path=c.data.name
            )
        else:
            return True

    # Check metadata cubes in metadata Group
    try:
        # Note: During the __post_init__ of constructing each MetadataCube,
        # several validation checks are performed, including ensuring that
        # there are corresponding datasets with x coordinates and y coordinates
        # of the correct length. If these elements are missing, exceptions
        # will get thrown.
        for cube in product.coordinate_grid_metadata_cubes():
            has_finite &= _metadata_cube_has_finite_pixels(cube)
            passes &= has_finite
            passes &= _metadata_cube_is_not_all_zeros(cube)
            passes &= _check_gdal(c=cube)

        spec = nisarqa.Version.from_string(product.product_spec_version)

        # Non-InSAR products have calibrationInformation groups
        if isinstance(product, nisarqa.NonInsarProduct):
            for freq in product.freqs:

                if spec >= nisarqa.Version(1, 1, 0):
                    for cube in product.nes0_metadata_cubes(freq):
                        has_finite = _metadata_cube_has_finite_pixels(cube)
                        passes &= has_finite
                        passes &= _metadata_cube_has_finite_pixels(cube)
                        passes &= _metadata_cube_is_not_all_zeros(cube)
                        passes &= _check_gdal(c=cube)

                    for cube in product.elevation_antenna_pat_metadata_cubes(
                        freq
                    ):
                        has_finite &= _metadata_cube_has_finite_pixels(cube)
                        passes &= has_finite
                        passes &= _metadata_cube_is_not_all_zeros(cube)
                        passes &= _check_gdal(c=cube)

                    if issubclass(type(product), nisarqa.RSLC):
                        for cube in product.geometry_metadata_cubes():
                            has_finite &= _metadata_cube_has_finite_pixels(cube)
                            passes &= has_finite
                            passes &= _metadata_cube_is_not_all_zeros(cube)

                        for cube in product.crosstalk_metadata_cubes():
                            has_finite &= _metadata_cube_has_finite_pixels(cube)
                            passes &= has_finite
                            passes &= _metadata_cube_is_not_all_zeros(cube)

    except (nisarqa.DatasetNotFoundError, ValueError) as e:
        nisarqa.get_logger().error(e.__traceback__)
        passes = False

    # SUMMARY LOG
    summary = nisarqa.get_summary()
    summary.check_metadata_cubes(result="PASS" if passes else "FAIL")

    if fail_if_all_nan and (not has_finite):
        raise nisarqa.InvalidRasterError(
            "One of the metadata cubes contains all non-finite (e.g. NaN,"
            " +/1 Inf) values or one of the z-dimension height layers in a"
            " 3D cube has all non-finite values. See log for full details."
        )


def is_gdal_friendly(input_filepath: str, ds_path: str) -> bool:
    """
    Return True if the Dataset is GDAL-friendly, False if not.

    This function uses GDAL to open the file and test if the specified
    Dataset is georeferenced in a way that GDAL can read. For example, this
    function tests if that Dataset has a valid `grid_mapping` attribute, if
    it has a valid spatial reference and projection, etc.

    Parameters
    ----------
    input_filepath : str
        Full filepath to the input NISAR product.
    ds_path : str
        HDF5 Path in `input_filepath` to a (hopefully) georeferenced Dataset.

    Returns
    -------
    passes : bool
        True if the Dataset at `ds_path` is GDAL-friendly, False if GDAL
        cannot successfully georeference the Dataset.
    """

    log = nisarqa.get_logger()

    good_msg = f"Dataset is GDAL-friendly: {ds_path}"
    bad_msg = good_msg.replace("is ", "is not ")

    gdal_ds = gdal.Open(f'NETCDF:"{input_filepath}":{ds_path}')

    # If Dataset does not have a `grid_mapping` attribute, then GetProjection()
    # returns an empty string. However, GDAL will still construct a spatial
    # reference from that empty string, which has a valid EPSG code.

    # However, if, for example, the Dataset's corresponding `xCoordinates`
    # Dataset is the incorrect length, then GetProjection() raises an
    # AttributeError.

    try:
        wkt = gdal_ds.GetProjection()
    except AttributeError:
        log.error(bad_msg)
        return False

    if wkt == "":
        log.error(bad_msg)
        return False

    proj = osr.SpatialReference(wkt=wkt)

    # If Dataset's corresponding "projection" Dataset is set to a bad EPSG
    # code (e.g. 9999), then `proj.GetAttrValue("AUTHORITY", 1)` returns None.
    # Otherwise, it returns the EPSG code, e.g. 32645
    epsg = proj.GetAttrValue("AUTHORITY", 1)

    # If Dataset does not have a `grid_mapping` attribute, then GetSpatialRef()
    # returns None.
    crs = gdal_ds.GetSpatialRef()

    if (crs is None) or (epsg is None):
        log.error(bad_msg)
        return False
    # Note: this is specifically checking that the dataset used a map
    # projection (e.g. UTM, UPS)
    # It is *not* checking that the dataset contained a projection, which
    # just means that it stored coordinate system info.
    # If the dataset is Lon/Lat, this check will be false
    elif crs.IsProjected() == 1:
        log.info(good_msg)
        return True
    else:
        log.error(bad_msg)
        return False


def _metadata_cube_has_finite_pixels(cube: nisarqa.MetadataCube1D) -> bool:
    """
    Return False if metedata cube contains all non-finite values; True otherwise.

    Parameters
    ----------
    cube : nisarqa.MetadataCube1D
        MetadataCube to be checked.

    Returns
    -------
    passes : bool
        True if the cube contains at least one finite pixel. (Meaning, it is
        considered a valid dataset.)
        False if the given metedata cube contain all non-finite (e.g. NaN,
        +/- Inf) values, it is likely a mal-formed Dataset.
        Or, if the cube is a 3D metadata cube, and if any of the z-dimension
        height layers is all non-finite, it is also considered malformed and we
        return False.
    """
    log = nisarqa.get_logger()

    if not np.isfinite(cube.data[()]).any():
        log.error(
            f"Metadata cube {cube.name} contains all non-finite"
            " (e.g. NaN) values."
        )
        return False

    # For 3-D cubes, check each z-layer individually for all-NaN values.
    if isinstance(cube, MetadataCube3D):
        for z in range(cube.shape[0]):
            if not np.isfinite(cube.data[z, :, :]).any():
                log.error(
                    f"Metadata cube {cube.name} z-axis layer number {z}"
                    " contains all non-finite (e.g. NaN) values."
                )
                return False

    return True


def _metadata_cube_is_not_all_zeros(cube: nisarqa.MetadataCube1D) -> bool:
    """
    Return False if metedata cube contains all near-zeros; True otherwise.

    Parameters
    ----------
    cube : nisarqa.MetadataCube1D
        MetadataCube to be checked.

    Returns
    -------
    Passes : bool
        True if the cube contains at least one non-near-zero pixel. (Meaning,
        it is considered a valid dataset.)
        False if the given metedata cube contain all near-zero ( <1e-12 )
        values, it is likely a malformed Dataset.
        Or, if the cube is a 3D metadata cube, and if any of the z-dimension
        height layers is all near-zero, it is also considered malformed and we
        return False.
    """
    log = nisarqa.get_logger()

    if np.all(np.abs(cube.data[()]) < 1e-12):
        # This check is likely to raise a lot of failures.
        # We do not want to halt processing during CalVal.
        # So, issue obnoxious warnings for now.
        # TODO - refine this check during CalVal once real data comes back.
        msg = (
            f"Metadata cube {cube.name} contains all near-zero"
            " (<1e-12) values."
        )
        log.warning(msg)
        warnings.warn(msg, RuntimeWarning)
        return False

    # For 3-D cubes, check each z-layer individually for all near-zero values.
    if isinstance(cube, MetadataCube3D):
        for z in range(cube.shape[0]):
            if np.all(np.abs(cube.data[z, :, :]) < 1e-12):
                # This check is likely to raise a lot of failures.
                # We do not want to halt processing during CalVal.
                # So, issue obnoxious warnings for now.
                # TODO - refine this check during CalVal once real data comes back.
                msg = (
                    f"Metadata cube {cube.name} z-axis layer number {z}"
                    " contains all near-zero (<1e-12) values."
                )
                log.warning(msg)
                warnings.warn(msg, RuntimeWarning)
                return False
    return True


__all__ = nisarqa.get_all(__name__, objects_to_skip)
