from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt
from osgeo import gdal, osr

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)


@dataclass
class MetadataCube2D:
    """
    2D metadata cube Dataset.

    Parameters
    ----------
    data : array_like
        Metadata cube with shape (Y, X). Can be a numpy.ndarray,
        h5py.Dataset, etc.
    name : str
        Name for this Dataset. If data is from an HDF5 file, suggest using
        the full path to the Dataset for `name`.
    y_coord_vector : array_like
        1D vector with shape (Y,) containing the coordinate values for the
        y axis of the datacube.
        For L1 products, this is the `zeroDopplerTime` corresponding to `data`.
        For L2 products, this is the `yCoordinates` corresponding to `data`.
    x_coord_vector : array_like
        1D vector with shape (X,) containing the coordinate values for the
        x axis of the datacube.
        For L1 products, this is the `slantRange` corresponding to `data`.
        For L2 products, this is the `xCoordinates` corresponding to `data`.
    """

    data: npt.ArrayLike
    name: str
    y_coord_vector: npt.ArrayLike
    x_coord_vector: npt.ArrayLike

    def __post_init__(self):
        if self.y_axis_dim != len(self.y_coord_vector):
            raise ValueError(
                f"Dataset {self.name} has y dimension {self.y_axis_dim},"
                f" which should match the length of the cooresponding y-axis"
                f" coordinate vector which is {len(self.y_coord_vector)}."
            )
        if self.x_axis_dim != len(self.x_coord_vector):
            raise ValueError(
                f"Dataset {self.name} has x dimension {self.x_axis_dim},"
                f" which should match the length of the cooresponding x-axis"
                f" coordinate vector which is {len(self.x_coord_vector)}."
            )
        arr = self.data[()]
        if not np.isfinite(arr).any():
            raise nisarqa.InvalidRasterError(
                f"Metadata cube {self.name} contains all non-finite"
                " (e.g. NaN) values."
            )
        if np.all(np.abs(arr) < 1e-12):
            raise nisarqa.InvalidRasterError(
                f"Metadata cube {self.name} contains all near-zero"
                " (<1e-12) values."
            )

    @property
    def y_axis_dim(self) -> int:
        """Y-axis dimension of the datacube."""
        return np.shape(self.data)[0]

    @property
    def x_axis_dim(self) -> int:
        """X-axis dimension of the datacube."""
        return np.shape(self.data)[1]


@dataclass
class MetadataCube3D(MetadataCube2D):
    """
    3D metadata cube Dataset.

    Parameters
    ----------
    data : array_like
        Metadata cube with shape (Z, Y, X). Can be a numpy.ndarray,
        h5py.Dataset, etc.
    name : str
        Name for this Dataset. If data is from an HDF5 file, suggest using
        the full path to the Dataset for `name`.
    x_coord_vector : array_like
        1D vector with shape (X,) containing the coordinate values for the
        x axis of the datacube.
        For L1 products, this is the `slantRange` corresponding to `data`.
        For L2 products, this is the `xCoordinates` corresponding to `data`.
    y_coord_vector : array_like
        1D vector with shape (Y,) containing the coordinate values for the
        y axis of the datacube.
        For L1 products, this is the `zeroDopplerTime` corresponding to `data`.
        For L2 products, this is the `yCoordinates` corresponding to `data`.
    z_coord_vector : array_like
        1D vector with shape (Z,) containing the coordinate values for the
        z axis of the datacube.
        For NISAR, this is `heightAboveEllipsoid` corresponding to `data`.
    """

    z_coord_vector: npt.ArrayLike

    def __post_init__(self):

        super().__post_init__()

        len_z = len(self.z_coord_vector)
        if self.z_axis_dim != len_z:
            if self.name.endswith("Baseline"):
                print(self.data[()])
                print("adfa \n", np.sum(np.isnan(self.data[()])))
                print(f"{np.size(self.data)}")

                # `parallelBaseline` and `perpendicularBaseline` Datasets
                # either have a height of 2 or the length of the z coordinates
                if self.z_axis_dim != 2:
                    raise nisarqa.InvalidRasterError(
                        f"Dataset {self.name} has z dimension {self.z_axis_dim},"
                        " which should either be 2 or match the length of the"
                        " cooresponding z-axis coordinate vector which is"
                        f" {len_z}."
                    )
                else:
                    len_z = 2
            else:
                raise nisarqa.InvalidRasterError(
                    f"Dataset {self.name} has z dimension {self.z_axis_dim},"
                    f" which should match the length of the cooresponding"
                    f" z-axis coordinate vector which is {len_z}."
                )

        for z in range(len_z):
            if not np.isfinite(self.data[z, :, :]).any():
                raise nisarqa.InvalidRasterError(
                    f"Metadata cube {self.name} z-axis layer number {z}"
                    " contains all non-finite (e.g. NaN) values."
                )
            if np.all(np.abs(self.data[z, :, :]) < 1e-12):
                raise nisarqa.InvalidRasterError(
                    f"Metadata cube {self.name} z-axis layer number {z}"
                    " contains all near-zero (<1e-12) values."
                )

    @property
    def z_axis_dim(self) -> int:
        """Z-axis dimension of the datacube."""
        return np.shape(self.data)[0]

    @property
    def y_axis_dim(self) -> int:
        return np.shape(self.data)[1]

    @property
    def x_axis_dim(self) -> int:
        return np.shape(self.data)[2]


def verify_metadata_cubes(
    product: nisarqa.NisarProduct,
) -> None:

    # Check metadata cubes in metadata Group
    # Note: During the __post_init__ of constructing each MetadataCube3D,
    # several validation checks are performed. Do not exit the loop early.
    for cube in product.coordinate_grid_metadata_cubes():
        if product.is_geocoded and isinstance(cube, h5py.Dataset):
            verify_gdal_friendly(
                input_filepath=product.filepath, ds_path=cube.data.name
            )

    # Non-InSAR products have calibrationInformation groups
    # for cube in product.cal_info_metadata_cubes():


def verify_gdal_friendly(input_filepath: str, ds_path: str) -> None:

    gdal_ds = gdal.Open(f'NETCDF:"{input_filepath}":{ds_path}')

    # Make sure it has a spatial reference
    proj = osr.SpatialReference(wkt=gdal_ds.GetProjection())
    epsg = proj.GetAttrValue("AUTHORITY", 1)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
