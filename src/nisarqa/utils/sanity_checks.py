from __future__ import annotations

from collections.abc import Callable

import h5py
import numpy as np
from numpy.typing import ArrayLike

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)


def validate_spacing(
    data: ArrayLike, spacing: float, dname: str, epsilon=1e-5
) -> None:
    """
    Validate that the values in `data` are in ascending order
    and equispaced, within a tolerance of +/- epsilon (EPS = 1.0E-05).

    Example usage: validate_spacing() can be used to validate that the
    timestamps in a zero doppler time array are always increasing,
    or that the distances in a slant range array are always increasing.

    Parameters
    ----------
    data : array_like
        The 1D data array to be validated.
    spacing : numeric
        The theoretical interval between each value in `data`.
    dataset_name : string
        Name of `data`, which will be used for log messages.
    """

    log = nisarqa.get_logger()

    # Validate that data's values are strictly increasing
    delta = data[1:] - data[:-1]

    if not np.all(delta > 0.0):
        idx = np.where(delta <= 0.0)
        log.info(
            f"{dname}: Found {len(idx[0])} elements with negative spacing: "
            f"{data[idx]} at locations {idx}"
        )

    # Validate that successive values in data are separated by `spacing`,
    # within a tolerance of +/- epsilon.
    EPS = 1.0e-05
    diff = np.abs(delta - spacing)
    try:
        assert np.all(diff <= EPS)
    except AssertionError as e:
        idx = np.where(diff > EPS)
        log.error(
            f"{dname}: Found {len(idx[0])} elements with unexpected steps: "
            f"{diff[idx]} at locations {idx}"
        )


def verify_shapes_are_consistent(product: nisarqa.NisarProduct) -> None:
    """
    Verify that the shape dimensions are consistent between datasets.

    For example, if the Dataset `zeroDopplerTime` has a shape of
    `frequencyALength` specified in the XML and has dimensions (132,),
    then all other Datasets with a shape of `frequencyALength` in the XML
    must also have a shape of (132,) in the HDF5.
    """

    # Create a dict of all groups and datasets inside the input file,
    # where the path is the key and the value is an hdf5 object

    # Parse all of the xml Shapes into a dict, initializing each to None.
    # key will be the Shape name,
    # value will be set the first time it is encountered.
    # Each subsequent time it is encountered, check that the shape is consistent

    # for every item in the xml_tree:
    #     if that path exists in the input file:
    #         check that the description matches the xml_file
    #         check that the units matches the xml_file

    #         Look at the shape:
    #             if Shape is in the NISAR Shapes:
    #                 if shape.value is None:
    #                     set the Value
    #                 else:
    #                     assert shape.value == the value in the dict

    #             if the shape is a known constant, confirm the actual data
    #                 has that shape

    #             else:
    #                 raise error, because each dataset should have a shape

    #         remove that path from the input file's dict

    # if the input file's dict is not empty:
    #     raise error - input file has extraneous datasets
    pass


def dataset_sanity_checks(product: nisarqa.NisarProduct) -> None:
    """
    Perform a series of verification checks on the input product's datasets.

    Parameters
    ----------
    product : nisarqa.NisarProduct
        Instance of the input product.
    """
    with h5py.File(product.filepath, "r") as f:

        identification_sanity_checks(
            id_group=f[product.identification_path],
            product_type=product.product_type,
        )


def identification_sanity_checks(
    id_group: h5py.Group, product_type: str
) -> None:
    """
    Perform sanity checks on Datasets in input product's identification group.

    Parameters
    ----------
    id_group : h5py.Group
        Handle to the `identification` group in the input product.
    product_type : str
        One of: 'rslc', 'gslc', 'gcov', 'rifg', 'runw', 'gunw', 'roff', 'goff',
        or their uppercase variants. (The NISAR convention is for the
        identification group's `productType` Dataset to be all uppercase.)
    """

    log = nisarqa.get_logger()

    def _full_path(ds_name: str) -> str:
        return f"{id_group.name}/{ds_name}"

    def _get_dataset(ds_name: str) -> np.ndarray | None:
        try:
            data = id_group[ds_name][()]
        except KeyError:
            log.error(f"Missing dataset: {_full_path(ds_name)}")
            return None
        return data

    def _get_integer_dataset(ds_name: str) -> int | None:
        data = _get_dataset(ds_name=ds_name)
        if (data is None) or np.issubdtype(data.dtype, np.integer):
            return data
        else:
            log.error(
                f"Dataset has dtype `{data.dtype}`, must be an integer type."
                f" Dataset: {_full_path(ds_name)}"
            )
            return None

    def _get_string_dataset(ds_name: str) -> str | None:
        data = _get_dataset(ds_name=ds_name)
        if isinstance(data, np.bytes_):
            return nisarqa.byte_string_to_python_str(data)
        elif data is None:
            return data
        else:
            log.error(
                f"Dataset has dtype `{data.dtype}`, must be a NumPy"
                f" byte string. Dataset: {_full_path(ds_name)}"
            )
            return None

    def _verify_greater_than_zero(value: int, ds_name: str) -> bool:
        if value <= 0:
            log.error(
                f"Dataset value is {value}, must be greater than zero."
                f" Dataset: {_full_path(ds_name)}"
            )
            return False
        return True

    passes = True

    ds_name = "absoluteOrbitNumber"
    data = _get_integer_dataset(ds_name=ds_name)
    if data is not None:
        passes &= _verify_greater_than_zero(value=data, ds_name=ds_name)
    else:
        passes = False

    ds_name = "trackNumber"
    data = _get_integer_dataset(ds_name=ds_name)
    if data is not None:
        passes &= _verify_greater_than_zero(value=data, ds_name=ds_name)
        if data > nisarqa.NUM_TRACKS:
            log.error(
                f"Dataset value is `{data}`, must be less than or equal to"
                f" total number of tracks, which is {nisarqa.NUM_TRACKS}."
                f" Dataset: {_full_path(ds_name)}"
            )
            passes = False
    else:
        passes = False

    ds_name = "frameNumber"
    data = _get_integer_dataset(ds_name=ds_name)
    if data is not None:
        passes &= _verify_greater_than_zero(value=data, ds_name=ds_name)
    else:
        passes = False

    ds_name = "productType"
    data = _get_string_dataset(ds_name=ds_name)
    if data is not None:
        if data != product_type.upper():
            log.error(
                f"Dataset value is {data}, must match the specified"
                f" product type of {product_type.upper()}."
                f" Dataset: {_full_path(ds_name)}"
            )
            passes = False
    else:
        passes = False

    ds_name = "lookDirection"
    data = _get_string_dataset(ds_name=ds_name)
    if data is not None:
        if data not in ("Left", "Right"):
            log.error(
                f"Dataset value is {data}, must be 'Left' or 'Right'."
                f" Dataset: {_full_path(ds_name)}"
            )
            passes = False
    else:
        passes = False

    summary = nisarqa.get_summary()
    if passes:
        summary.check_identification_group(result="PASS")
    else:
        summary.check_identification_group(result="FAIL")


__all__ = nisarqa.get_all(__name__, objects_to_skip)
