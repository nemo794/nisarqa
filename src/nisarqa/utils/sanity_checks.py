from __future__ import annotations

from collections.abc import Callable

import h5py
import numpy as np
from numpy.typing import ArrayLike

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)


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
                f"Dataset value is {data!r}, must be 'Left' or 'Right'."
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
