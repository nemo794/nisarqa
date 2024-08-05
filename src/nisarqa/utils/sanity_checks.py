from __future__ import annotations

from collections.abc import Callable, Sequence

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
        if (data is not None) and nisarqa.verify_str_meets_isce3_conventions(
            ds=id_group[ds_name]
        ):
            return nisarqa.byte_string_to_python_str(data)
        else:
            return None

    def _verify_greater_than_zero(value: int | None, ds_name: str) -> bool:
        if (value is None) or (value <= 0):
            log.error(
                f"Dataset value is {value}, must be greater than zero."
                f" Dataset: {_full_path(ds_name)}"
            )
            return False
        return True

    def _verify_data_in_list_of_strings(
        value: str | None, list_of_valid_strings: Container[str], ds_name: str
    ) -> bool:
        if (value is None) or (value not in list_of_valid_strings):
            log.error(
                f"Dataset value is {value!r}, must be one of "
                f" {list_of_valid_strings}. Dataset: {_full_path(ds_name)}"
            )
            return False
        return True

    passes = True

    keys_checked = set()

    ds_name = "absoluteOrbitNumber"
    keys_checked.add(ds_name)
    data = _get_integer_dataset(ds_name=ds_name)
    passes &= _verify_greater_than_zero(value=data, ds_name=ds_name)

    ds_name = "trackNumber"
    keys_checked.add(ds_name)
    data = _get_integer_dataset(ds_name=ds_name)
    passes &= _verify_greater_than_zero(value=data, ds_name=ds_name)
    if (data is None) or (data > nisarqa.NUM_TRACKS):
        log.error(
            f"Dataset value is {data}, must be less than or equal to"
            f" total number of tracks, which is {nisarqa.NUM_TRACKS}."
            f" Dataset: {_full_path(ds_name)}"
        )
        passes = False

    ds_name = "frameNumber"
    keys_checked.add(ds_name)
    data = _get_integer_dataset(ds_name=ds_name)
    passes &= _verify_greater_than_zero(value=data, ds_name=ds_name)
    if (data is None) or (data > nisarqa.NUM_FRAMES):
        log.error(
            f"Dataset value is {data}, must be less than or equal to"
            f" total number of frames, which is {nisarqa.NUM_FRAMES}."
            f" Dataset: {_full_path(ds_name)}"
        )
        passes = False

    ds_name = "diagnosticModeFlag"
    keys_checked.add(ds_name)
    data = _get_integer_dataset(ds_name=ds_name)
    if (data is None) or (data not in (0, 1, 2)):
        log.error(
            f"Dataset value is {data}, must be 0, 1, or 2."
            f" Dataset: {_full_path(ds_name)}"
        )
        passes = False

    ds_name = "productType"
    keys_checked.add(ds_name)
    data = _get_string_dataset(ds_name=ds_name)
    if (data is None) or (data != product_type.upper()):
        log.error(
            f"Dataset value is {data}, must match the specified"
            f" product type of {product_type.upper()}."
            f" Dataset: {_full_path(ds_name)}"
        )
        passes = False

    ds_name = "lookDirection"
    keys_checked.add(ds_name)
    data = _get_string_dataset(ds_name=ds_name)
    passes &= _verify_data_in_list_of_strings(
        value=data, list_of_valid_strings=("Left", "Right"), ds_name=ds_name
    )

    ds_name = "productLevel"
    keys_checked.add(ds_name)
    data = _get_string_dataset(ds_name=ds_name)
    passes &= _verify_data_in_list_of_strings(
        value=data,
        list_of_valid_strings=("L0A", "L0B", "L1", "L2"),
        ds_name=ds_name,
    )

    ds_name = "radarBand"
    keys_checked.add(ds_name)
    data = _get_string_dataset(ds_name=ds_name)
    passes &= _verify_data_in_list_of_strings(
        value=data, list_of_valid_strings=("L", "S"), ds_name=ds_name
    )

    ds_name = "orbitPassDirection"
    keys_checked.add(ds_name)
    data = _get_string_dataset(ds_name=ds_name)
    passes &= _verify_data_in_list_of_strings(
        value=data,
        list_of_valid_strings=("Ascending", "Descending"),
        ds_name=ds_name,
    )

    ds_name = "processingType"
    keys_checked.add(ds_name)
    data = _get_string_dataset(ds_name=ds_name)
    passes &= _verify_data_in_list_of_strings(
        value=data,
        list_of_valid_strings=("Nominal", "Urgent", "Custom", "Undefined"),
        ds_name=ds_name,
    )

    # Verify Boolean Datasets
    for ds_name in (
        "isDithered",
        "isGeocoded",
        "isMixedMode",
        "isUrgentObservation",
    ):
        keys_checked.add(ds_name)
        data = _get_string_dataset(ds_name=ds_name)
        if data is not None:
            passes &= nisarqa.verify_isce3_boolean(ds=id_group[ds_name])

    # Verify "Version" Datasets (major, minor, patch)
    for ds_name in (
        "productVersion",
        "productSpecificationVersion",
    ):
        keys_checked.add(ds_name)
        data = _get_string_dataset(ds_name=ds_name)
        if data is not None:
            try:
                nisarqa.Version.from_string(version_str=data)
            except ValueError:
                passes = False

    # Verify datetime Datasets
    # TODO: Confirm correct "processingDateTime" precision with ADT.
    # (seconds, nanoseconds, microseconds, etc.)
    ds_name = "processingDateTime"
    keys_checked.add(ds_name)
    data = _get_string_dataset(ds_name=ds_name)
    if data is not None:
        passes &= nisarqa.check_datetime_string(
            datetime_str=data,
            dataset_name=_full_path(ds_name),
            precision="seconds",
        )

    if product_type.lower() in nisarqa.LIST_OF_INSAR_PRODUCTS:
        dt_datasets = (
            "referenceZeroDopplerStartTime",
            "secondaryZeroDopplerStartTime",
            "referenceZeroDopplerEndTime",
            "secondaryZeroDopplerEndTime",
        )
    else:
        dt_datasets = (
            "zeroDopplerStartTime",
            "zeroDopplerEndTime",
        )

    for ds_name in dt_datasets:
        keys_checked.add(ds_name)
        data = _get_string_dataset(ds_name=ds_name)
        if data is not None:
            passes &= nisarqa.check_datetime_string(
                datetime_str=data,
                dataset_name=_full_path(ds_name),
                precision="nanoseconds",
            )

    # These are datasets which need more-robust pattern-matching checks.
    # For now, just check that they are being populated with a non-dummy value.
    for ds_name in (
        "compositeReleaseId",
        "granuleId",
        "missionId",
        "plannedObservationId",
        "processingCenter",
        "listOfFrequencies",
        "boundingPolygon",
        "instrumentName",
        "plannedDatatakeId",
    ):
        keys_checked.add(ds_name)
        data = _get_string_dataset(ds_name=ds_name)
        # TODO: Improve error message by adding another conditional for a string
        # representation of a list of empty strings, e.g. "['' '' '' '' '']".
        if (data is None) or (data == "") or (data == "0") or (data == "['0']"):
            log.error(
                f"Dataset value is {data!r}, which is not a valid value."
                f" Dataset: {_full_path(ds_name)}"
            )
            passes = False
        else:
            log.warn(
                f"Dataset value is {data!r}, but it has not be automatically"
                f" verified during checks. Dataset: {_full_path(ds_name)}"
            )

    # Log if any Datasets were not verified
    keys_in_product = set(id_group.keys())
    difference = keys_in_product - keys_checked
    if len(difference) > 0:
        log.error(
            "Datasets found in product's `identification` group but not"
            f" verified: {difference}"
        )

    summary = nisarqa.get_summary()
    if passes:
        summary.check_identification_group(result="PASS")
    else:
        summary.check_identification_group(result="FAIL")


__all__ = nisarqa.get_all(__name__, objects_to_skip)
