from __future__ import annotations

import h5py

import nisarqa

objects_to_skip = nisarqa.get_all(__name__)


def save_percent_nan_to_stats_h5(
    percentage: float, stats_h5: h5py.File, grp_path: str
) -> None:
    """
    Save the final percentage of NaN pixels to the stats H5 file.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is NaN.
    h5_file : h5py.File
        HDF5 File handle to save this dataset to.
    grp_path : str
        Path to h5py Group to add the dataset and attributes to.
    """
    nisarqa.verify_valid_percent(percentage)

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentNan",
        ds_data=percentage,
        ds_units="1",
        ds_description="Percent of dataset elements with a NaN value.",
    )


def save_percent_inf_to_stats_h5(
    percentage: float, stats_h5: h5py.File, grp_path: str
) -> None:
    """
    Save the final percentage of +/- Inf pixels to the stats H5 file.

    Hint: When computing the percentage, np.isinf() works for both
    real and complex data. For complex data, if either real or imag
    part is +/- Inf, then the pixel is considered inf.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is +/- Inf.
    h5_file : h5py.File
        HDF5 File handle to save this dataset to.
    grp_path : str
        Path to h5py Group to add the dataset and attributes to.
    """
    nisarqa.verify_valid_percent(percentage)

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentInf",
        ds_data=percentage,
        ds_units="1",
        ds_description="Percent of dataset elements with a +/- inf value.",
    )


def save_percent_fill_to_stats_h5(
    percentage: float, fill_value: float, stats_h5: h5py.File, grp_path: str
) -> None:
    """
    Save the final percentage of fill pixels to the stats H5 file.

    Note: By QA convention, if the fill value is NaN, it's ok that
    this check is redundant to the % NaN value metrics.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is fill value.
    fill_value : float
        The fill value for the raster. (Will be used for description.)
    h5_file : h5py.File
        HDF5 File handle to save this dataset to.
    grp_path : str
        Path to h5py Group to add the dataset and attributes to.
    """
    nisarqa.verify_valid_percent(percentage)

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentFill",
        ds_data=percentage,
        ds_units="1",
        ds_description=(
            "Percent of dataset elements containing the fill value, which"
            f" is: {fill_value}."
        ),
    )


def save_percent_near_zero_to_stats_h5(
    percentage: float, epsilon: float, stats_h5: h5py.File, grp_path: str
) -> None:
    """
    Save the final percentage of near-zero pixels to the stats H5 file.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is fill value.
    epsilon : float or int
        Absolute tolerance that was used for determining if a raster pixel
        was 'almost zero'.
    h5_file : h5py.File
        HDF5 File handle to save this dataset to.
    grp_path : str
        Path to h5py Group to add the dataset and attributes to.
    """
    nisarqa.verify_valid_percent(percentage)

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentNearZero",
        ds_data=percentage,
        ds_units="1",
        ds_description=(
            f"Percent of dataset elements that are within {epsilon} of zero."
        ),
    )


def save_percent_total_invalid_to_stats_h5(
    percentage: float, stats_h5: h5py.File, grp_path: str, zero_is_invalid: bool
) -> None:
    """
    Save the final percentage of total invalid pixels to the stats H5 file.

    Invalid pixels include NaN, Inf, fill, or (optionally) near-zero valued
    pixels.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is invalid pixels.
    h5_file : h5py.File
        HDF5 File handle to save this dataset to.
    grp_path : str
        Path to h5py Group to add the dataset and attributes to.
    zero_is_invalid : bool
        True if near-zero pixels were included in the
        total number of invalid pixels. False if they were excluded.
        (This will impact the description for the Dataset.)
    """
    nisarqa.verify_valid_percent(percentage)

    if zero_is_invalid:
        msg = (
            f"Percent of dataset elements that are either NaN, Inf, fill,"
            " or near-zero valued pixels."
        )
    else:
        msg = (
            f"Percent of dataset elements that are either NaN, Inf, or fill"
            " valued pixels. (Near-zero valued pixels are not included.)"
        )

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentTotalInvalid",
        ds_data=percentage,
        ds_units="1",
        ds_description=msg,
    )


def _percent_value_is_within_threshold(
    name_of_value: str,
    count: int,
    arr_size: int,
    threshold_percentage: float,
    arr_name: str,
) -> tuple[bool, float]:
    """
    Private function: Check if a percentage is within threshold; note in log.

    The percentage of the array that is considered to be `name_of_value`
    is computed by `count / arr_size`.

    Parameters
    ----------
    name_of_value : str
        Name of the metric being checked, e.g. "NaN" or "+/- Inf". This will
        be used for the log messages.
    count : int
        Number of pixels that are `name_of_value`.
    arr_size : int
        Total size of array.
    threshold_percentage : float
        Percentage of a raster that is okay to be `name_of_value`.
        If `percentage` is greater than `threshold_percentage`,
        it will be logged as an error. A threshold value of -1 indicates
        to always log as info (not as an error), and `passes_metric` will
        always return `True`.
    arr_name : str
        Name of the array; will be used in log messages and similar.

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    percentage : float
        Percentage of a raster that is `name_of_value`.
    """
    log = nisarqa.get_logger()

    percentage = count / arr_size * 100

    nisarqa.verify_valid_percent(percentage)

    msg = f"Array {arr_name} contains {count}/{arr_size} ({percentage}%) {name_of_value} pixels."

    if threshold_percentage != -1:
        nisarqa.verify_valid_percent(threshold_percentage)
        msg += f" Acceptable threshold is {threshold_percentage} percent {name_of_value}"

    if (percentage > threshold_percentage) and (threshold_percentage != -1):
        log.error(msg)
        passes_metric = False
    else:
        log.info(msg)
        passes_metric = True

    return passes_metric, percentage


def percent_nan_is_within_threshold(
    count: int,
    arr_size: int,
    threshold_percentage: float,
    arr_name: str,
) -> tuple[bool, float]:
    """
    Check if % of NaN values is within threshold; note in log and summary csv.

    Parameters
    ----------
    count : int
        Number of pixels that are `name_of_value`.
    arr_size : int
        Total size of array.
    threshold_percentage : float
        Percentage of a raster that is okay to be NaN.
        If `percentage` is greater than `threshold_percentage`,
        it will be logged as an error. A threshold value of -1 indicates
        to always log as info (not as an error), and `passes_metric` will
        always return `True`.
    arr_name : str
        Name of the array; will be used in log messages and similar.

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    percentage : float
        Percentage of a raster that is NaN.
    """
    passes_metric, percentage = _percent_value_is_within_threshold(
        name_of_value="NaN",
        count=count,
        arr_size=arr_size,
        threshold_percentage=threshold_percentage,
        arr_name=arr_name,
    )

    if threshold_percentage != -1:
        summary_threshold = str(threshold_percentage)
    else:
        summary_threshold = ""

    nisarqa.get_summary().check_nan_pixels_within_threshold(
        result="PASS" if passes_metric else "FAIL",
        threshold=summary_threshold,
        actual=f"{percentage:.2f}",
        notes=arr_name,
    )

    return passes_metric, percentage


def percent_inf_is_within_threshold(
    count: int,
    arr_size: int,
    threshold_percentage: float,
    arr_name: str,
) -> tuple[bool, float]:
    """
    Check if % of +/- Inf values is within threshold; note in log.

    Parameters
    ----------
    count : int
        Number of pixels that are `name_of_value`.
    arr_size : int
        Total size of array.
    threshold_percentage : float
        Percentage of a raster that is okay to be +/- Inf.
        If `percentage` is greater than `threshold_percentage`,
        it will be logged as an error. A threshold value of -1 indicates
        to always log as info (not as an error), and `passes_metric` will
        always return `True`.
    arr_name : str
        Name of the array; will be used in log messages and similar.

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    percentage : float
        Percentage of a raster that is +/- Inf.
    """
    return _percent_value_is_within_threshold(
        name_of_value="+/- Inf",
        count=count,
        arr_size=arr_size,
        threshold_percentage=threshold_percentage,
        arr_name=arr_name,
    )


def percent_fill_is_within_threshold(
    count: int,
    arr_size: int,
    threshold_percentage: float,
    fill_value: float,
    arr_name: str,
) -> tuple[bool, float]:
    """
    Check if % of fill values is within threshold; note in log.

    Parameters
    ----------
    count : int
        Number of pixels that are `name_of_value`.
    arr_size : int
        Total size of array.
    threshold_percentage : float
        Percentage of a raster that is okay to be fill.
        If `percentage` is greater than `threshold_percentage`,
        it will be logged as an error. A threshold value of -1 indicates
        to always log as info (not as an error), and `passes_metric` will
        always return `True`.
    fill_value : float
        The fill value for the raster. (Will be used for description.)
    arr_name : str
        Name of the array; will be used in log messages and similar.

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    percentage : float
        Percentage of a raster that is fill.
    """

    return _percent_value_is_within_threshold(
        name_of_value=f"fill value (fill value is {fill_value})",
        count=count,
        arr_size=arr_size,
        threshold_percentage=threshold_percentage,
        arr_name=arr_name,
    )


def percent_near_zero_is_within_threshold(
    count: int,
    arr_size: int,
    threshold_percentage: float,
    arr_name: str,
) -> tuple[bool, float]:
    """
    Check if % of near-zero values is within threshold; note in log.

    Parameters
    ----------
    count : int
        Number of pixels that are `name_of_value`.
    arr_size : int
        Total size of array.
    threshold_percentage : float
        Percentage of a raster that is okay to be near-zero.
        If `percentage` is greater than `threshold_percentage`,
        it will be logged as an error. A threshold value of -1 indicates
        to always log as info (not as an error), and `passes_metric` will
        always return `True`.
    arr_name : str
        Name of the array; will be used in log messages and similar.

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    percentage : float
        Percentage of a raster that is near-zero.
    """

    return _percent_value_is_within_threshold(
        name_of_value="near-zero",
        count=count,
        arr_size=arr_size,
        threshold_percentage=threshold_percentage,
        arr_name=arr_name,
    )


def percent_total_invalid_is_within_threshold(
    count: int,
    arr_size: int,
    threshold_percentage: float,
    arr_name: str,
    zero_is_invalid: bool,
) -> tuple[bool, float]:
    """
    Check if % of total invalid values is within threshold; note in log.

    Invalid pixels include NaN, Inf, fill, or (optionally) near-zero valued
    pixels.

    Parameters
    ----------
    count : int
        Number of pixels that are `name_of_value`.
    arr_size : int
        Total size of array.
    threshold_percentage : float
        Percentage of a raster that is okay to be invalid.
        If `percentage` is greater than `threshold_percentage`,
        it will be logged as an error. A threshold value of -1 indicates
        to always log as info (not as an error), and `passes_metric` will
        always return `True`.
    arr_name : str
        Name of the array; will be used in log messages and similar.
    zero_is_invalid : bool
        True if near-zero pixels were included in the
        total number of invalid pixels. False if they were excluded.
        (This will impact the description for the Dataset.)

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    percentage : float
        Percentage of a raster that is invalid.
    """

    if zero_is_invalid:
        msg = "NaN, Inf, fill, or near-zero"
    else:
        msg = "NaN, Inf, or fill-valued"

    passes_metric, percentage = _percent_value_is_within_threshold(
        name_of_value=f"invalid ({msg})",
        count=count,
        arr_size=arr_size,
        threshold_percentage=threshold_percentage,
        arr_name=arr_name,
    )

    # Note the metrics in the SUMMARY CSV
    if threshold_percentage is -1:
        summary_threshold = ""
    else:
        summary_threshold = str(threshold_percentage)

    nisarqa.get_summary().check_invalid_pixels_within_threshold(
        result="PASS" if passes_metric else "FAIL",
        threshold=summary_threshold,
        actual=f"{percentage:.2f}",
        notes=arr_name,
    )

    return passes_metric, percentage


__all__ = nisarqa.get_all(__name__, objects_to_skip)
