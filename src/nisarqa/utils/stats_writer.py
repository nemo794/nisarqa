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


def percent_nan_is_within_threshold(
    percentage: float,
    threshold_percentage: float,
    arr_name: str,
) -> bool:
    """
    Check if % of NaN values is within threshold; note in log and summary csv.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is NaN.
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
    """
    nisarqa.verify_valid_percent(percentage)

    log = nisarqa.get_logger()

    msg = f"Array {arr_name} is {percentage} percent NaN pixels."

    if threshold_percentage != -1:
        nisarqa.verify_valid_percent(threshold_percentage)
        msg += f" Acceptable threshold is {threshold_percentage} percent NaN."
        summary_threshold = threshold_percentage
    else:
        summary_threshold = ""

    if (percentage > threshold_percentage) and (threshold_percentage != -1):
        log.error(msg)
        passes_metric = False
        nan_pass = "FAIL"
    else:
        log.info(msg)
        passes_metric = True
        nan_pass = "PASS"

    nisarqa.get_summary().check_nan_pixels_within_threshold(
        result=nan_pass,
        threshold=str(summary_threshold),
        actual=f"{percentage:.2f}",
        notes=arr_name,
    )

    return passes_metric


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


def percent_inf_is_within_threshold(
    percentage: float,
    threshold_percentage: float,
    arr_name: str,
) -> bool:
    """
    Check if % of +/- Inf values is within threshold; note in log.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is +/- Inf.
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
    """
    nisarqa.verify_valid_percent(percentage)

    log = nisarqa.get_logger()

    msg = f"Array {arr_name} is {percentage} percent +/- Inf pixels."

    if threshold_percentage != -1:
        nisarqa.verify_valid_percent(threshold_percentage)
        msg += (
            f" Acceptable threshold is {threshold_percentage} percent +/- Inf."
        )

    if (percentage > threshold_percentage) and (threshold_percentage != -1):
        log.error(msg)
        passes_metric = False
    else:
        log.info(msg)
        passes_metric = True

    return passes_metric


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


def percent_fill_is_within_threshold(
    percentage: float,
    threshold_percentage: float,
    fill_value: float,
    arr_name: str,
) -> bool:
    """
    Check if % of fill values is within threshold; note in log.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is fill.
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
    """
    nisarqa.verify_valid_percent(percentage)

    log = nisarqa.get_logger()

    msg = (
        f"Array {arr_name} is {percentage} percent fill value pixels."
        f" (Fill value is {fill_value}."
    )
    if threshold_percentage != -1:
        nisarqa.verify_valid_percent(threshold_percentage)
        msg += (
            f" Acceptable threshold is {threshold_percentage} percent fill"
            " value.)"
        )
    if (percentage > threshold_percentage) and (threshold_percentage != -1):
        log.error(msg)
        passes_metric = False
    else:
        log.info(msg)
        passes_metric = True

    return passes_metric


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


def percent_near_zero_is_within_threshold(
    percentage: float,
    threshold_percentage: float,
    arr_name: str,
) -> bool:
    """
    Check if % of near-zero values is within threshold; note in log.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is near-zero.
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
    """
    nisarqa.verify_valid_percent(percentage)

    log = nisarqa.get_logger()

    msg = f"Array {arr_name} is {percentage} percent near-zero value pixels."
    if threshold_percentage != -1:
        nisarqa.verify_valid_percent(threshold_percentage)
        msg += (
            f" Acceptable threshold is {threshold_percentage} percent near-zero"
            " pixels.)"
        )
    if (percentage > threshold_percentage) and (threshold_percentage != -1):
        log.error(msg)
        passes_metric = False
    else:
        log.info(msg)
        passes_metric = True

    return passes_metric


def save_percent_total_invalid_to_stats_h5(
    percentage: float, stats_h5: h5py.File, grp_path: str
) -> None:
    """
    Save the final percentage of total invalid pixels to the stats H5 file.

    Invalid pixels include NaN, Inf, fill, or near-zero valued pixels.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is fill value.
    h5_file : h5py.File
        HDF5 File handle to save this dataset to.
    grp_path : str
        Path to h5py Group to add the dataset and attributes to.
    """
    nisarqa.verify_valid_percent(percentage)

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentTotalInvalid",
        ds_data=percentage,
        ds_units="1",
        ds_description=(
            f"Percent of dataset elements that are either NaN, Inf, fill,"
            " or near-zero valued pixels."
        ),
    )


def percent_total_invalid_is_within_threshold(
    percentage: float,
    threshold_percentage: float,
    arr_name: str,
) -> bool:
    """
    Check if % of total invalid values is within threshold; note in log.

    Invalid pixels include NaN, Inf, fill, or near-zero valued pixels.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is invalid.
    threshold_percentage : float
        Percentage of a raster that is okay to be invalid.
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
    """
    nisarqa.verify_valid_percent(percentage)

    log = nisarqa.get_logger()

    msg = (
        f"Array {arr_name} is {percentage} percent invalid pixels"
        " (NaN, Inf, fill, or near-zero)."
    )
    if threshold_percentage != -1:
        nisarqa.verify_valid_percent(threshold_percentage)
        msg += (
            f" Acceptable threshold is {threshold_percentage} percent invalid"
            " pixels.)"
        )
    if (percentage > threshold_percentage) and (threshold_percentage != -1):
        log.error(msg)
        passes_metric = False
    else:
        log.info(msg)
        passes_metric = True

    # Note the metrics in the SUMMARY CSV
    nisarqa.get_summary().check_invalid_pixels_within_threshold(
        result="PASS" if passes_metric else "FAIL",
        threshold=str(threshold_percentage),
        actual=f"{percentage:.2f}",
        notes=arr_name,
    )

    return passes_metric


__all__ = nisarqa.get_all(__name__, objects_to_skip)
