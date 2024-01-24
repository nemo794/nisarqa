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
    nisarqa.verify_valid_percentage(percentage)

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
    nan_is_valid_value: bool,
    number_of_nan: int = None,
) -> bool:
    """
    Check if % of NaN values is within threshold; note in log and summary csv.

    Parameters
    ----------
    percentage : float
        Percentage of a raster that is NaN.
    threshold_percentage : float
        Percentage of a raster that is okay to be NaN.
        (If `percentage` is greater than or equal to `threshold_percentage`,
        this is considered an error.)
    arr_name : str
        Name of the array; will be used in log messages and similar.
    nan_is_valid_value : bool
        True if NaN is an expected, valid value in the raster; for example,
        if an array's fill value is NaN, then this should be set to True.
        False if NaN is not expected in the array. For example, RSLC backscatter
        arrays are expected to be have zeros as fill, and connected components
        layers are expected to have 255 as fill; if even a single NaN appears
        in these rasters, then there was an error in processing.
    number_of_nan : int, optional
        The exact number of nan that appeared in the raster. Only required if
        `nan_is_valid_value` is True. Otherwise, will be ignored.

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    """
    nisarqa.verify_valid_percentage(percentage)
    nisarqa.verify_valid_percentage(threshold_percentage)

    if nan_is_valid_value and ((number_of_nan is None) or (number_of_nan < 0)):
        raise ValueError(
            f"{nan_is_valid_value=}, so `number_of_nan` must be a non-negative"
            f" integer, but it is `{nan_is_valid_value}`."
        )

    log = nisarqa.get_logger()

    passes_metric = True

    if nan_is_valid_value:
        # Check the percentage against the threshold.
        msg = (
            f"Array {arr_name} is {percentage} percent NaN pixels. (Acceptable"
            f" threshold is {threshold_percentage} percent NaN.)"
        )
        if percentage >= threshold_percentage:
            log.error(msg)
            passes_metric = False
            nan_pass = "FAIL"
        else:
            log.info(msg)
            nan_pass = "PASS"
        nan_threshold = str(threshold_percentage)

    else:
        msg = (
            f"Array {arr_name} contains {number_of_nan} NaN pixels"
            f" ({percentage} percent NaN). It should contain no NaN pixels."
        )
        if number_of_nan > 0:
            log.error(msg)
            passes_metric = False
            nan_pass = "FAIL"
        else:
            log.info(msg)
            nan_pass = "PASS"
        nan_threshold = "0"

    nisarqa.get_summary().check_nan_pixels_within_threshold(
        result=nan_pass,
        threshold=nan_threshold,
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
    nisarqa.verify_valid_percentage(percentage)

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
        (If `percentage` is greater than or equal to `threshold_percentage`,
        this is considered an error.)
    arr_name : str
        Name of the array; will be used in log messages and similar.

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    """
    nisarqa.verify_valid_percentage(percentage)
    nisarqa.verify_valid_percentage(threshold_percentage)

    log = nisarqa.get_logger()

    passes_metric = True

    msg = (
        f"Array {arr_name} is {percentage} percent +/- infinity pixels. "
        f" (Acceptable threshold is {threshold_percentage} percent inf.)"
    )
    if percentage >= threshold_percentage:
        log.error(msg)
        passes_metric = False
    else:
        log.info(msg)

    # TODO - why did we not include this in the summary.csv file??

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
    nisarqa.verify_valid_percentage(percentage)

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
        (If `percentage` is greater than or equal to `threshold_percentage`,
        this is considered an error.)
    fill_value : float
        The fill value for the raster. (Will be used for description.)
    arr_name : str
        Name of the array; will be used in log messages and similar.

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    """
    nisarqa.verify_valid_percentage(percentage)
    nisarqa.verify_valid_percentage(threshold_percentage)

    log = nisarqa.get_logger()

    passes_metric = True

    msg = (
        f"Array {arr_name} is {percentage} percent fill value pixels."
        f" (Fill value is {fill_value}. Acceptable threshold is"
        f" {threshold_percentage} percent fill value.)"
    )
    if percentage >= threshold_percentage:
        log.error(msg)
        passes_metric = False
    else:
        log.info(msg)

    # TODO - why did we not include this in the summary.csv file??

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
        The tolerance used for computing if raster pixels are "almost zero".
    h5_file : h5py.File
        HDF5 File handle to save this dataset to.
    grp_path : str
        Path to h5py Group to add the dataset and attributes to.
    """
    nisarqa.verify_valid_percentage(percentage)

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
    treat_all_zeros_as_error: bool,
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
        (If `percentage` is greater than or equal to `threshold_percentage`,
        this is considered an error.)
    treat_all_zeros_as_error : bool
        True to have this function issue an error if the raster contains
        more than `threshold_percentage` percentage of near-zero pixels.
        False to have this function simply log that info as INFO.

        For some rasters, if there are greater than `threshold_percentage`
        near-zero pixels, then this should be an error and an indication of
        a faulty NISAR input product. In this case, set
        `treat_all_zeros_as_error` to True.
        However, some raster layers (e.g. GUNW's `ionospherePhaseScreen`)
        are known to be populated with all zero values if the ionosphere phase
        module was disabled for GUNW ISCE3 processing. In this case, set
        `treat_all_zeros_as_error` to False.
    arr_name : str
        Name of the array; will be used in log messages and similar.

    Returns
    -------
    passes_metric : bool
        True if the percentage is within acceptable limits. False if not.
    """
    nisarqa.verify_valid_percentage(percentage)
    nisarqa.verify_valid_percentage(threshold_percentage)

    log = nisarqa.get_logger()

    passes_metric = True

    msg = (
        f"Array {arr_name} is {percentage} percent near-zero pixels."
        f" (Acceptable threshold is {threshold_percentage} percent zeros.)"
    )
    if (percentage >= threshold_percentage) and treat_all_zeros_as_error:
        log.error(msg)
        passes_metric = False
    else:
        log.info(msg)

    # TODO - why did we not include this in the summary.csv file??

    return passes_metric


__all__ = nisarqa.get_all(__name__, objects_to_skip)
