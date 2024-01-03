from __future__ import annotations

import warnings
from collections.abc import Sequence

import h5py
import numpy as np
from numpy.typing import ArrayLike

import nisarqa

objects_to_skip = nisarqa.get_all(__name__)


def arr2pow(arr):
    """
    Compute power in linear units of the input array.

    Power is computed as magnitude squared.

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    power : numpy.ndarray
        Power of input array, with the same shape as the input.
    """

    power = np.abs(arr) ** 2

    return power


def pow2db(power):
    """
    Convert a power quantity from linear units to decibels.

    Parameters
    ----------
    power : array_like
        Input in linear units.

    Returns
    -------
    power_db : numpy scalar or numpy.ndarray
        Output in decibels, with the same shape as the input.
    """
    return 10.0 * np.log10(power)


def nearest_odd_int(k):
    """Compute the nearest odd integer to `k`"""
    result = int(np.floor(k))
    if result % 2 == 0:
        result = result + 1

    # Sanity Check
    assert result % 2 == 1, print("the result should be an odd value.")
    assert isinstance(result, int), print("the result should be an integer.")

    return result


def next_greater_odd_int(k):
    """Compute the next odd integer greater than or equal to `k`"""
    # Find the next-largest even or odd integer
    if abs(k - np.rint(k)) < 0.0001:
        # Accommodate floating point error
        result = int(np.rint(k))
    else:
        # Round up to nearest integer
        result = int(np.ceil(k))

    if result % 2 == 0:
        result = result + 1

    # Sanity Check
    assert result % 2 == 1, "the result should be an odd value."
    assert isinstance(result, int), "the result should be an integer."

    return result


def counts2density(counts, bins):
    """
    Compute the probability density for the given counts and bins.

    This function implements numpy.histogram's 'density' parameter.
    Each bin will display the bin's raw count divided by the
    total number of counts and the bin width
    (density = counts / (sum(counts) * np.diff(bins))),
    so that the area under the histogram integrates to 1
    (np.sum(density * np.diff(bins)) == 1).

    Parameters
    ----------
    counts : array_like
        The values of the histogram bins, such as returned from np.histogram.
        This is an array of length (len(bins) - 1).
    bins : array_like
        The edges of the bins. Length is the number of bins + 1,
        i.e. len(counts) + 1.

    Returns
    -------
    density : numpy.ndarray
        Each bin will contain that bin's density (as described above),
        so that the area under the histogram integrates to 1.
    """

    # Formula per numpy.histogram's documentation:
    density = counts / (np.sum(counts) * np.diff(bins))

    # Sanity check
    actual = np.sum(density * np.diff(bins))
    assert np.abs(actual - 1) < 1e-6

    return density


def normalize(
    arr: ArrayLike, min_max: Sequence[float] | None = None
) -> np.ndarray:
    """
    Normalize input array to range [0,1], ignoring any NaN values.

    Parameters
    ----------
    arr : Arraylike
        Input array to be normalized
    min_max : pair of numeric or None, optional
        Defaults to None, which means that `arr`'s min and max will be
        computed and used for normalization. (This is most common.)
        If provided, the normalization computation will use `min_max` as
        the range which determines the scaling to [0,1].
        Format: [<minimum>, <maximum>]

    Returns
    -------
    normalized_arr : numpy.ndarray
        A normalized copy of `arr`.
    """
    arr = np.asanyarray(arr)

    if min_max is None:
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
    else:
        if min_max[0] >= min_max[1]:
            raise ValueError(
                f"{min_max=}, but min_max[0] must be less than min_max[1]"
            )
        arr_min = min_max[0]
        arr_max = min_max[1]

    return (arr - arr_min) / (arr_max - arr_min)


def compute_fft(slc_arr: ArrayLike, axis: int = 1) -> np.ndarray:
    """
    Compute FFT of the input array along the given axis.

    Non-finite values in `slc_arr` will be filled with zero.
    No normalization will be applied.

    Parameters
    ----------
    slc_arr : array_like
        Input SLC array
    axis : int, optional
        Axis along which to take the FFT. Default is axis = 1.
        (For NISAR, axis 1 is the range axis, and axis 0 is the azimuth axis.)

    Returns
    -------
    fft : numpy.ndarray
        The FFT of `slc_arr` along `axis`. Non-finite values filled with zero.

    Notes
    -----
    Similar to ISCE3, non-finite values in `slc_arr` will be filled with zeros.
    An alternative would be to implement non-equispace FFT's, but that could be
    too computationally and algorithmically complicated for the QA Code.
    """

    finite_slc_arr = np.where(np.isfinite(slc_arr), slc_arr, 0.0)

    fft = np.fft.fft(finite_slc_arr, axis=axis)

    return fft


def generate_fft_freqs(
    num_samples: int, sampling_rate: float, fft_shift: bool = True
) -> np.ndarray:
    """
    Return the Discrete Fourier Transform sample frequencies.

    Parameters
    ----------
    num_samples : int
        Window length.
    sampling_rate : numeric
        Sample rate (inverse of the sample spacing).
    fft_shift : bool, optional
        True to have the frequencies in `fft_freqs` be continuous from
        negative (min) -> positive (max) values.

        False to leave `fft_freqs` as the output from `numpy.fft.fftfreq()`,
        where this discrete FFT operation orders values
        from 0 -> max positive -> min negative -> 0- . (This creates
        a discontinuity in the interval's values.)

        Defaults to True.

    Returns
    -------
    fft_freqs : numpy.ndarray
        Array of length `num_samples` containing the sample frequencies
        (frequency bin centers) in the same units as `sampling_rate`.
        For instance, if the sampling rate is in Hz, then the frequency
        unit is also cycles/second.
    """
    fft_freqs = np.fft.fftfreq(num_samples, 1.0 / sampling_rate)

    if fft_shift:
        # Shift fft_freqs to be continuous from
        # negative (min) -> positive (max) values.
        # (The output from the discrete FFT operation orders values
        # from 0 -> max positive -> min negative -> 0- .
        # Doing this will remove that discontinuity.)
        fft_freqs = np.fft.fftshift(fft_freqs)

    return fft_freqs


def hz2mhz(arr: np.ndarray) -> np.ndarray:
    """Convert input array from Hz to MHz."""
    return arr * 1.0e-06


def compute_and_save_basic_statistics(
    arr: ArrayLike,
    is_geocoded: bool,
    units: str,
    grp_path: str,
    stats_h5: h5py.File,
    fill_value=np.nan,
    arr_name: str = "",
) -> None:
    """
    Compute and save min, max, mean, std, % nan, and % zero statistics to HDF5.

    Warning: Entire input array will be read into memory. Only use this
    function for small datasets.

    Parameters
    ----------
    arr : ArrayLike
        Input array.
    is_geocoded : bool
        Set to `True` if `arr` is a geocoded product, otherwise False.
        This flag will be used to set the thresholds for alerting users if the
        percentage of NaN or zero valued pixels to above a certain threshold.
        If False, this threshold will be set to 25%; images in range Doppler
        products should be mostly numeric.
        If True, this threshold will be set to 95.0%; images in geocoded
        products have fill values in non-imagery areas, which are likely
        to make up a significant portion of the raster.
    units : str
        The units of the input array. If `data` is numeric but unitless
        (e.g ratios), by NISAR convention please use the string "1".
    grp_path : str
        Path to h5py Group to add the computed statistics to.
    stats_h5 : h5py.File
        The output file to save QA metrics to.
    fill_value : float, optional
        The fill value for `arr`. Defaults to NaN.
    arr_name : string, optional
        A human-readable name for `arr`. (To be used in log messages.)
    """
    log = nisarqa.get_logger()
    arr_size = arr.size
    threshhold = 95.0 if is_geocoded else 25.0

    # First, compute percentage of invalid pixels. Afterwards, we'll fill
    # all of these invalid pixels with NaN to compute min/max/mean/std.

    # Compute NaN value metrics
    num_nan = np.sum(np.isnan(arr))
    percent_nan = 100 * num_nan / arr_size
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentNan",
        ds_data=percent_nan,
        ds_units="1",
        ds_description="Percent of dataset elements with NaN value.",
    )
    if np.isnan(fill_value):
        msg = (
            f"(%s) PASS/FAIL: Array {arr_name} is {percent_nan} percent NaN"
            " pixels, which is greater than the threshold of"
            f" {threshhold} percent NaN."
        )
        if percent_nan >= threshhold:
            log.error(msg % "FAIL")
        else:
            log.info(msg % "PASS")

    else:
        # Fill Value is not NaN. If the fill value is not NaN, then the
        # raster should not contain NaN values.
        msg = (
            f"(%s) PASS/FAIL: Array {arr_name} has a fill value of"
            f" {fill_value}, so it should contain no NaN pixels. (It contains"
            f" {num_nan} NaN pixels.)"
        )
        if num_nan > 0:
            log.error(msg % "FAIL")
        else:
            log.info(msg % "PASS")

    # Compute non-finite elements metrics.
    # (This is counts +/- inf elements, excluding NaN values.)
    num_inf = np.sum(np.isinf(arr))
    percent_inf = 100 * num_inf / arr_size
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentInfinite",
        ds_data=percent_inf,
        ds_units="1",
        ds_description="Percent of dataset elements with +/- inf value.",
    )
    msg = (
        f"(%s) PASS/FAIL: Array {arr_name} is {percent_nan} percent +/-"
        " infinity pixels, which is greater than the threshold of"
        f" {threshhold} percent inf."
    )
    if percent_inf >= threshhold:
        log.error(msg % "FAIL")
    else:
        log.info(msg % "PASS")

    # Compute fill value metrics. (It's ok if this is redundant to the
    # NaN value metrics. The more info, the better!)
    num_fill = np.sum(arr == fill_value)
    percent_fill = 100 * num_fill / arr_size
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentFill",
        ds_data=percent_fill,
        ds_units="1",
        ds_description=(
            "Percent of dataset elements containing the fill value, which"
            f" is: {fill_value}."
        ),
    )
    msg = (
        f"(%s) PASS/FAIL: Array {arr_name} is {percent_fill} percent fill"
        " value pixels, which is greater than the threshold of"
        f" {threshhold} percent fill value."
    )
    if percent_fill >= threshhold:
        log.error(msg % "FAIL")
    else:
        log.info(msg % "PASS")

    # Compute number of zeros metrics.
    num_zero = np.sum(np.abs(arr) < 1e-6)
    percent_zero = 100 * num_zero / arr.size
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentZero",
        ds_data=percent_zero,
        ds_units="1",
        ds_description=(
            "Percent of dataset elements that are within 1e-6 of zero."
        ),
    )

    msg = (
        f"(%s) PASS/FAIL: Array {arr_name} is {percent_zero} percent zero"
        f" pixels, which is greater than the threshold of {threshhold} percent"
        " zeros."
    )
    if percent_zero >= threshhold:
        log.error(msg % "FAIL")
    else:
        log.info(msg % "PASS")

    # Fill all invalid pixels in the array with NaN, to easily compute metrics
    arr_copy = np.where((np.isfinite(arr) & arr != fill_value), arr, np.nan)

    # Compute min/max/mean/std of valid pixels
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="min_value",
        ds_data=np.nanmin(arr_copy),
        ds_units=units,
        ds_description="Minimum value of the numeric data points",
    )

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="max_value",
        ds_data=np.nanmax(arr_copy),
        ds_units=units,
        ds_description="Maximum value of the numeric data points",
    ),

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="mean_value",
        ds_data=np.nanmean(arr_copy),
        ds_units=units,
        ds_description="Arithmetic average of the numeric data points",
    )

    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="sample_standard_deviation",
        ds_data=np.nanstd(arr_copy),
        ds_units=units,
        ds_description="Sample standard deviation of the numeric data points",
    )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
