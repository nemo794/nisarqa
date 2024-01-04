from __future__ import annotations

import warnings
from collections import namedtuple
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
    raster: nisarqa.Raster,
    stats_h5: h5py.File,
    threshold: float = nisarqa.STATISTICS_THRESHOLD_PERCENTAGE,
    epsilon: float = 1e-6,
    nearly_all_zeros_is_ok: bool = False,
) -> None:
    """
    Compute and save min, max, mean, std, % nan, % zero, % fill, % inf to HDF5.

    Warning: Entire input array will be read into memory and processed.
    Only use this function for small datasets.

    Parameters
    ----------
    raster : nisarqa.Raster
        Input Raster.
    stats_h5 : h5py.File
        The output file to save QA metrics to.
    threshold : float, optional
        The threshold value for alerting users to possible malformed datasets.
        If the percentage of NaN-, zero-, fill-, or Inf-valued pixels
        is above `threshold`, it will be logged as an error.
        Defaults to `nisarqa.STATISTICS_THRESHOLD_PERCENTAGE`.
    epsilon : float or int, optional
        The tolerance used for computing if raster pixels are "almost zero".
        This will be used during the check for percentage of near-zero pixels.
        Defaults to 1e-6.
    nearly_all_zeros_is_ok : bool, optional
        False to have this function issue an error if the raster contains
        more than `threshold` percentage of near-zero pixels.
        True to have this function suppress that same error. The percentage
        will still be computed, logged, and saved to the STATS.h5 file, but
        it will not be considered an error.

        For most rasters, if there are greater than `threshold` percent
        near-zero pixels, then this should be an error and an indication of
        a faulty NISAR input product. In this case, set
        `nearly_all_zeros_is_ok` to False.
        However, some raster layers (e.g. GUNW's `ionospherePhaseScreen`)
        are known to be populated with all zero values if the ionosphere phase
        module was disabled for GUNW ISCE3 processing. In this case, set
        `nearly_all_zeros_is_ok` to True.


    Notes
    -----
    If the fill value is set to None in the input *Raster, that field will
    not be computed nor included in the STATS.h5 file.
    """
    # Create a flag, to be used in the PASS/FAIL Summary CSV
    all_metrics_pass = True

    log = nisarqa.get_logger()

    arr = raster.data
    units = raster.units
    grp_path = raster.stats_h5_group_path
    fill_value = raster.fill_value
    arr_name = raster.name

    arr_size = arr.size

    # First, compute percentage of invalid (non-finite and/or "fill") pixels.

    # Compute NaN value metrics
    # (np.isnan works for both real and complex data. For complex data, if
    # either the real or imag part is NaN, then the pixel is considered NaN.)
    num_nan = np.sum(np.isnan(arr))
    percent_nan = 100 * num_nan / arr_size
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentNan",
        ds_data=percent_nan,
        ds_units="1",
        ds_description="Percent of dataset elements with a NaN value.",
    )
    if np.isnan(fill_value):
        # If the fill value is NaN, then check the threshold.
        msg = (
            f"Array {arr_name} is {percent_nan} percent NaN pixels. (Acceptable"
            f" threshold is {threshold} percent NaN.)"
        )
        if percent_nan >= threshold:
            log.error(msg)
            all_metrics_pass = False
        else:
            log.info(msg)

    else:
        # If the fill value is not NaN, then
        # the raster should not contain any NaN values.
        # Note: the only known case where the fill value is intentionally
        # set to None (meaning, it does not exist) is for RSLC backscatter
        # datasets. In this case, there should not be any NaN pixels.
        msg = (
            f"Array {arr_name} contains {num_nan} NaN pixels"
            f" ({percent_nan} percent NaN), but it has a fill value of"
            f" {fill_value}, so it should contain no NaN pixels."
        )
        if num_nan > 0:
            log.error(msg)
            all_metrics_pass = False
        else:
            log.info(msg)

    # Compute +/- inf metrics.
    # (np.isinf works for both real and complex data. For complex data, if
    # either real or imag part is +/- inf, then the pixel is considered inf.)
    num_inf = np.sum(np.isinf(arr))
    percent_inf = 100 * num_inf / arr_size
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentInf",
        ds_data=percent_inf,
        ds_units="1",
        ds_description="Percent of dataset elements with a +/- inf value.",
    )
    msg = (
        f"Array {arr_name} is {percent_inf} percent +/- infinity pixels. "
        f" (Acceptable threshold is {threshold} percent inf.)"
    )
    if percent_inf >= threshold:
        log.error(msg)
        all_metrics_pass = False
    else:
        log.info(msg)

    if fill_value is not None:
        # Compute fill value metrics. (If the fill value is NaN, it's ok that
        # this is redundant to the NaN value metrics.)
        if np.isnan(fill_value):
            # `np.nan == np.nan` evaluates to False, so handle this case here
            num_fill = num_nan
        else:
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
            f"Array {arr_name} is {percent_fill} percent fill value pixels."
            f" (Acceptable threshold is {threshold} percent fill value.)"
        )
        if percent_fill >= threshold:
            log.error(msg)
            all_metrics_pass = False
        else:
            log.info(msg)

    # Compute number of zeros metrics.
    # By using np.abs(), for complex values this will compute the magnitude.
    # The magnitude will only be ~0.0 if both real and imaj are ~0.0.
    num_zero = np.sum(np.abs(arr) < epsilon)
    percent_zero = 100 * num_zero / arr.size
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentNearZero",
        ds_data=percent_zero,
        ds_units="1",
        ds_description=(
            f"Percent of dataset elements that are within {epsilon} of zero."
        ),
    )

    msg = (
        f"Array {arr_name} is {percent_zero} percent near-zero pixels."
        f" (Acceptable threshold is {threshold} percent zeros.)"
    )
    if (percent_zero >= threshold) and not nearly_all_zeros_is_ok:
        log.error(msg)
        all_metrics_pass = False
    else:
        log.info(msg)

    # Note the metrics in the SUMMARY CSV
    summary = (
        f"(%s) PASS/FAIL: All metrics within threshold for raster {arr_name}."
    )
    if all_metrics_pass:
        log.info(summary % "PASS")
    else:
        log.error(summary % "FAIL")

    def _compute_min_max_mean_std(arr: np.ndarray, data_type: str) -> None:
        """
        Compute min, max, mean, and samples standard deviation; save to HDF5.

        Parameters
        ----------
        arr : numpy.ndarray
            The input array to have statistics run on. Note: If `arr` has a
            complex dtype, this function will need to access the `arr.real`
            and `arr.imag` parts separately. Unfortunately, h5py.Dataset
            instances do not allow access to these parts using that syntax,
            so to be safe, please pass in a Numpy array.
        data_type : str
            The type data being passed in.
            One of: "float", "real_comp", or "imag_comp".

        Notes
        -----
        TODO: This is a clunky, kludgy function. When these statistics get
        implemented for RSLC, GSLC, and GCOV after R4, the developer
        should consider pulling this function out into a standalone function.
        For expediency, for R4, all InSAR products will use this function,
        so this information can live here.
        """

        # Step 1: Create a dict to hold common naming conventions:

        # Per ISCE3 R4 conventions, for floating-point datasets, use:
        # min_value
        # mean_value
        # max_value
        # sample_stddev

        # For complex-valued dataset, use:
        # min_real_value
        # mean_real_value
        # max_real_value
        # sample_stddev_real
        # min_imag_value
        # mean_imag_value
        # max_imag_value
        # sample_stddev_imag
        Stat = namedtuple("Stat", "name descr")

        my_dict = {
            "float": {
                "min": Stat(
                    "min_value", "Minimum value of the numeric data points"
                ),
                "max": Stat(
                    "max_value", "Maximum value of the numeric data points"
                ),
                "mean": Stat(
                    "mean_value",
                    "Arithmetic average of the numeric data points",
                ),
                "std": Stat(
                    "sample_stddev",
                    "Sample standard deviation of the numeric data points",
                ),
            },
            "real_comp": {
                "min": Stat(
                    "min_real_value",
                    "Minimum value of the real component of the numeric data"
                    " points",
                ),
                "max": Stat(
                    "max_real_value",
                    "Maximum value of the real component of the numeric data"
                    " points",
                ),
                "mean": Stat(
                    "mean_real_value",
                    "Arithmetic average of the real component of the numeric"
                    " data points",
                ),
                "std": Stat(
                    "sample_stddev_real",
                    "Sample standard deviation of the real component of the"
                    " numeric data points",
                ),
            },
            "imag_comp": {
                "min": Stat(
                    "min_imag_value",
                    "Minimum value of the imaginary component of the numeric"
                    " data points",
                ),
                "max": Stat(
                    "max_imag_value",
                    "Maximum value of the imaginary component of the numeric"
                    " data points",
                ),
                "mean": Stat(
                    "mean_imag_value",
                    "Arithmetic average of the imaginary component of the"
                    " numeric data points",
                ),
                "std": Stat(
                    "sample_stddev_imag",
                    "Sample standard deviation of the imaginary component of"
                    " the numeric data points",
                ),
            },
        }

        try:
            my_dict = my_dict[data_type]
        except KeyError:
            raise ValueError(f"{data_type=}, must be one of {my_dict.keys()}.")

        # Fill all invalid pixels in the array with NaN, to easily compute metrics
        arr_copy = np.where(
            (np.isfinite(arr) & (arr != fill_value)), arr, np.nan
        )

        # Compute min/max/mean/std of valid pixels
        for key, func in [
            ("min", np.nanmin),
            ("max", np.nanmax),
            ("mean", np.nanmean),
            ("std", lambda x: np.nanstd(x, ddof=1)),
        ]:
            nisarqa.create_dataset_in_h5group(
                h5_file=stats_h5,
                grp_path=grp_path,
                ds_name=my_dict[key].name,
                ds_data=func(arr_copy),
                ds_units=units,
                ds_description=my_dict[key].descr,
            )

    if np.issubdtype(arr, np.complexfloating):
        # HDF5 Datasets cannot access .real nor .imag, so we need
        # to read the array into a numpy array in memory first.
        _compute_min_max_mean_std(arr[()].real, "real_comp")
        _compute_min_max_mean_std(arr[()].imag, "imag_comp")
    else:
        _compute_min_max_mean_std(arr[()], "float")


__all__ = nisarqa.get_all(__name__, objects_to_skip)
