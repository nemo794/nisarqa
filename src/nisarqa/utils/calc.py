from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence

import h5py
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
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


def amp2db(amp: ArrayLike) -> np.ndarray:
    """
    Convert a root-power quantity from linear units to decibels.

    Parameters
    ----------
    amp : array_like
        Input in linear units.

    Returns
    -------
    db : numpy scalar or numpy.ndarray
        Output in decibels, with the same shape as the input.
    """
    return 20.0 * np.log10(amp)


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
    arr : array_like
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
    params: nisarqa.ThresholdParamGroup,
) -> None:
    """
    Compute and save min, max, mean, std, % nan, % zero, % fill, % inf to HDF5.

    Parameters
    ----------
    raster : nisarqa.Raster
        Input Raster.
    stats_h5 : h5py.File
        The output file to save QA metrics to.
    params : nisarqa.ThresholdParamGroup
        A structure containing the parameters for checking the percentage
        of invalid pixels in a raster.

    Warnings
    --------
    The entire input array will be read into memory and processed.
    Only use this function for small datasets.

    Notes
    -----
    If the fill value is set to None in the input *Raster, that field will
    not be computed nor included in the STATS.h5 file.
    """
    arr = raster.data
    units = raster.units
    grp_path = raster.stats_h5_group_path
    fill_value = raster.fill_value

    # Step 1: Compute min/max/mean/STD
    # TODO: refactor aka redesign this code chunk.

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

    # Step 2: Compute % NaN, % Inf, % Fill, % near-zero, % invalid
    compute_percentage_metrics(
        raster=raster,
        stats_h5=stats_h5,
        params=params,
    )


def compute_nan_count(arr: ArrayLike) -> int:
    """
    Get the number of NaN elements in the input array.

    Parameters
    ----------
    arr : array_like
        Input array; can have a real or complex dtype.
        (For complex data, if either the real or imag part is NaN,
        then the element is considered NaN.)

    Returns
    -------
    count : int
        Number of NaN elements in `arr`.
    """
    # np.isnan works for both real and complex data. For complex data, if
    # either the real or imag part is NaN, then the pixel is considered NaN.
    return np.sum(np.isnan(arr))


def compute_inf_count(arr: ArrayLike) -> int:
    """
    Get the number of +/- Inf elements in the input array.

    Parameters
    ----------
    arr : array_like
        Input array; can have a real or complex dtype.
        (For complex data, if either the real or imag part is +/- Inf,
        then the element is considered +/- Inf.)

    Returns
    -------
    count : int
        Number of +/- Inf elements in `arr`.
    """
    # (np.isinf works for both real and complex data. For complex data, if
    # either real or imag part is +/- inf, then the pixel is considered inf.)
    return np.sum(np.isinf(arr))


def compute_fill_count(
    arr: ArrayLike, fill_value: int | float | complex
) -> int:
    """
    Get the number of fill value elements in the input array.

    Parameters
    ----------
    arr : array_like
        Input array.
    fill_value : int or float or complex
        The fill value for `arr`. The type should correspond to dtype of `arr`.

    Returns
    -------
    count : int
        Number of elements in `arr` that are `fill_value`.
    """
    if np.isnan(fill_value):
        # `np.nan == np.nan` evaluates to False, so handle this case here
        return compute_nan_count(arr=arr)

    return np.sum(np.equal(arr, fill_value))


def compute_near_zero_count(arr: ArrayLike, epsilon: float = 1e-6) -> int:
    """
    Get the number of near-zero elements in the input array.

    Parameters
    ----------
    arr : array_like
        Input array; can have a real or complex dtype.
        (For complex data, the magnitude must be ~0.0.)
    epsilon : float, optional
        Absolute tolerance for determining if a raster pixel is 'near zero'.
        Defaults to 1e-6.

    Returns
    -------
    count : int
        Number of near-zero elements in `arr`.
    """
    # By using np.abs(), for complex values this will compute the magnitude.
    return np.sum(np.abs(arr) < epsilon)


def compute_percentage_metrics(
    raster: nisarqa.Raster,
    params: nisarqa.ThresholdParamGroup,
    stats_h5: h5py.File,
) -> None:
    """
    Check % nan, % zero, % fill, % inf, % total invalid; save to HDF5 and CSV.

    Parameters
    ----------
    raster : nisarqa.Raster
        Input Raster.
    stats_h5 : h5py.File
        The output file to save QA metrics to.
    params : nisarqa.ThresholdParamGroup
        A structure containing the parameters for checking the percentage
        of invalid pixels in a raster.

    Warnings
    --------
    The entire input array will be read into memory and processed.
    Only use this function for small datasets.

    Notes
    -----
    If the fill value is set to None in the input *Raster, that field will
    not be computed nor included in the STATS.h5 file.
    """
    # Create flags, to be used for the PASS/FAIL Summary CSV
    all_metrics_pass = True

    # Total number of pixels that were NaN, +/-Inf, fill, or near-zero
    total_num_invalid = 0

    log = nisarqa.get_logger()
    summary = nisarqa.get_summary()

    arr = raster.data
    grp_path = raster.stats_h5_group_path
    fill_value = raster.fill_value
    arr_name = raster.name
    arr_size = np.size(arr)

    nan_threshold = params.nan_threshold
    inf_threshold = params.inf_threshold
    fill_threshold = params.fill_threshold
    near_zero_threshold = params.near_zero_threshold
    epsilon = params.epsilon
    zero_is_invalid = params.zero_is_invalid
    invalid_threshold = params.total_invalid_threshold

    def _percent_of_arr(count: int) -> float:
        return count / arr_size * 100

    # Compute NaN value metrics
    num_nan = compute_nan_count(arr)
    total_num_invalid += num_nan
    percent_nan = _percent_of_arr(count=num_nan)

    nisarqa.save_percent_nan_to_stats_h5(
        percentage=percent_nan, stats_h5=stats_h5, grp_path=grp_path
    )

    all_metrics_pass &= nisarqa.percent_nan_is_within_threshold(
        percentage=percent_nan,
        threshold_percentage=nan_threshold,
        arr_name=arr_name,
    )

    # Compute +/- inf metrics.
    num_inf = compute_inf_count(arr)
    total_num_invalid += num_inf
    percent_inf = _percent_of_arr(count=num_inf)

    nisarqa.save_percent_inf_to_stats_h5(
        percentage=percent_inf, stats_h5=stats_h5, grp_path=grp_path
    )

    all_metrics_pass &= nisarqa.percent_inf_is_within_threshold(
        percentage=percent_inf,
        threshold_percentage=inf_threshold,
        arr_name=arr_name,
    )

    # Compute number of zeros metrics.
    num_zero = compute_near_zero_count(arr, epsilon=epsilon)

    if zero_is_invalid:
        total_num_invalid += num_zero

    percent_zero = _percent_of_arr(count=num_zero)

    nisarqa.save_percent_near_zero_to_stats_h5(
        percentage=percent_zero,
        epsilon=epsilon,
        stats_h5=stats_h5,
        grp_path=grp_path,
    )

    all_metrics_pass &= nisarqa.percent_near_zero_is_within_threshold(
        percentage=percent_zero,
        threshold_percentage=near_zero_threshold,
        arr_name=arr_name,
    )

    # Compute fill value metrics. Do not double-count NaNs nor zeros.
    if fill_value is not None:
        fill_is_zero = np.isclose(fill_value, 0.0, atol=epsilon, rtol=0.0)

        if np.isnan(fill_value):
            percent_fill = percent_nan
            # We already accumulated the number of NaN to `total_num_invalid`,
            # skip doing that here so that we do not double-count the NaN
        elif fill_is_zero:
            percent_fill = percent_zero
            if not zero_is_invalid:
                # Fill values should always be included as invalid pixels.
                total_num_invalid += num_zero
        else:
            num_fill = compute_fill_count(arr, fill_value=fill_value)
            total_num_invalid += num_fill
            percent_fill = _percent_of_arr(count=num_fill)

        nisarqa.save_percent_fill_to_stats_h5(
            percentage=percent_fill,
            fill_value=fill_value,
            stats_h5=stats_h5,
            grp_path=grp_path,
        )

        all_metrics_pass &= nisarqa.percent_fill_is_within_threshold(
            percentage=percent_fill,
            threshold_percentage=fill_threshold,
            fill_value=fill_value,
            arr_name=arr_name,
        )

    # Compute cumulative total invalid pixels
    assert total_num_invalid <= arr_size

    percent_invalid = _percent_of_arr(count=total_num_invalid)

    nisarqa.save_percent_total_invalid_to_stats_h5(
        percentage=percent_invalid, stats_h5=stats_h5, grp_path=grp_path
    )

    all_metrics_pass &= nisarqa.percent_total_invalid_is_within_threshold(
        percentage=percent_invalid,
        threshold_percentage=invalid_threshold,
        arr_name=arr_name,
    )

    # Now, all metrics have been computed and logged. Raise exception
    # if an issue was identified.
    if not all_metrics_pass:

        msg = (
            f"Array {arr_name} did not pass at least one of the percentage"
            " threshold metrics; either the % Nan, % Inf, % 'fill', % near-zero"
            " and/or % total invalid pixels was greater than its requested"
            " threshold. See the log for exact details."
        )

        raise nisarqa.InvalidRasterError(msg)


def get_unique_elements_and_percentages(
    arr: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the sorted unique elements of an array and the percentage they occur.

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    unique_values : numpy.ndarray
        1D array of the sorted unique values in `arr`.
    percentages : numpy.ndarray
        1D array of the relative frequency that each of the unique values in
        `unique_values` occurs in `arr`, as a percentage of the total array
        size. Elements in `percentages` correspond to the elements in
        `unique_values`.
    """

    unique_values, counts = np.unique(arr, return_counts=True)

    percentages = (counts / np.size(arr)) * 100.0

    return unique_values, percentages


def connected_components_metrics(
    cc_raster: nisarqa.RadarRaster | nisarqa.GeoRaster,
    stats_h5: PdfPages,
    max_num_cc: int | None = None,
) -> None:
    """
    Compute metrics specific to Connected Components; save to HDF5 and CSV.

    Parameters
    ----------
    raster : nisarqa.Raster
        Input Raster.
    stats_h5 : h5py.File
        The output file to save QA metrics to.
    max_num_cc : int or None, optional
        Maximum number of valid connected components allowed.
        If the number of valid connected components (not including
        zero nor the fill value) is greater than this value,
        it will be recorded in the summary file and an exception will be raised.
        If None, this error check will be skipped.
        Defaults to None.

    Warnings
    --------
    The entire input array will be read into memory and processed.
    Only use this function for small datasets.
    """

    log = nisarqa.get_logger()
    grp_path = cc_raster.stats_h5_group_path
    name = cc_raster.name

    # Number of valid CC
    # (exclude 0, exclude 255 when computing this metric)
    # FYI - in vast majority of cases (~95% of cases?), only 1 CC is expected.
    # If 2 or more CC are found, then we want to know the % of area
    # labeled with each connected component.
    labels, percentages = get_unique_elements_and_percentages(
        arr=cc_raster.data[()]
    )

    log.info(
        f"Raster {cc_raster.name} contains Connected Component labels: {labels}"
    )

    # "The Breakdown"
    for label, percent in zip(labels, percentages):
        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name=f"connectedComponent{label}",
            ds_data=percent,
            ds_units="1",
            ds_description=(
                "Percentage of raster covered by connected component labeled"
                f" {label}."
            ),
        )

        log.info(
            f"Connected Component {label} covers {percent} percent of the"
            f" {name} raster."
        )

    # Exclude 0 and the fill value for the remaining metrics.
    # For ease, let's simply remove them from the lists.
    labels_list = list(labels)
    percentages_list = list(percentages)

    if 0 in labels_list:
        zero_idx = labels_list.index(0)
        del labels_list[zero_idx]
        del percentages_list[zero_idx]

    fill_value = cc_raster.fill_value
    if fill_value in labels_list:
        fill_idx = labels_list.index(fill_value)
        del labels_list[fill_idx]
        del percentages_list[fill_idx]

    num_valid_cc = len(labels_list)
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="numValidConnectedComponents",
        ds_data=num_valid_cc,
        ds_units="1",
        ds_description=(
            "Number of valid connected components, excluding 0 and"
            f" {fill_value}. ({fill_value} is fill value)"
        ),
    )
    log.info(
        f"Raster {name} contains {num_valid_cc} valid connected components"
        f" (excluding 0 and fill value of {fill_value})."
    )

    # Percentage of pixels in largest connected component
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentPixelsInLargestCC",
        ds_data=max(percentages_list),
        ds_units="1",
        ds_description=(
            "Percentage of pixels in the largest valid connected component. (0"
            f" and fill value ({fill_value}) are not valid connected"
            " components, but their pixels are included when computing the"
            " percentage."
        ),
    )

    # Percentage of pixels with non-zero, non-fill connected components
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=grp_path,
        ds_name="percentPixelsWithNonZeroCC",
        ds_data=sum(percentages_list),
        ds_units="1",
        ds_description=(
            "Percentage of pixels with non-zero, non-fill connected components."
            f" (0 and fill value ({fill_value}) are not valid connected"
            " components, but their pixels are included when computing the"
            " percentage."
        ),
    )

    # If there are too many connected components, raise an exception.
    if max_num_cc is not None:
        summary = nisarqa.get_summary()
        summary_kwargs = {
            "threshold": max_num_cc,
            "actual": num_valid_cc,
            "notes": name,
        }
        if num_valid_cc > max_num_cc:
            summary.check_connected_components_within_threshold(
                result="FAIL", **summary_kwargs
            )
            msg = (
                f"Raster {name} contains {num_valid_cc} valid connected"
                f" components (excluding 0 and fill value of {fill_value}). It"
                f" is only permitted to contain a max of {max_num_cc} valid"
                " connected components."
            )
            raise ValueError(msg)
        else:
            summary.check_connected_components_within_threshold(
                result="PASS", **summary_kwargs
            )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
