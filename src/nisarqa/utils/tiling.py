from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)

###############################################
######      Main Tiling Functions       #######
###############################################


class TileIterator:
    def __init__(
        self,
        arr_shape,
        tile_ncols=-1,
        tile_nrows=-1,
        col_stride=1,
        row_stride=1,
    ):
        """
        Simple iterator class to iterate over a 1D or 2D array by tiles.

        Parameters
        ----------
        arr_shape : tuple
            The shape of the 1D or 2D array that this TileIterator is for.
        tile_ncols : int
            Number of columns for each tile
            Defaults to -1, meaning entire rows will be processed.
        tile_nrows : int, optional
            Number of rows for each tile.
            Defaults to -1, meaning entire columns will be processed.
            To process entire columns in an array, set tile_nrows = arr_shape[0]
            Will be ignored if arr_shape is for a 1D array.
        col_stride : int, optional
            Amount to decimate the input array along axis 1.
            Ex: If `col_stride` is 5, then the slices returned during the
            iteration process will have a column-wise step value of 5, i.e.
            columns 0, 5, 10,... will be returned
            Defaults to 1 (no decimation).
        row_stride : int, optional
            Amount to decimate the input array along axis 0.
            Ex: If `row_stride` is 5, then the slices returned during the
            iteration process will have a row-wise step value of 5, i.e.
            rows 0, 5, 10,... will be returned
            Defaults to 1 (no decimation).
        """

        self.arr_shape = arr_shape

        if tile_ncols == -1:
            self.tile_ncols = arr_shape[1]
        else:
            self.tile_ncols = tile_ncols

        self.col_stride = col_stride

        self.num_dim = len(arr_shape)
        if self.num_dim not in (1, 2):
            raise ValueError(
                f"Provided array shape has {self.num_dim} dimensions"
                " but only 1 or 2 dimensions are currently supported."
            )

        # If the array is 2D, set tile_nrows and row_stride
        if self.num_dim == 2:
            if tile_nrows == -1:
                self.tile_nrows = arr_shape[0]
            else:
                self.tile_nrows = tile_nrows

            self.row_stride = row_stride
        else:
            # There is no tile_nrows nor row_stride for a 1D array
            # Set to None so that errors are raised if these are called.
            self.tile_nrows = None
            self.row_stride = None

        msg = (
            "The %s stride length %d is not an integer"
            + "multiple of the %s decimation value %s."
            + "This will lead to incorrect decimation of the array."
        )

        # Warn if the `tile_ncols` is not an integer multiple of `col_stride`
        if self.tile_ncols % self.col_stride != 0:
            warnings.warn(
                msg % ("column", self.tile_ncols, "column", self.col_stride),
                RuntimeWarning,
            )

        # Warn if the `tile_nrows` is not an integer multiple of `row_stride`
        if self.tile_nrows % self.row_stride != 0:
            warnings.warn(
                msg % ("row", self.tile_nrows, "row", self.row_stride),
                RuntimeWarning,
            )

    def __iter__(self):
        """
        Iterator for TileIterator class.

        Yields
        ------
        np_slice : tuple of slice objects
            A tuple of slice objects that can be used for
            indexing into the next tile of an array_like object.
        """
        for row_start in range(0, self.arr_shape[0], self.tile_nrows):
            for col_start in range(0, self.arr_shape[1], self.tile_ncols):
                if self.num_dim == 2:
                    yield np.s_[
                        row_start : row_start
                        + self.tile_nrows : self.row_stride,
                        col_start : col_start
                        + self.tile_ncols : self.col_stride,
                    ]

                else:  # 1 dimension array
                    yield np.s_[
                        col_start : col_start
                        + self.tile_ncols : self.col_stride
                    ]


def process_arr_by_tiles(
    in_arr, out_arr, func, input_batches, output_batches, in_arr_2=None
):
    """
    Map a function to tiles of an array.

    Apply `func` to the input array sequentially by tiles
    and store the result in `out_arr`.

    TODO - at end of development, see if there is a reason to
    keep in_arr_2 parameter.

    Parameters
    ----------
    in_arr : array_like
        Input 1D or 2D array
    out_arr : array_like
        Output 1D or 2D array. Will be populated by this function.
    func : function or partial function
        Function to apply to every tile in the input array(s).
        The function should take an array as a single positional argument
        (or two arrays as positional arguments if `in_arr_2` is not None)
        and return an output array. For a given input tile shape,
        `func` must return an array with the same shape as an output tile shape.
    input_batches : TileIterator
        Iterator for the input array
    output_batches : TileIterator
        Iterator for the output array
    in_arr_2 : array_like, optional
        Optional 2nd input argument of same shape as `in_arr`.
    """

    for out_slice, in_slice in zip(output_batches, input_batches):
        # Process this batch
        if in_arr_2 is None:
            tmp_out = func(in_arr[in_slice])
        else:
            tmp_out = func(in_arr[in_slice], in_arr_2[in_slice])

        # Write the batch output to the output array
        out_arr[out_slice] = tmp_out


###############################################
######       RSLC Tiling Functions      #######
###############################################


def compute_multilooked_backscatter_by_tiling(
    arr, nlooks, input_raster_represents_power=False, tile_shape=(512, -1)
):
    """
    Compute the multilooked backscatter array (linear units) by tiling.

    Parameters
    ----------
    arr : array_like
        The input 2D array
    nlooks : tuple of ints
        Number of looks along each axis of the input array to be
        averaged during multilooking.
        Format: (num_rows, num_cols)
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either power or root power.
        If `True`, then QA SAS assumes the input data already represents
        power and uses the pixels' magnitudes for computations.
        If `False`, then QA SAS assumes the input data represents root power
        aka magnitude and will handle the full computation to power using
        the formula:  power = abs(<magnitude>)^2 .
        Defaults to False (root power).
    tile_shape : tuple of ints
        Shape of each tile to be processed. If `tile_shape` is
        larger than the shape of `arr`, or if the dimensions of `arr`
        are not integer multiples of the dimensions of `tile_shape`,
        then smaller tiles may be used.
        -1 to use all rows / all columns (respectively).
        Format: (num_rows, num_cols)
        Defaults to (512, -1) to use all columns (i.e. full rows of data)
        and leverage Python's row-major ordering.

    Returns
    -------
    multilook_img : numpy.ndarray
        The multilooked backscatter image in linear units

    Notes
    -----
    If the length of the input array along a given axis is not evenly divisible by the
    specified number of looks, any remainder samples from the end of the array will be
    discarded in the output.

    If a cell in the input array is nan (invalid), then the corresponding cell in the
    output array will also be nan.

    """
    arr_shape = np.shape(arr)

    if len(arr_shape) != 2:
        raise ValueError(
            f"Input array has shape {arr_shape} but can only have 2 dimensions."
        )

    if (arr_shape[0] < nlooks[0]) or (arr_shape[1] < nlooks[1]):
        raise ValueError(
            f"{nlooks=} but the array has has dimensions {arr_shape}. For each "
            "dimension, `nlooks` must be <= the length of that dimension."
        )

    if tile_shape[0] == -1:
        tile_shape = (arr_shape[0], tile_shape[1])
    if tile_shape[1] == -1:
        tile_shape = (tile_shape[0], arr_shape[1])

    if len(nlooks) != 2:
        raise ValueError(f"`nlooks` must be a tuple of length 2: {nlooks}")
    if not all(isinstance(x, int) for x in nlooks):
        raise ValueError(f"`nlooks` must contain only ints: {nlooks}")

    # Compute the portion (shape) of the input array
    # that is integer multiples of nlooks.
    # This will be used to trim off (discard) the 'uneven edges' of the image,
    # i.e. the pixels beyond the largest integer multiples of nlooks.
    in_arr_valid_shape = tuple(
        [(m // n) * n for m, n in zip(arr_shape, nlooks)]
    )

    # Compute the shape of the output multilooked array
    final_out_arr_shape = tuple([m // n for m, n in zip(arr_shape, nlooks)])

    # Adjust the tiling shape to be integer multiples of nlooks
    # Otherwise, the tiling will get messy to book-keep.

    # If a tile dimension is smaller than nlooks, grow it to be the same length
    if tile_shape[0] < nlooks[0]:
        tile_shape = (nlooks[0], tile_shape[1])
    if tile_shape[1] < nlooks[1]:
        tile_shape = (tile_shape[0], nlooks[1])

    # Next, shrink the tile shape to be an integer multiple of nlooks
    in_tiling_shape = tuple([(m // n) * n for m, n in zip(tile_shape, nlooks)])

    out_tiling_shape = tuple([m // n for m, n in zip(tile_shape, nlooks)])

    # Create the Iterators
    input_iter = TileIterator(
        in_arr_valid_shape,
        tile_nrows=in_tiling_shape[0],
        tile_ncols=in_tiling_shape[1],
    )
    out_iter = TileIterator(
        final_out_arr_shape,
        tile_nrows=out_tiling_shape[0],
        tile_ncols=out_tiling_shape[1],
    )

    # Create an inner function for this use case.
    def calc_backscatter_and_multilook(arr):
        # square the pixel values (e.g to convert from magnitude to power),
        # if requested.
        # Otherwise, take the absolute value to ensure we're using the
        # magnitude for either real or complex values
        out = (
            np.abs(arr)
            if input_raster_represents_power
            else nisarqa.arr2pow(arr)
        )

        # Multilook
        out = nisarqa.multilook(out, nlooks)

        return out

    # Instantiate the output array
    multilook_img = np.zeros(
        final_out_arr_shape, dtype=np.float32
    )  # 32 bit precision

    # Ok to pass the full input array; the tiling iterators
    # are constrained such that the 'uneven edges' will be ignored.
    process_arr_by_tiles(
        arr,
        multilook_img,
        calc_backscatter_and_multilook,
        input_batches=input_iter,
        output_batches=out_iter,
    )

    return multilook_img


def compute_histogram_by_tiling(
    arr: ArrayLike,
    bin_edges: np.ndarray,
    arr_name: str = "",
    data_prep_func: Callable | None = None,
    density: bool = False,
    decimation_ratio: tuple[int, int] = (1, 1),
    tile_shape: tuple[int, int] = (512, -1),
) -> np.ndarray:
    """
    Compute decimated histograms by tiling.

    Parameters
    ----------
    arr : array_like
        The input array
    bin_edges : numpy.ndarray
        The bin edges to use for the histogram
    arr_name : str
        Name for the array. (Will be used for log messages.)
    data_prep_func : Callable or None, optional
        Function to process each tile of data through before computing
        the histogram counts. For example, this function can be used
        to convert the values in each tile of raw data to backscatter,
        dB scale, etc. before taking the histogram.
        If `None`, then histogram will be computed on `arr` as-is,
        and no pre-processing of the data will occur.
        Defaults to None.
    density : bool, optional
        If True, return probability densities for histograms:
        Each bin will display the bin's raw count divided by the
        total number of counts and the bin width
        (density = counts / (sum(counts) * np.diff(bins))),
        so that the area under the histogram integrates to 1
        (np.sum(density * np.diff(bins)) == 1).
        Defaults to False.
    decimation_ratio : pair of int, optional
        The step size to decimate the input array for computations.
        For example, (2,3) means every 2nd azimuth line and
        every 3rd range line will be used to compute the histograms.
        Defaults to (1,1), i.e. no decimation will occur.
        Format: (<azimuth>, <range>)
    tile_shape : tuple of ints, optional
        Shape of each tile to be processed. If `tile_shape` is
        larger than the shape of `arr`, or if the dimensions of `arr`
        are not integer multiples of the dimensions of `tile_shape`,
        then smaller tiles may be used.
        -1 to use all rows / all columns (respectively).
        Format: (num_rows, num_cols)
        Defaults to (512,-1) to use all columns (i.e. full rows of data)
        and leverage Python's row-major ordering.

    Returns
    -------
    hist_counts : numpy.ndarray
        The histogram counts.
        If `density` is True, then the backscatter and phase histogram
        densities (respectively) will be returned instead.

    Notes
    -----
    If a cell in the input array is non-finite (invalid),
    then it will not be included in the counts for either
    backscatter nor phase.

    If a cell in the input array is almost zero, then it will not
    be included in the counts for phase.
    """
    arr_shape = np.shape(arr)

    if (arr_shape[0] < decimation_ratio[0]) or (
        arr_shape[1] < decimation_ratio[1]
    ):
        raise ValueError(
            f"{decimation_ratio=} but the array has has dimensions {arr_shape}."
            " For axis 0 and axis 1, `decimation_ratio` must be <= the length"
            " of that dimension."
        )

    if tile_shape[0] == -1:
        tile_shape = (arr_shape[0], tile_shape[1])
    if tile_shape[1] == -1:
        tile_shape = (tile_shape[0], arr_shape[1])

    # Shrink the tile shape to be an even multiple of the decimation ratio.
    # Otherwise, the decimation will get messy to book-keep.
    in_tiling_shape = tuple(
        [m - (m % n) for m, n in zip(tile_shape, decimation_ratio)]
    )

    # Create the Iterator over the input array
    input_iter = TileIterator(
        arr_shape,
        tile_nrows=in_tiling_shape[0],
        tile_ncols=in_tiling_shape[1],
        row_stride=decimation_ratio[0],
        col_stride=decimation_ratio[1],
    )

    # Initialize accumulator arrays
    # Use dtype of int to avoid floating point errors
    # (The '- 1' is because the final entry in the *_bin_edges array
    # is the endpoint, which is not considered a bin itself.)
    hist_counts = np.zeros((len(bin_edges) - 1,), dtype=int)

    # Do calculation and accumulate the counts
    for tile_slice in input_iter:
        arr_slice = arr[tile_slice]

        # Remove invalid entries
        # Note: for generating histograms, we do not need to retain the
        # original shape of the array.
        arr_slice = arr_slice[np.isfinite(arr_slice)]

        # Prep the data
        if data_prep_func is not None:
            arr_slice = data_prep_func(arr_slice)

        # Clip the array so that it falls within the bounds of the histogram
        arr_slice = np.clip(arr_slice, a_min=bin_edges[0], a_max=bin_edges[-1])

        # Accumulate the counts
        counts, _ = np.histogram(arr_slice, bins=bin_edges)
        hist_counts += counts

    # If the histogram counts are all zero, then the raster likely did not
    # contain any valid imagery pixels. (Typically, this occurs when
    # there was an issue with ISCE3 processing, and the raster is all NaNs.)
    if np.any(hist_counts):
        sum_check = "PASS"
    else:
        sum_check = "FAIL"
        errmsg = (
            f"{arr_name} histogram contains all zero values. This often occurs"
            " if the source raster contains all NaN values."
        )
        nisarqa.get_logger().error(errmsg)

    # Note result of the check in the summary file before raising an Exception.
    nisarqa.get_summary().check_invalid_pixels_within_threshold(
        result=sum_check,
        threshold="",
        actual="",
        notes=(
            f"{arr_name}: all histogram bin counts are zero."
            " This likely indicates that the raster contained no valid data."
            " Note: check performed on decimated raster not full raster."
        ),
    )

    if sum_check == "FAIL":
        raise nisarqa.InvalidRasterError(errmsg)

    if density:
        # Change dtype to float
        hist_counts = hist_counts.astype(float)

        # Compute density
        hist_counts = nisarqa.counts2density(hist_counts, bin_edges)

    return hist_counts


def compute_range_spectra_by_tiling(
    arr: ArrayLike,
    sampling_rate: float,
    az_decimation: int = 1,
    tile_height: int = 512,
    fft_shift: bool = True,
) -> np.ndarray:
    """
    Compute the normalized range power spectral density in dB re 1/Hz by tiling.

    Parameters
    ----------
    arr : array_like
        The input array, representing a two-dimensional discrete-time signal.
    sampling_rate : numeric
        Range sample rate (inverse of the sample spacing) in Hz.
    az_decimation : int, optional
        The stride to decimate the input array along the azimuth axis.
        For example, `4` means every 4th range line will
        be used to compute the range spectra.
        If `1`, no decimation will occur (but is slower to compute).
        Must be greater than zero. Defaults to 1.
    tile_height : int, optional
        User-preferred tile height (number of range lines) for processing
        images by batches. Actual tile shape may be modified by QA to be
        an integer multiple of `az_decimation`. -1 to use all rows.
        Note: full rows must be read in, so the number of columns for each tile
        will be fixed to the number of columns in the input raster.
        Defaults to 512.
    fft_shift : bool, optional
        True to have the frequencies in `range_power_spec` be continuous from
        negative (min) -> positive (max) values.

        False to leave `range_power_spec` as the output from
        `numpy.fft.fftfreq()`, where this discrete FFT operation orders values
        from 0 -> max positive -> min negative -> 0- . (This creates
        a discontinuity in the interval's values.)

        Defaults to True.

    Returns
    -------
    S_out : numpy.ndarray
        Normalized range power spectral density in dB re 1/Hz of `arr`.
    """
    shape = np.shape(arr)
    if len(shape) != 2:
        raise ValueError(
            f"Input array has {len(shape)} dimensions, but must be 2D."
        )

    nrows, ncols = shape

    if az_decimation > nrows:
        raise ValueError(
            f"{az_decimation=}, must be <= the height of the input array"
            f" ({nrows} pixels)."
        )

    if tile_height == -1:
        tile_height = nrows

    # Adjust the tile height to be an integer multiple of `az_decimation`.
    # Otherwise, the decimation will get messy to book-keep.
    if tile_height < az_decimation:
        # Grow tile height
        tile_height = az_decimation
    else:
        # Shrink tile height (less risk of memory issues)
        tile_height = tile_height - (tile_height % az_decimation)

    # Compute total number of range lines that will be used to generate
    # the range power spectra. This will be used for averaging.
    # The TileIterator will truncate the array azimuth direction to be
    # an integer multiple of the stride, so use integer division here.
    num_range_lines = nrows // az_decimation

    # Create the Iterator over the input array
    input_iter = TileIterator(
        np.shape(arr), tile_nrows=tile_height, row_stride=az_decimation
    )

    # Initialize the accumulator array
    S_avg = np.zeros(ncols)

    for tile_slice in input_iter:
        arr_slice = arr[tile_slice]

        # Compute fft over range axis (axis 1)
        # Note: Ensure no normalization occurs in this FFT! We'll handle that
        # manually below.
        # Units of `fft` are the same as the units of `arr_slice`: unitless
        fft = nisarqa.compute_fft(arr_slice, axis=1)

        # Compute the power
        S = np.abs(fft) ** 2

        # Normalize the transform
        S /= ncols  # ncols is the number of fft bins

        # Average over the azimuth axis
        # (We are processing the raster by tiles, but the total summation of
        # power for the entire array might cause float overflow.
        # So, during the accumulation process, if we divide each tile's
        # power density by the total number of range lines used, then the
        # final accumulated array will be mathmatically equivalent to
        # the average of the entire array.)
        S_avg += np.sum(S, axis=0) / num_range_lines

    # Normalize by the sampling rate
    # This makes the units unitless/Hz
    S_out = S_avg / sampling_rate

    # Convert to dB
    with nisarqa.ignore_runtime_warnings():
        # This line throws these warnings:
        #   "RuntimeWarning: divide by zero encountered in log10"
        # when there are zero values. Ignore those warnings.
        S_out = nisarqa.pow2db(S_out)

    if fft_shift:
        # Shift S_out to be aligned with the
        # shifted FFT frequencies.
        S_out = np.fft.fftshift(S_out)

    return S_out


__all__ = nisarqa.get_all(__name__, objects_to_skip)
