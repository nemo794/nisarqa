from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

import nisarqa

objects_to_skip = nisarqa.get_all(__name__)


def multilook(arr, nlooks):
    """
    Multilook an array by simple averaging.

    Performs spatial averaging and decimation. Each element in the output array is the
    arithmetic mean of neighboring cells in the input array.

    Parameters
    ----------
    arr : numpy.ndarray
        1D or 2D Input array with a dtype of float. Invalid values should be np.nan.
    nlooks : int or iterable of int
        Number of looks along each axis of the input array.

    Returns
    -------
    out : numpy.ndarray
        Multilooked array.

    Notes
    -----
    If the length of the input array along a given axis is not evenly divisible by the
    specified number of looks, any remainder samples from the end of the array will be
    discarded in the output.

    If a cell in the input array is nan (invalid), then the corresponding cell in the
    output array will also be nan.
    """

    # Step 1: Prepare and validate the inputs
    if arr.ndim not in (1, 2):
        raise ValueError(f"Input array has {arr.ndim} but must be 1D or 2D.")
    nisarqa.verify_real_or_complex_dtype(arr)
    nlooks = normalize_nlooks(nlooks, arr)

    with nisarqa.ignore_runtime_warnings():
        validate_nlooks(nlooks, arr)

    # Step 2: Initialize output array with zeros.
    out_shape = tuple([m // n for m, n in zip(arr.shape, nlooks)])
    out = np.zeros_like(arr, shape=out_shape)

    # Step 3: Compute the local average of samples by accumulating a weighted sum of
    # cells within each multilook window.

    # Step 3.1: take a view without the 'uneven edges' beyond even multiples of nlooks
    valid_portion = arr[: out_shape[0] * nlooks[0], : out_shape[1] * nlooks[1]]

    # Step 3.2: sum across Axis 1 first. (Idea: collapsing vertical blinds).
    # This takes advantage of numpy's row-major order and vectorization
    # to get a performance boost.

    # Step 3.2.1: Create a list of axis 1 indices at the start of each multilook window
    range_subindices = [i for i in range(0, valid_portion.shape[1], nlooks[1])]

    # Normalization factor (uniform weighting).
    w = 1.0 / np.prod(nlooks)

    # Step 3.2.2: Weight each value in the valid portion of the input array,
    # and then sum each row within a multilook span.
    out = np.add.reduceat(valid_portion * w, range_subindices, axis=1)

    # Step 3.3: sum across Axis 0 next. (Idea: collapsing horizontal blinds).

    # Step 3.3.1: Create a list of axis 0 indices at the start of each multilook window
    az_subindices = [i for i in range(0, valid_portion.shape[0], nlooks[0])]

    # Step 3.3.2: Sum each column within a multilook span.
    # Note that values were already weighted in step 3.2.2.
    out = np.add.reduceat(out, az_subindices, axis=0)

    return out


def normalize_nlooks(nlooks, arr):
    # Normalize `nlooks` into a tuple with length equal to `arr.ndim`. If `nlooks` was a
    # scalar, take the same number of looks along each axis in the array.
    if isinstance(nlooks, int):
        nlooks = (nlooks,) * arr.ndim
    else:
        nlooks = tuple([int(n) for n in nlooks])
        if len(nlooks) != arr.ndim:
            raise ValueError(
                f"length mismatch: length of nlooks ({len(nlooks)}) must match"
                f" input array rank ({arr.ndim})"
            )

    return nlooks


def validate_nlooks(nlooks: int | Sequence[int], arr: ArrayLike) -> None:
    log = nisarqa.get_logger()

    # The number of looks must be at least 1 and at most the size of the input array
    # along the corresponding axis.
    for m, n in zip(arr.shape, nlooks):
        if n < 1:
            raise ValueError("number of looks must be >= 1")
        elif n > m:
            raise ValueError("number of looks should not exceed array shape")

    # Warn if the array shape is not an integer multiple of `nlooks`. Warn at
    # most once (even if multiple axes have this issue).
    for m, n in zip(arr.shape, nlooks):
        if m % n != 0:
            warnings.warn(
                "input array shape is not an integer multiple of nlooks --"
                " remainder samples will be excluded from output",
                RuntimeWarning,
            )
            break


def compute_square_pixel_nlooks(
    img_shape, sample_spacing, longest_side_max=2048, epsilon=1e-2
):
    """
    Computes the nlooks values required to achieve approx. square pixels
    in a multilooked image.

    `nlooks` values will be rounded to the nearest odd value to maintain
    the same coordinate grid as the array that will be multilooked.
    Using an even-valued 'look' would cause the multilooked image's
    coordinates to be shifted a half-pixel from the source image's coordinates.

    Parameters
    ----------
    img_shape : pair of int
        The M x N dimensions of the source array to be multilooked.
        Format: (M, N)  <==>  (<Y direction>, <X direction>)
    sample_spacing : pair of float
        The Y direction sample spacing and X direction sample spacing
        of the source array.
        For radar-domain products, Y direction corresponds to azimuth,
        and X direction corresponds to range.
        Only the magnitude (absolute value) of the sample spacing is used.
        Format: (dy, dx)
    longest_side_max : int, optional
        The maximum number of pixels allowed for the longest side of the final
        2D multilooked image.
        Defaults to 2048.
    epsilon : float or int, optional
        If the values in `sample_spacing` are within `epsilon` of each other,
        it is likely that the image already was created with square pixels,
        and so this function will check that the computed nlooks values
        are equal. Typically, a user will use the default.
        Defaults to 1e-2.

    Returns
    -------
    nlooks : pair of int
        The nlooks values for Y direction and X direction.
        Format: (ky, kx)

    Notes
    -----
    The function assumes that the source image for `img_shape` will
    be truncated for multilooking so that its dimensions are integer
    multiples of `nlooks`. If instead the user intends to pad the source
    image array with zeros prior to multilooking, then the user should
    update `img_shape` so that the longer dimension (in ground distance)
    is increased to be an integer multiple of `longest_side_max`, with the
    shorter dimension increased proportionally.

    Derivation for formulas:

    Assumptions and Variable Definitions:
    Source Image has:
        - dimensions M x N
            - M : (int) # Y rows
            - N : (int) # X columns
        - sample spacing dX and dY
            - dy : (float) distance between samples in Y direction
            - dx : (float) distance between samples in X direction

    Output Multilooked Image has:
        - Same units as Source Image (e.g. sample spacing is in meters for both)
        - dimensions M_1 x N_1
            - M_1 : (int) # Y rows
            - N_1 : (int) # X columns
        - sample spacing dX_1 and dY_1
            - dy_1 : (float) distance between samples in Y direction
            - dx_1 : (float) distance between samples in X direction
    Number of Looks:
        - These will be Real numbers during the computation, but then
          rounded to the nearest odd integer for final output.
        - ky : number of looks in the Y direction
        - kx : number of looks in the X direction

    Problem Setup:
    (1) dy * ky = dy_1      # Y sample spacing is scaled by nlooks
    (2) dx * kx = dx_1      # X sample spacing is scaled by nlooks
    (3) M / ky = M_1        # num Y rows is scaled by nlooks
    (4) N / kx = N_1        # num X columns is scaled by nlooks
    (5) dx_1 = dy_1         # Set equal (the multilooked image will have ~square pixels)
    (6) Dy = M * dy         # total extent of image in y real space coordinates
    (7) Dx = N * dx         # total extent of image in x real space coordinates

    Derivation:
    (8) WLOG, let Dy > Dx                 # Determine via a conditional
    (9) ky = M / longest_side_max         # By definitions
    (9.5) ky = next_greater_odd_int(ky)   # round to next greater odd int here
                                          #    for "square-er" pixels
    (10) ky * dy = kx * dx                # from (5), (1), (2)
    (11) kx = (ky * dy) / dx              # from (10), (9)
    (11.5) kx = next_greater_odd_int(kx)  # nlooks values are int

    ** `longest_side_max` provides an upper limit on the length of the
    longest side of the final multilooked image. So, during
    computation of ky and kx, they will be rounded to the nearest
    odd integer greater than the exact Real solution for ky and kx.
    Rounding up will ensure that we do not exceed `longest_side_max`.
    Rounding to odd values will maintain the same coordinate grid
    as the source image array during multilooking.
    """

    for input in (img_shape[0], img_shape[1], longest_side_max):
        if (not isinstance(input, int)) or (input < 1):
            raise TypeError(
                f"`Input is type {type(input)} "
                f"but must be a positive int: {input}"
            )

    # Variables
    M = img_shape[0]  # Y dimension
    N = img_shape[1]  # X dimension
    dy = np.abs(sample_spacing[0])  # Y
    dx = np.abs(sample_spacing[1])  # X

    if M * dy >= N * dx:
        # Y (azimuth) extent is longer
        ky = M / longest_side_max

        # Adjust ky to be an odd integer before computing kx. This will
        # keep the final multilooked pixels closer to being square pixels.
        ky = nisarqa.next_greater_odd_int(ky)

        kx = (ky * dy) / dx  # Formula (11)
        kx = nisarqa.next_greater_odd_int(kx)

    else:
        # X (range) ground distance is longer
        kx = N / longest_side_max
        kx = nisarqa.next_greater_odd_int(kx)

        ky = (kx * dx) / dy  # Formula (11)
        ky = nisarqa.next_greater_odd_int(ky)

    # Sanity Check
    assert N // kx <= longest_side_max
    assert M // ky <= longest_side_max

    # Sanity Check
    # If the pair of values in `sample_spacing` are within epsilon of each
    # other, it is likely that the input image's pixels are already
    # "square pixels". As a sanity check for this QA algorithm, check that
    # the returned `nlooks` values are equal to each other.
    # For example, GCOV products have square pixels.
    if (np.abs(dy - dx) < epsilon) and (kx != ky):
        nisarqa.get_logger().warning(
            "Computation of nlooks did not retain square pixels. "
            "Image aspect ratio will be distorted."
        )

    return (ky, kx)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
