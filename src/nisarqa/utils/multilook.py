import numpy as np
import warnings

import nisarqa

def multilook(arr, nlooks):
    '''
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
    '''

    # Step 1: Prepare and validate the inputs
    if arr.ndim not in (1, 2):
        raise ValueError(f'Input array has {arr.ndim} but must be 1D or 2D.')
    nisarqa.verify_real_or_complex_dtype(arr)
    nlooks = normalize_nlooks(nlooks, arr)
    validate_nlooks(nlooks, arr)

    # Step 2: Initialize output array with zeros.
    out_shape = tuple([m // n for m, n in zip(arr.shape, nlooks)])
    out = np.zeros_like(arr, shape=out_shape)

    # Step 3: Compute the local average of samples by accumulating a weighted sum of
    # cells within each multilook window.

    # Step 3.1: take a view without the 'uneven edges' beyond even multiples of nlooks
    valid_portion = arr[:out_shape[0]*nlooks[0], : out_shape[1]*nlooks[1]]

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
        nlooks = (n,) * arr.ndim
    else:
        nlooks = tuple([int(n) for n in nlooks])
        if len(nlooks) != arr.ndim:
            raise ValueError(
                f'length mismatch: length of nlooks ({len(nlooks)}) must match input'
                f' array rank ({arr.ndim})'
            )

    return nlooks


def validate_nlooks(nlooks, arr):
    # The number of looks must be at least 1 and at most the size of the input array
    # along the corresponding axis.
    for m, n in zip(arr.shape, nlooks):
        if n < 1:
            raise ValueError('number of looks must be >= 1')
        elif n > m:
            raise ValueError('number of looks should not exceed array shape')

    # Warn if the array shape is not an integer multiple of `nlooks`. Warn at most once
    # (even if multiple axes have this issue).
    for m, n in zip(arr.shape, nlooks):
        if m % n != 0:
            warnings.warn(
                'input array shape is not an integer multiple of nlooks -- remainder'
                ' samples will be excluded from output',
                RuntimeWarning,
            )
            break


def compute_square_pixel_nlooks(img_shape, sample_spacing, num_mpix=4.0):
    '''
    Computes the nlooks values required to achieve approx. square pixels
    in a multilooked image.

    `nlooks` values will be rounded to the nearest odd value to maintain
    the same coordinate grid as the array that will be multilooked.
    Using an even-valued 'look' would cause the multilooked image's 
    coordinates to be shifted a half-pixel from the source image's coordinates.

    Parameters
    ----------
    img_shape : pair of int
        the M x N dimensions of the source array to be multilooked
        Format: (M, N)
    sample_spacing : pair of float
        The azimuth sample spacing (da) and range sample spacing (dr)
        of the source array.
        Format: (da, dr)
    num_mpix : scalar
        The approx. size (in megapixels) for the final multilooked image.
        Defaults to 4.0 MPix.

    Returns
    -------
    nlooks : pair of int
        The nlooks values for azimuth and range.
        Format: (ka, kr)

    Notes
    -----
    Derivation for the formulas:

    Assumptions and Variable Definitions:
    Source Image has:
        - dimensions M X N
            - M : (int) # range lines
            - N : (int) # azimuth lines
        - sample spacing dr and da
            - dr : (int) distance between samples in range direction
            - da : (int) distance between samples in azimuth direction
    Output Multilooked Image has:
        - Same units as Source Image (e.g. sample spacing is in meters for both)
        - dimensions M_1 X N_1
            - M_1 : (int) # range lines
            - N_1 : (int) # azimuth lines
        - sample spacing dr_1 and da_1
            - dr_1 : (int) distance between samples in range direction
            - da_1 : (int) distance between samples in azimuth direction
    Number of Looks:
        - These will be Real numbers during the computation, but then
          rounded to the nearest odd integer for final output.
        - ka : number of looks in the azimuth direction
        - kr : number of looks in the range direction

    Problem Setup:
    (1) da * ka = da_1                # az sample spacing is scaled by nlooks
    (2) dr * kr = dr_1                # range sample spacing is scaled by nlooks
    (3) M / ka = M_1                  # num az lines is scaled by nlooks
    (4) N / kr = N_1                  # num range lines is scaled by nlooks
    (5) num_Pix = `num_mpix` * 1e6    # convert Megapixels to pixels
    (6) M_1 * N_1 = num_Pix           # output Multilooked image is `num_mpix` MPix
    (7) dr_1 = da_1                   # output Multilooked image should have square pixels

    Derivation:
    (8) kr * dr = ka * da                       # from (7), (1), (2)
    (9) kr = (ka * da) / dr                     # rearrange (8)
    (10) (M / ka) * (N / kr) = num_Pix          # from (6), (3), (4)
    (11) (M / ka) * ((N * dr) / (ka * da)) =    # from (10), (9) -- substitute kr
                            num_Pix
    (12) ka**2 = (M * N * dr) / (da * num_Pix)  # rearrange (11)

    Formula (12) can give the Real-valued ka. This can be used with (8)
    to compute kr.

    Because it is convenient for nlooks to be odd integer values,
    by computing ka and kr as Real values instead of integer values,
    rounding to the nearest odd-valued integers is easily computed.
    '''

    # Variables
    M = img_shape[0]
    N = img_shape[1]
    da = sample_spacing[0]
    dr = sample_spacing[1]
    num_Pix = num_mpix * 1e6

    # Formula (12) -- see docstring
    ka_sqrd = (M * N * dr) / (da * num_Pix)

    # Get Real-Valued ka and kr
    ka = np.sqrt(ka_sqrd)
    kr = (ka * da) / dr  # Formula (8)

    kr = nisarqa.nearest_odd_int(kr)
    ka = nisarqa.nearest_odd_int(ka)

    return (ka, kr)

