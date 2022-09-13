import h5py
from contextlib import contextmanager
import numpy as np

from utils import input_verification as iv

@contextmanager
def open_h5_file(in_file, mode='r'):
    """
    Open or create a handle for a h5 file.

    Parameters
    ----------
    in_file : string
        Filepath to a h5 file
    mode : char
        The mode to open the input file.
        'r'         - Readonly, file must exist (default)
        'r+'        - Read/write, file must exist
        'w'         - Create file, truncate if exists
        'w-' or 'x' - Create file, fail if exists
        'a'         - Read/write if exists, create otherwise
    
    Returns
    -------
    handle : h5py file handle
        Handle to `filepath` file
    """
    try:
        input_file = h5py.File(in_file, mode)
    
    # TODO If file is already open, this error is thrown: BlockingIOError
    except (FileNotFoundError, IOError) as e:
        print("File couldn't open. Add logger and remove this print statement.")
        # logger.log_message(logging_base.LogFilterError, \
        #                     "File %s has a Fatal Error(s): %s" % (rslc_file, errors))
        raise
    else:
        yield input_file
    finally:
        input_file.close()


def compute_non_zero_mask(arr, epsilon=1.0E-05):
    """
    Create a mask of the non-zero pixels in the input array.

    Parameters
    ----------
    arr : array_like
        The input array.
    epsilon : float, optional
        Tolerance for if an element in `arr` is considered "zero"

    Returns
    -------
    mask : Boolean array
        Array with same shape as `arr`.
        True for non-zero entries.
        False for entries where both the real and imag components
        are less than `epsilon`.
    """
    if np.issubdtype(arr.dtype, np.complexfloating):
        zero_real = np.abs(arr.real) < epsilon
        zero_imag = np.abs(arr.imag) < epsilon
        return ~(zero_real & zero_imag)
    else:
        zero_real = np.abs(arr) < epsilon
        return ~zero_real


def compute_mask_ok(arr, finite_entries=None, non_zero=None, \
                    epsilon=1.0E-05):
    """
    Create a mask of the valid (finite, non-zero) pixels in arr.

    Parameters
    ----------
    arr : array_like
        The input array
    finite_entries : Boolean array, optional
        Array with same shape as `arr`.
        True for entries that are not inf and not nan.
    non_zero : Boolean array, optional
        Array with same shape as `arr`.
        True for entries that are not zero
    epsilon : float, optional
        Tolerance for if an element in `arr` is considered "zero"

    Returns
    -------
    mask_ok : array_like
        Array with same shape as `arr`.
        True for valid entries. Valid entries finite values that are
        not inf, not nan, and not "zero" (where zero is < params.EPS)
        False for entries that have a nan or inf in either the real
        or imaginary component, or a zero in both real and imag components.

    See Also
    --------
    numpy.isfinite : Can be used to compute `finite_entries`
    utility.compute_non_zero_mask : Can be used to compute `non_zero`
    """

    if finite_entries is None:
        finite_mask = np.isfinite(arr)
        # print("Number of nan's: ", np.sum(~finite_mask))
        # print("Percentage of pixels are nan: ", (np.sum(~finite_mask)/finite_mask.size) * 100)

    if non_zero is None:
        non_zero_mask = compute_non_zero_mask(arr, epsilon)
        # print("Number of zero's: ", np.sum(~non_zero_mask))
        # print("Percentage of pixels are zero: ", (np.sum(~non_zero_mask)/non_zero_mask.size) * 100)

    mask_ok = finite_mask & non_zero_mask

    return mask_ok

