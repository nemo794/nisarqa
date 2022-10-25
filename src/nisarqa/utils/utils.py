import h5py
from contextlib import contextmanager
import numpy as np
import nisarqa

objects_to_skip = nisarqa.get_all(name=__name__)

@contextmanager
def open_h5_file(in_file, mode='r'):
    '''
    Open or create a handle for a h5 file.

    Parameters
    ----------
    in_file : str
        Filepath to an HDF5 file.
    mode : char
        The mode to open the input file.
        'r'         - Readonly, file must exist (default)
        'r+'        - Read/write, file must exist
        'w'         - Create file, truncate if exists
        'w-' or 'x' - Create file, fail if exists
        'a'         - Read/write if exists, create otherwise
    
    Returns
    -------
    handle : h5py.File
        Handle to `filepath` file
    '''
    try:
        input_file = h5py.File(in_file, mode)
    
    # TODO If file is already open, this error is thrown: BlockingIOError
    except (FileNotFoundError, IOError) as e:
        print('Could not open file. Add logger and remove this print statement.')
        # logger.log_message(logging_base.LogFilterError,
        #                     'File %s has a Fatal Error(s): %s' % (rslc_file, errors))
        raise
    else:
        yield input_file
    finally:
        input_file.close()


class DatasetNotFoundError(Exception):
    '''Custom exception name for when a dataset is
    not found in an e.g. HDF5 file.
    '''
    def __init__(self):
        super().__init__("Dataset not found.")


def compute_non_zero_mask(arr, epsilon=1.0E-05):
    '''
    Create a mask of the non-zero pixels in the input array.

    Elements in the input array that are approximately equal to zero, 
    based on the specified tolerance, are masked out.
    TODO - after development of the RSLC QA code is complete,
    check that this function is used. If not, delete.

    Parameters
    ----------
    arr : array_like
        The input array.
    epsilon : float, optional
        Absolute tolerance for determining if an element in `arr` 
        is nearly zero.

    Returns
    -------
    mask : Boolean array
        Array with same shape as `arr`.
        True for non-zero entries. False where the absolute 
        value of the entry is less than `epsilon`.
    '''
    zero_real = np.abs(arr) < epsilon
    return ~zero_real


def compute_mask_ok(arr, epsilon=1.0E-05):
    '''
    Create a mask of the valid (finite, non-zero) pixels in arr.

    TODO - after development of the RSLC QA code is complete,
    check that this function is used. If not, delete.

    Parameters
    ----------
    arr : array_like
        The input array
    epsilon : float, optional
        Tolerance for if an element in `arr` is considered 'zero'

    Returns
    -------
    mask_ok : array_like
        Array with same shape as `arr`.
        True for valid entries. Valid entries are finite values that are
        not approximately equal to zero.
        False for entries that have a nan or inf in either the real
        or imaginary component, or a zero in both real and imag components.

    See Also
    --------
    numpy.isfinite : Can be used to compute `finite_entries`
    utils.compute_non_zero_mask : Can be used to compute `non_zero`
    '''

    finite_mask = np.isfinite(arr)
    non_zero_mask = compute_non_zero_mask(arr, epsilon)
    mask_ok = finite_mask & non_zero_mask

    return mask_ok

__all__ = nisarqa.get_all(__name__, objects_to_skip)
