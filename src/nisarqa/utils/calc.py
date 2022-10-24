import numpy as np
import nisarqa

objects_to_skip = nisarqa.get_all(__name__)

def arr2pow(arr):
    '''
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
    '''

    power = np.abs(arr)**2

    return power


def pow2db(power):
    '''
    Convert a power quantity from linear units to decibels.

    Parameters
    ----------
    power : array_like
        Input in linear units.

    Returns
    -------
    power_db : numpy scalar or numpy.ndarray
        Output in decibels, with the same shape as the input.
    '''
    return 10.0*np.log10(power)


def nearest_odd_int(k):
    '''Compute the nearest odd integer to `k`
    '''
    result = int(np.floor(k))
    if result % 2 == 0:
        result = result + 1

    # Sanity Check
    assert result % 2 == 1, print('the result should be an odd value.')
    assert isinstance(result, int), print('the result should be an integer.')

    return result

__all__ = nisarqa.get_all(__name__, objects_to_skip)
