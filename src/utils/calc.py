import numpy as np
from utils import input_verification as iv

def arr2pow(arr):
    """
    Compute power in linear units of the input array.

    Power is computed as magnitude squared.

    Parameters
    ----------
    arr : array_like
        Complex-valued input array.

    Returns
    -------
    power : numpy.ndarray
        Power of input array, with the same shape as the input.
    """

    power = arr.real**2 + arr.imag**2

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
    power_db : scalar or numpy.ndarray
        Output in decibels, with the same shape as the input.
    """
    return 10.0*np.log10(power)
