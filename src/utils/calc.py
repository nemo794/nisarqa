import numpy as np
from utils import input_verification as iv

def arr2pow(arr, linear_units=True):
    """
    Compute power of the input array.

    Power is computed as magnitude squared. Defaults to linear units.

    Parameters
    ----------
    arr : array_like
        Complex-valued input array.
    linear_units : bool
        True to compute power in linear units, False for decibel units.
        Defaults to True.

    Returns
    -------
    power : numpy.ndarray
        Power of input array, with the same shape as the input.
    """

    power = arr.real**2 + arr.imag**2

    if not linear_units:
        # Convert to dB
        power = pow2db(power)

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
