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


def counts2density(counts, bins):
    '''
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
    '''

    # Formula per numpy.histogram's documentation:
    density = counts / (np.sum(counts) * np.diff(bins))

    # Sanity check
    actual=np.sum(density * np.diff(bins))
    assert np.abs(actual - 1) < 1e-6

    return density


def normalize(arr):
    '''Normalize input array to range [0,1], ignoring
    any NaN values.
    '''
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    
    return (arr - arr_min) / (arr_max - arr_min)

    
__all__ = nisarqa.get_all(__name__, objects_to_skip)
