import numpy as np


def verify_valid_percentile(percentile):
    '''Verify that the input percentile is in range [0.0, 100.0].
    '''
    if percentile < 0.0 or percentile > 100.0:
        raise TypeError(
            f'The percentile provided is {percentile} but must be '
            f'in the range [0, 100].'
        )


def verify_complex_dtype(arr):
    '''Check that input array has a complex datatype
    '''
    if not np.issubdtype(arr.dtype, np.complexfloating):
        raise TypeError(
            f'array is type {arr.dtype} but must be a subtype of complex.'
        )

def verify_real_or_complex_dtype(arr):
    '''Check that input array has a float or complex-float datatype
    '''
    if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.complexfloating)):
        raise TypeError(
            f'array is type {arr.dtype} but must have a dtype that is a subtype of float '
            f'or complex floating.'
        )

