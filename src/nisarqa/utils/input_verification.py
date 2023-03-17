import os

import nisarqa
import numpy as np

objects_to_skip = nisarqa.get_all(__name__)

def verify_valid_percentile(percentile):
    '''Verify that the input percentile is in range [0.0, 100.0].
    '''
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError(
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


def validate_is_file(filepath, parameter_name, extension=None):
    '''
    Raise exception if `filepath` is not a str.

    Parameters
    ----------
    filepath : str
        The filename (with path) to the file to be validated
    parameter_name : str
        The name of the variable; Displays in the Exception message.
    extension : str, optional
        If `extension` is provided, will check if `filepath` ends with
        `extension`.
        Examples: '.csv', '.h5'
    '''
    if not isinstance(filepath, str):
        raise TypeError(f'`{parameter_name}` must be a str')

    if not os.path.isfile(filepath):
        raise TypeError(
            f'`{parameter_name}` is not a valid file: {filepath}')

    if (extension is not None) and (not filepath.endswith(extension)):
        raise TypeError(
            f'`{parameter_name}` must end with {extension}: {filepath}')


__all__ = nisarqa.get_all(__name__, objects_to_skip)
