from __future__ import annotations

import os

import nisarqa
import numpy as np

from typing import Optional
from collections.abc import Sequence

objects_to_skip = nisarqa.get_all(__name__)


def verify_valid_percentile(percentile):
    """Verify that the input percentile is in range [0.0, 100.0]."""
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError(
            f"The percentile provided is {percentile} but must be "
            "in the range [0, 100]."
        )


def verify_complex_dtype(arr):
    """Check that input array has a complex datatype"""
    if not np.issubdtype(arr.dtype, np.complexfloating):
        raise TypeError(
            f"array is type {arr.dtype} but must be a subtype of complex."
        )


def verify_real_or_complex_dtype(arr):
    """Check that input array has a float or complex-float datatype"""
    if not (
        np.issubdtype(arr.dtype, np.floating)
        or np.issubdtype(arr.dtype, np.complexfloating)
    ):
        raise TypeError(
            f"array is type {arr.dtype} but must have a dtype that is a subtype"
            " of float or complex floating."
        )


def validate_is_file(filepath, parameter_name, extension=None):
    """
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
    """
    if not isinstance(filepath, str):
        raise TypeError(f"`{parameter_name}` must be a str")

    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"`{parameter_name}` is not a valid file: {filepath}"
        )

    if (extension is not None) and (not filepath.endswith(extension)):
        raise ValueError(
            f"`{parameter_name}` must end with {extension}: {filepath}"
        )


def validate_pair_of_numeric(
    input: Optional[Sequence[int]],
    param_name: str,
    min: Optional[int | float] = None,
    max: Optional[int | float] = None,
    none_is_valid_value: bool = False,
    monotonically_increasing: bool = False,
) -> None:
    """
    Validate a pair of numbers.

    Parameters
    ----------
    input : None or pair of int or float
        Sequence of two int or float value, or None.
    param_name : str
        Name of `input` parameter. Will be used for the error message.
    min, max : None or int or float, optional
        Min and Max values (respectively) for the values in `input`.
    none_is_valid_value : bool
        True if `None` is a valid value. Defaults to False.
    monotonically_increasing : bool
        True if `input_value[0]` must be less than `input_value[1]`.
        Defaults to False.
    """
    if input is None:
        if none_is_valid_value:
            return
        else:
            raise TypeError(
                f"`{param_name}` is None, but must be a pair of numeric."
            )

    if not isinstance(input, (list, tuple)):
        msg = f"`{param_name}` must be a sequence"
        if none_is_valid_value:
            msg += " or None."
        raise TypeError(msg)

    if not len(input) == 2:
        raise ValueError(f"{param_name}={input}; must have a length of two.")

    if not all(isinstance(e, (float, int)) for e in input):
        raise TypeError(
            f"{param_name}={input}; must contain only float or int."
        )

    if (min is not None) and (any((e < min) for e in input)):
        raise ValueError(
            f"{param_name}={input}; must be in range [{min}, {max}]."
        )

    if (max is not None) and (any((e > max) for e in input)):
        raise ValueError(
            f"{param_name}={input}; must be in range [{min}, {max}]."
        )

    if monotonically_increasing:
        if input[0] >= input[1]:
            raise ValueError(
                f"{param_name}={input}; values must be in"
                " monotonically increasing."
            )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
