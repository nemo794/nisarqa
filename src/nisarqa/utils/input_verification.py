import os
import warnings

import h5py
import numpy as np

import nisarqa

objects_to_skip = nisarqa.get_all(__name__)


def verify_isce3_boolean(ds: h5py.Dataset) -> None:
    """
    Verify if a boolean value uses the correct ISCE3 convention.

    By convention, ISCE3 uses 'True' or 'False' fixed-length byte strings
    to represent boolean values.

    Parameters
    ----------
    ds : h5py.Dataset
        Dataset to be verified. This should represent a boolean quantity.
    """
    log = nisarqa.get_logger()

    verify_str_meets_isce3_conventions(ds)

    data = nisarqa.byte_string_to_python_str(ds[()])

    if data not in ("True", "False"):
        errmsg = (
            f"Dataset `{ds.name}` represents a boolean but contains the value"
            f" {ds[()]}. By ISCE3 convention it must be a byte string"
            " b'True' or b'False'."
        )
        log.warning(errmsg)


def verify_str_meets_isce3_conventions(ds: h5py.Dataset) -> None:
    """
    Verify that input dataset contains fixed-length byte string(s).

    If a dataset does not meet ISCE3 conventions, then a warning is raised.
    This function also checks for correct string lengths and null terminator
    characters that do not meet ISCE3 conventions.

    Parameters
    ----------
    ds : h5py.Dataset
        A dataset whose value should be a fixed-length string or a 1D array
        of fixed-length byte strings.

    Notes
    -----
    As of Nov 2023, the current ISCE3 conventions are:
    * Scalar strings:
        A fixed-length byte string with no null terminators.
        Ex: The string "lulu" should be b'lulu' with dtype |S4.
        (The "S" in "|S4" signifies it is a byte string.)
    * 1D Array of same-length strings:
        NumPy array of fixed-length byte strings with no null terminators.
        Ex: The array numpy.bytes_(["A", "B"]) should have dtype |S1
    * 1D Array of mixed-length strings:
        NumPy array of fixed-length byte strings with no null terminator
        characters on the longest string, and null character padding
        on the shorter strings.

        Ex: The array numpy.bytes_(["A", "BB"]) has dtype |S2.
        The first element ("A") has 1 null character padding,
        and the second element ("BB") has no null padding.
        In the command line, `h5dump` can be used to check for null padding.
        In practice, when h5py or NumPy access each element, they
        check its shape and provide a scalar with the null padding removed.
        This means that when accessed, the first element's scalar has
        dtype |S1 instead of |S2 (which is the dtype of the array), and
        the second element's scalar has dtype |S2.
    """
    log = nisarqa.get_logger()

    # Check if the dtype is byte string.
    if not np.issubdtype(ds.dtype, np.bytes_):
        errmsg = (
            f"Dataset `{ds.name}` has dtype `{ds.dtype}`, but must be a"
            " sub-dtype of `numpy.bytes_` (i.e. a byte string)."
        )
        log.warning(errmsg)

        # Return early; none of the other verification checks can be used
        return

    if ds.shape == ():
        # Dataset is a scalar string
        verify_length_of_scalar_byte_string(ds)
    elif len(ds.shape) == 1:
        # Dataset is a 1D array of strings
        verify_1D_array_of_byte_strings(ds)
    else:
        # multi-dimensional arrays of strings not currently supported.
        errmsg = (
            f"Shape of Dataset `{ds.name}` is {ds.shape}, but ISCE3 does not"
            " generate arrays of strings with more than one dimension."
        )
        log.warning(errmsg)


def verify_byte_string(my_string: np.bytes_) -> None:
    """
    Verify if a byte string is, in fact, a byte string and not a Python string.

    Parameters
    ----------
    my_string : numpy.bytes_
        Byte string to be verified.
    """
    log = nisarqa.get_logger()

    if not isinstance(my_string, np.bytes_):
        errmsg = (
            f"Input {my_string} has type {type(my_string)}, but must be an"
            " instance of `numpy.bytes_` (i.e. a byte string)."
        )
        log.warning(errmsg)


def verify_length_of_scalar_byte_string(ds: h5py.Dataset) -> None:
    """
    Verify the length of a scalar byte string matches its dtype.

    By ISCE3 convention, there should not be any null terminator characters
    for scalar, fixed-length byte strings.
    For example, the byte string b'lulu' should have a dtype of |S4.
    But, if it had a single null terminator, it would have dtype of |S5.

    Parameters
    ----------
    ds : h5py.Dataset
        Dataset to be verified. This should represent a scalar string.
    """
    log = nisarqa.get_logger()

    # This function can only check scalar byte strings.
    verify_byte_string(ds[()])
    if not ds.shape == ():
        errmsg = f"Dataset `{ds.name}` must be a scalar dataset."
        raise ValueError(errmsg)

    # Get the official string length from the dtype
    official_strlen = int(str(ds.dtype).split("S")[-1])

    # Get the actual number of meaningful characters for this given string.
    # (When reading in datasets, h5py "magically" removes null terminators.)
    actual_strlen = int(str(ds[()].dtype).split("S")[-1])

    if official_strlen > actual_strlen:
        # HDF5 truncates byte strings that are longer than the dtype, so it is
        # not possible for `official_strlen` to be < `actual_strlen`.
        errmsg = (
            f"Dataset `{ds.name}` contains trailing null characters. It has"
            f" dtype `{ds.dtype}`, but should have dtype `{ds[()].dtype}`."
        )
        log.warning(errmsg)


def verify_1D_array_of_byte_strings(ds: h5py.Dataset) -> None:
    """
    Verify if a Dataset is properly formatted as a 1D array of byte strings.

    Parameters
    ----------
    ds : h5py.Dataset
        Dataset to be verified. This should represent a 1D array of byte
        strings.
    """
    # Verify input is a 1D array. (This function only supports 1D arrays.)
    if len(ds.shape) != 1:
        errmsg = f"Dataset `{ds.name}` has {ds.shape}, but must be a 1D array."
        raise ValueError(errmsg)

    # Check that the array only contains byte strings.
    arr = ds[()]
    for idx in range(ds.shape[0]):
        my_string = arr[idx]
        if not my_string:
            # empty string
            errmsg = (
                f"Dataset `{ds.name}` is a 1D array of strings, but string at"
                f" index {idx} is an empty string."
            )
            warnings.warn(errmsg)

        verify_byte_string(my_string)

    # Check that the reported maximum string length is actually the length
    # of the longest string in the array.
    official_strlen = int(str(ds.dtype).split("S")[-1])
    # `ds` is a 1D array, so no need to flatten it for iterating
    max_strlen = max(len(s) for s in ds[()])
    if official_strlen != max_strlen:
        errmsg = (
            f"Dataset `{ds.name}` has dtype `{ds.dtype}`, but the"
            f" longest string it contains is {max_strlen} characters,"
            f" so the Dataset should have dtype `|S{max_strlen}`."
        )
        warnings.warn(errmsg)


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


__all__ = nisarqa.get_all(__name__, objects_to_skip)
