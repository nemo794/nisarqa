import os
from datetime import datetime

import h5py
import numpy as np

import nisarqa

objects_to_skip = nisarqa.get_all(__name__)


def verify_isce3_boolean(ds: h5py.Dataset) -> bool:
    """
    Verify if a boolean value uses the correct ISCE3 convention.

    By convention, ISCE3 uses 'True' or 'False' fixed-length byte strings
    to represent boolean values.

    Parameters
    ----------
    ds : h5py.Dataset
        Dataset to be verified. This should represent a boolean quantity.

    Returns
    -------
    passes : bool
        True if `ds` uses the correct ISCE3 convention, False otherwise.
    """
    log = nisarqa.get_logger()

    if not verify_str_meets_isce3_conventions(ds):
        return False

    raw_data = ds[()]
    data = nisarqa.byte_string_to_python_str(raw_data)

    if data not in ("True", "False"):
        errmsg = (
            f"Dataset `{ds.name}` represents a boolean but contains the value"
            f" {raw_data!r}. By ISCE3 convention it must be a byte string"
            " b'True' or b'False'."
        )
        log.error(errmsg)
        return False
    return True


def verify_str_meets_isce3_conventions(ds: h5py.Dataset) -> bool:
    """
    Verify that input dataset contains fixed-length byte string(s).

    If a dataset does not meet ISCE3 conventions, then an error is logged.
    This function also checks for correct string lengths and null terminator
    characters that do not meet ISCE3 conventions.

    Parameters
    ----------
    ds : h5py.Dataset
        A dataset whose value should be a fixed-length string or a 1D array
        of fixed-length byte strings.

    Returns
    -------
    passes : bool
        True if `ds` uses the correct ISCE3 convention, False otherwise.

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
        log.error(errmsg)

        # Return early; none of the other verification checks can be used
        return False

    if ds.shape == ():
        # Dataset is a scalar string
        return verify_length_of_scalar_byte_string(ds)
    elif len(ds.shape) == 1:
        # Dataset is a 1D array of strings
        return verify_1D_array_of_byte_strings(ds)
    else:
        # multi-dimensional arrays of strings not currently supported.
        errmsg = (
            f"Shape of Dataset `{ds.name}` is {ds.shape}, but ISCE3 does not"
            " generate arrays of strings with more than one dimension."
        )
        log.error(errmsg)
        return False


def verify_byte_string(my_string: np.bytes_) -> bool:
    """
    Verify if a byte string is, in fact, a byte string and not a Python string.

    Parameters
    ----------
    my_string : numpy.bytes_
        Byte string to be verified.

    Returns
    -------
    passes : bool
        True if `my_string` uses the correct ISCE3 convention; False otherwise.
    """
    log = nisarqa.get_logger()

    if not isinstance(my_string, np.bytes_):
        errmsg = (
            f"Input {my_string} has type {type(my_string)}, but must be an"
            " instance of `numpy.bytes_` (i.e. a byte string)."
        )
        log.error(errmsg)
        return False
    return True


def verify_length_of_scalar_byte_string(ds: h5py.Dataset) -> bool:
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

    Returns
    -------
    passes : bool
        True if `ds` uses the correct ISCE3 convention, False otherwise.
    """
    log = nisarqa.get_logger()

    # This function can only check scalar byte strings.
    if not verify_byte_string(ds[()]):
        return False

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
        log.error(errmsg)
        return False
    return True


def verify_1D_array_of_byte_strings(ds: h5py.Dataset) -> bool:
    """
    Verify if a Dataset is properly formatted as a 1D array of byte strings.

    Parameters
    ----------
    ds : h5py.Dataset
        Dataset to be verified. This should represent a 1D array of byte
        strings.

    Returns
    -------
    passes : bool
        True if `ds` uses the correct ISCE3 convention, False otherwise.
    """
    # Verify input is a 1D array. (This function only supports 1D arrays.)
    if len(ds.shape) != 1:
        errmsg = f"Dataset `{ds.name}` has {ds.shape}, but must be a 1D array."
        raise ValueError(errmsg)

    log = nisarqa.get_logger()

    # Check that the array only contains byte strings.
    arr = ds[()]
    passes = True
    for idx in range(ds.shape[0]):
        my_string = arr[idx]
        if not my_string:
            # empty string
            # Reason why warning and not error: By changes introduced to RSLC
            # in PR: isce#1584, empty strings may be intentionally written to
            # some datasets. This means an empty string is valid in some cases,
            # but most datasets will not contain empty strings. Ref:
            # https://github-fn.jpl.nasa.gov/isce-3/isce/pull/1584/files#r19689
            msg = (
                f"Dataset `{ds.name}` is a 1D array of strings, but string at"
                f" index {idx} is an empty string."
            )
            log.warning(msg)

        passes &= verify_byte_string(my_string)

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
        log.error(errmsg)
        return False
    return passes


def verify_valid_percent(percent: float) -> None:
    """Verify that the input percent is in range [0.0, 100.0]."""
    if percent < 0.0 or percent > 100.0:
        raise ValueError(f"`{percent=}`, must be in the range [0.0, 100.0].")


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


def verify_datetime_format(datetime_str: str, prefix: str = "") -> None:
    """
    Verify that a string contains a properly-formatted datetime.

    Parameters
    ----------
    datetime_str : str
        A string which ends with a datetime. By NISAR convention for R4,
        this should have the format: 'YYYY-mm-ddTHH:MM:SS'.
    prefix : str, optional
        The beginning of `datetime_str`, which includes all characters
        before the datetime appears. Defaults to ''.
        Ex: If `datetime_str` is "seconds since %Y-%m-%dT%H:%M:%S", then
            prefix` should be "seconds since ", (including the space).

    Raises
    ------
    ValueError
        If `datetime_str` does not match the NISAR convention for
        datetime format. Note: If the format is only incorrect by a 'T',
        an InvalidNISARProductError will be raised instead.
    nisarqa.InvalidNISARProductError
        If the input string has nearly the correct datetime format, but
        is missing the 'T' between the date and the time.
        In this case, the string still contains useful information; the
        calling function can catch this exception, and handle accordingly.
    """
    format = f"{prefix}{nisarqa.NISAR_DATETIME_FORMAT_PYTHON}"
    human_format = f"{prefix}{nisarqa.NISAR_DATETIME_FORMAT_HUMAN}"

    try:
        # If this does not error, then string has correct format. Yay!
        datetime.strptime(datetime_str, format)
    except ValueError:
        # Old test datasets used the format: YYYY-mm-dd HH:MM:SS.
        # If so, that is still meaningful to human users.
        old_format = f"{prefix}%Y-%m-%d %H:%M:%S"
        try:
            datetime.strptime(datetime_str, old_format)
        except ValueError:
            # The format does not match the known formats.
            raise ValueError(
                f"The provided datetime string has format '{datetime_str}',"
                f" but should follow format '{human_format}'"
            )
        else:
            # The string still contains useful information, so raise a
            # special-case exception.
            raise nisarqa.InvalidNISARProductError(
                f"The provided datetime string is '{datetime_str}'. It uses"
                f" old datetime format: '{old_format}'. Please update to"
                f" new format: '{format}'."
            )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
