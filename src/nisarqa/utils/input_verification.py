from __future__ import annotations

import os
import re
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


def _get_nisar_integer_seconds_template() -> str:
    """Return integer-seconds datetime template string per NISAR conventions."""
    # NISAR conventions require the "T"
    return "YYYY-mm-ddTHH:MM:SS"


def _get_nisar_integer_seconds_regex(require_t: bool = True) -> str:
    """Return integer-seconds datetime regex string, per NISAR conventions."""
    if require_t:
        # NISAR conventions require the "T"
        return r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    else:
        return r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"


def get_nisar_datetime_format_conventions(precision: str) -> tuple[str, re.Pattern]:
    """
    Return the datetime template string and regex Pattern for a given precision.

    The requested pattern will follow the NISAR conventions agreed to by ADT.

    Parameters
    ----------
    precision : str
        Precision for the requested datetime template string.
        Only "seconds" or "nanoseconds" supported.

    Returns
    -------
    template_string : str
        Template string for the requested datetime format, which can be used
        for log messages. The mapping of `precision` to `template_string` is:
            "seconds"     =>  "YYYY-mm-ddTHH:MM:SS"
            "nanoseconds" =>  "YYYY-mm-ddTHH:MM:SS.sssssssss"
    regex : re.Pattern
        Regex Pattern for the requested parameters, corresponding to
        the patterns noted in `template_string` docstring.
        The Pattern will require that there are no additional characters
        before or after the datetime string itself.
        If `precision` is "seconds", the Pattern enforces no decimel values.

    Notes
    -----
    Per NISAR ADT on 2024-07-25, the NISAR convention should be:
        For time points that don't require sub-second precision (e.g. reference
        epochs and processing datetimes) use integer seconds only, like this:
            “YYYY-mm-ddTHH:MM:SS”
        For all other time points, like `zeroDopplerStartTime` and
        `zeroDopplerEndTime`, use nanosecond precision.
    """

    template_integer_sec = _get_nisar_integer_seconds_template()  # includes T
    regex_integer_sec = _get_nisar_integer_seconds_regex(require_t=True)

    if precision == "seconds":
        template = template_integer_sec
        exclude_decimals = "(?!\.\d)"
        regex = f"^{regex_integer_sec}{exclude_decimals}$"

    elif precision == "nanoseconds":
        template = f"{template_integer_sec}.sssssssss"
        require_decimals = "\.\d{9}"
        regex = f"^{regex_integer_sec}{require_decimals}$"
    else:
        raise ValueError(f"{precision=}, must be 'seconds' or 'nanoseconds'.")

    return template, re.compile(regex)


def verify_nisar_datetime_template_string(
    datetime_template_string: str,
    dataset_name: str,
    precision: str | None = None,
) -> bool:
    """
    Compare a string against the datetime template convention for NISAR.

    Logs if there is a discrepancy.

    Parameters
    ----------
    datetime_template_string : str
        Datetime template string. Should not contain additional text.
        Example that passes: "YYYY-mm-ddTHH:MM:SS".
        Example that fails: "seconds since YYYY-mm-ddTHH:MM:SS".
    dataset_name : str
        Name of dataset associated with `datetime_template_string`; ideally
        this is the full HDF5 path to the dataset. (Used for logging.)
    precision : str or None, optional
        Precision for the requested datetime template string.
        Must be one of: "seconds" or "nanoseconds" or None.
        If None, template string must match either the integer seconds
        format or the nanoseconds format.
        Defaults to None.

    Returns
    -------
    passes : bool
         True if `datetime_template_string` matches the NISAR convention,
         False if not.
    """
    int_sec_format, _ = get_nisar_datetime_format_conventions(
        precision="seconds"
    )
    ns_format, _ = get_nisar_datetime_format_conventions(
        precision="nanoseconds"
    )

    if precision == "seconds":
        options = (int_sec_format,)
    elif precision == "nanoseconds":
        options = (ns_format,)
    elif precision is None:
        options = (int_sec_format, ns_format)
    else:
        raise ValueError(
            f"{precision=!r}, must be 'seconds', 'nanoseconds', or None."
        )

    if datetime_template_string not in options:
        nisarqa.get_logger().error(
            f"The given datetime template string {datetime_template_string!r}"
            f" does not match specified NISAR datetime template convention in"
            f" {options} - Dataset {dataset_name}"
        )
        return False
    return True


def verify_nisar_datetime_string_format(
    datetime_str: str, precision: str, dataset_name: str
) -> bool:
    """
    Verify that a string is a datetime string and matches NISAR conventions.

    Discrepancies will be logged as an error.

    Parameters
    ----------
    datetime_str : str
        The datetime string to be checked. Should not contain additional text.
        Example that passes: "2023-10-31T11:59:32".
        Example that fails: "seconds since 2023-10-31T11:59:32".
    precision : str
        Expected precision of the datetime string.
        Must be one of: "seconds" or "nanoseconds".
    dataset_name : str
        Name of the dataset associated with `datatime_str`. (Used for logging.)

    Returns
    -------
    passes : bool
        True if `datetime_str` conforms to the NISAR convention; False if not.
    """
    log = nisarqa.get_logger()
    template, regex = get_nisar_datetime_format_conventions(precision=precision)

    if not regex.match(datetime_str):
        log.error(
            f"HDF5 datetime string '{datetime_str}' does not conform to"
            f" NISAR datetime format convention '{template}' -"
            f" Dataset {dataset_name}"
        )
        return False
    return True


def contains_datetime_template_substring(input_str: str) -> bool:
    """
    Check if input string contains a generic datetime template substring.

    This function does not require that the substring conforms to NISAR
    conventions.

    Parameters
    ----------
    input_str : str
        The string to be parsed, possibly containing a datetime
        template substring like "YYYY-mm-ddTHH:MM:SS" or
        "YYYY-mm-ddTHH:MM:SS.sssssssss". Any number of decimals is allowed,
        and the "T" can be a space (" ").

    Returns
    -------
    contains_dt : bool
        True if there is at least one datetime template substring contained
        in the input string. False if not.
    """

    template_sec = _get_nisar_integer_seconds_template()

    # convert to a regex to make the T optional...
    regex = template_sec.replace("T", "[T ]")
    # ...and allow for (optional) decimals (decimals denoted by lowercase "s"s)
    pattern = re.compile(f"{regex}(?:\.s+)?")

    match = pattern.search(input_str)

    return False if match is None else True


def extract_datetime_template_substring(
    input_str: str, dataset_name: str
) -> str:
    """
    Extract a generic datetime template substring from input string.

    The substring should consist of "YYYY-mm-ddTHH:MM:SS" at minimum, but
    any number of decimal seconds (denoted by ".sss") are allowed.
    This function does not verify that the substring conforms to NISAR
    conventions.

    Parameters
    ----------
    input_str : str
        The string to be parsed. Must contain exactly one datetime
        template substring, with a format like "YYYY-mm-ddTHH:MM:SS" or
        "YYYY-mm-ddTHH:MM:SS.sssssssss". (Any number of decimal digits is allowed,
        and the "T" can be a space (" ").)
    dataset_name : str
        Name of the dataset associated with `input_str`. (Used for logging.)

    Returns
    -------
    datetime_str : str
        The datetime template substring contained in the input string.

    Raises
    ------
    ValueError
        If `input_str` contains zero or more than one datetime strings.
        As of July 2024, NISAR product specs contain no fields
        with multiple datetime strings; handling these edge cases would
        cause unncessary code complexity.
    """

    template_sec = _get_nisar_integer_seconds_template()

    # convert to a regex to make the T optional...
    regex = template_sec.replace("T", "[T ]")
    # ...and allow for (optional) decimals (decimals denoted by lowercase "s"s)
    pattern = re.compile(f"{regex}(?:\.s+)?")

    matches = re.findall(pattern, input_str)

    # input string should contain exactly one instance of a datetime template
    if len(matches) > 1 or len(matches) == 0:
        raise ValueError(
            f"{input_str=}, but must contain exactly one datetime template"
            f" string. Dataset: {dataset_name}"
        )

    return matches[0]


def contains_datetime_value_substring(input_str: str) -> bool:
    """
    Check if input string contains a datetime substring.

    This function does not require that the substring conforms to NISAR
    conventions.

    Parameters
    ----------
    input_str : str
        The string to be parsed, possibly containing a datetime
        substring that follows the format like "YYYY-mm-ddTHH:MM:SS"
        or "YYYY-mm-ddTHH:MM:SS.sssssssss". Any number of decimal digits is allowed,
        and the "T" can be a space (" ").
        Example: "seconds since 2023-10-31T11:59:32.123"
        Example: "seconds since 2023-10-31 11:59:32"

    Returns
    -------
    contains_dt : bool
        True if there is at least one datetime substring contained in the
        input string. False if not.
    """

    regex_int = _get_nisar_integer_seconds_regex(require_t=False)

    # Include optional decimals group in the regex
    regex = re.compile(f"{regex_int}(?:\.\d+)?")

    matches = re.findall(regex, input_str)

    return True if len(matches) > 0 else False


def extract_datetime_value_substring(input_str: str, dataset_name: str) -> str:
    """
    Extract a generic datetime substring from input string.

    This should follow the format "YYYY-mm-ddTHH:MM:SS" at minimum, but
    any number of decimal seconds are allowed and the "T" can be a space (" ").
    This function does not verify that the substring conforms to NISAR
    conventions.

    Parameters
    ----------
    input_str : str
        The string to be parsed. Must contain exactly one substring in a
        format like "YYYY-mm-ddTHH:MM:SS" or "YYYY-mm-ddTHH:MM:SS.sssssssss".
        (Any number of decimal digits is allowed, and the "T" can be a space (" ").)
        Example: "seconds since 2023-10-31T11:59:32.123"
    dataset_name : str
        Name of the dataset associated with `input_str`. (Used for logging.)

    Returns
    -------
    datetime_str : str
        The datetime substring contained in the input string,
        e.g. "2023-10-31T11:59:32".

    Raises
    ------
    ValueError
        If `input_str` contains zero or multiple datetime strings.
        As of July 2024, NISAR product specs contain no fields
        with multiple datetime strings; handling these edge cases would
        cause unncessary code complexity.
    """
    regex_int = _get_nisar_integer_seconds_regex(require_t=False)

    # Include optional decimals group in the regex
    regex = re.compile(f"{regex_int}(?:\.\d+)?")

    matches = re.findall(regex, input_str)

    # input string should contain exactly one instance of a datetime string
    if len(matches) > 1 or len(matches) == 0:
        raise ValueError(
            f"{input_str=!r}, but should contain exactly one datetime string."
            f" Dataset: {dataset_name}"
        )

    return matches[0]


def verify_datetime_string_matches_template(
    *, dt_value_str: str, dt_template_str: str, dataset_name: str
) -> bool:
    """
    Compare the format of a datetime string against a datetime template string.

    This is a generic function; the template string should contain
    "YYYY-mm-ddTHH:MM:SS" at minimum, but any number of decimal seconds
    (denoted by ".sss") are allowed.
    This function does not verify conformance to NISAR conventions.

    Parameters
    ----------
    dt_value_str : str
        Datetime string which should follow the format of `dt_template_str`.
        Should not contain additional text.
        Examples: "2023-10-31T11:59:32" or "2023-10-31T11:59:32.123456789".
    dt_template_str : str
        Datetime template string, with a
        format like "YYYY-mm-ddTHH:MM:SS" or "YYYY-mm-ddTHH:MM:SS.sssssssss".
        (Any number of decimal digits is allowed.) Should not contain additional text.
    dataset_name : str
        Name of dataset associated with the input strings. (Used for logging.)

    Returns
    -------
    passes : bool
        True if the format of `dt_value_str` matches the format specified in
        `dt_template_str`. False if not.

    See Also
    --------
    verify_datetime_matches_template_with_addl_text
        Wrapper around this function, which compares strings with datetimes
        which may also contain prefixes and suffixes to that datetime.
    """
    log = nisarqa.get_logger()

    # Compute number of decimals in template string
    template_sec = _get_nisar_integer_seconds_template()
    pattern = f"^{template_sec}(\.s+)?$"

    template_match = re.search(pattern, dt_template_str)
    if template_match is None:
        raise ValueError(
            f"{dt_template_str=}, must contain this pattern: '{pattern}'."
        )

    n_decimals = len(template_match[0].lstrip(template_sec + "."))

    # Check that the value string uses the same format as the template string
    regex_sec = _get_nisar_integer_seconds_regex()
    pattern = f"^{regex_sec}\.\d{{{n_decimals}}}$"

    val_match = re.search(pattern, dt_value_str)

    if val_match is not None:
        return True
    else:
        log.error(
            f"Provided dateime string is {dt_value_str!r}, but must match the"
            f" template format: '{dt_template_str}'. Dataset: {dataset_name}"
        )
        return False


def verify_datetime_matches_template_with_addl_text(
    *, dt_value_str: str, dt_template_str: str, dataset_name: str
) -> bool:
    """
    Compare string with datetime substring against a template string.

    This is a wrapper around `verify_datetime_string_matches_template()`
    to account for prefixes and suffixes.

    Parameters
    ----------
    dt_value_str : str
        String which must contain exactly one datetime substring. Ideally, it should
        conform to the format of `dt_template_str` (including prefix and suffix).
            Example 1: "seconds since 2023-10-31T11:59:32"
            Example 2: "2023-10-31T11:59:32.123456789"
    dt_template_str : str
        String containing exactly one datetime template substring. May contain prefix
        or suffix to the datetime. Must contain the substring
        "YYYY-mm-ddTHH:MM:SS" or "YYYY-mm-ddTHH:MM:SS.sssssssss"
        (Any number of decimal digits is allowed, and the "T" can be a space (" ").)
            Example 1: "seconds since YYYY-mm-ddTHH:MM:SS"
            Example 2: "YYYY-mm-ddTHH:MM:SS.sssssssss"
    dataset_name : str
        Name of dataset associated with the input strings. (Used for logging.)

    Returns
    -------
    matches : bool
        True if `dt_template_str` contains a datetime template,
        `dt_value_str` contains a datetime string matching the format
        specified in `dt_template_str`, and both of their prefixes and
        suffixes to the datetime portion are identical.
        False otherwise.
    """
    dt_val = extract_datetime_value_substring(
        input_str=dt_value_str, dataset_name=dataset_name
    )

    dt_tmpl = extract_datetime_template_substring(
        input_str=dt_template_str, dataset_name=dataset_name
    )
    if not dt_val:
        raise ValueError(
            f"{dt_value_str=}, must contain a datetime string."
            f" Dataset: {dataset_name}"
        )

    if not dt_tmpl:
        raise ValueError(
            f"{dt_template_str=}, must contain a datetime template string."
            f" Dataset: {dataset_name}"
        )

    # Split strings into (ideally) prefix, datetime, and suffix
    dt_val_split = dt_value_str.split(dt_val)
    dt_tmpl_split = dt_template_str.split(dt_tmpl)

    if len(dt_val_split) != len(dt_tmpl_split):
        return False
    else:
        for val, tmpl in zip(dt_val_split, dt_tmpl_split):
            if tmpl == dt_tmpl:
                if not verify_datetime_string_matches_template(
                    dt_value_str=val,
                    dt_template_str=tmpl,
                    dataset_name=dataset_name,
                ):
                    return False
            elif val != tmpl:
                return False

    return True


__all__ = nisarqa.get_all(__name__, objects_to_skip)
