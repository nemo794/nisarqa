from __future__ import annotations

import itertools
import re
from collections.abc import Iterable, Mapping
from string import Template
from typing import Tuple

# List of objects from the import statements that
# should not be included when importing this module
import nisarqa
from nisarqa import pol_options, subswaths_options

objects_to_skip = nisarqa.get_all(name=__name__)


def check_path_for_substrings(
    path: str, all_options: Iterable[str], valid_options: Iterable[str]
) -> bool:
    """
    Compare a path to a rule and return its accepted/rejected status.

    Parameters
    ----------
    path : str
        The path to check.
    all_options : Iterable[str]
        A list of all substrings to check for. If none are found, the string will be
        accepted by default.
    valid_options : Iterable[str]
        The subset of all_options which will be accepted if found.

    Returns
    -------
    bool
        True if accepted, False if rejected.
    """
    return (
        match_substrings(
            path=path, all_options=all_options, valid_options=valid_options
        )
        is not None
    )


def match_substrings(
    path: str, all_options: Iterable[str], valid_options: Iterable[str]
) -> str | None:
    """
    Find substrings in a path and return a valid substring if it is found.

    Parameters
    ----------
    path : str
        The path to check.
    all_options : Iterable[str]
        A list of all substrings to check for. If none are found, the string will be
        accepted by default.
    valid_options : Iterable[str]
        The subset of all_options which will be accepted if found.

    Returns
    -------
    str | None:
        Will return a string with the accepted substring if a valid substring is found.
        Will return None if a substring is found that is not valid.
        Will return an empty string if no substring was found.
    """
    for option in all_options:
        # For cases in which one valid option is a substring of an invalid one or vice
        # versa, we avoid false positives/negatives by ensuring that the entire dataset
        # matches the option with surrounding slash characters or end of string.
        option_re = re.compile(f"(.*/{option}/.*|.*/{option}$)")
        if option_re.match(path):
            if option in valid_options:
                return option
            # If an option is in all options but not valid options and matches the path,
            # this is a rejected path. Return None.
            return None
    # If a path did not match any of the options, it is trivially accepted.
    return ""


def check_path_by_freq_pol(
    path: str,
    all_freqs: Iterable[str],
    all_pols: Iterable[str],
    valid_freq_pols: Mapping[str, Iterable[str]],
) -> bool:
    """
    Check if a given path conforms to the frequencies and polarities associated with
    the product.

    Parameters
    ----------
    path : str
        The path to check.
    all_freqs : Iterable[str]
        A set containing all possible frequencies for this product.
        Defaults to the policy default frequency set.
    all_pols : Iterable[str]
        A set containing all possible polarizations for this product.
        Defaults to the policy default polarization set.
    valid_freq_pols : Mapping[str, Iterable[str]]
        A mapping of valid frequencies to their valid polarities for this product.

    Returns
    -------
    bool
        True if the path is accepted, False if it is rejected.
    """
    valid_freqs = valid_freq_pols.keys()
    check_result = match_substrings(
        path=path, all_options=all_freqs, valid_options=valid_freqs
    )

    if check_result is None:
        return False
    if check_result == "":
        return True

    # If a frequency was found, then the result of the check will be the name
    # of the frequency dataset.
    frequency = check_result
    # Get the part of the path that comes after the frequency.
    remaining_string = path.split(frequency, maxsplit=1)[1]
    # Get all polarizations associated with the identified frequency dataset.
    valid_pols = valid_freq_pols[frequency]

    # Return the results of a substring check for the polarizations over
    # the part of the path that follows the frequency.
    return check_path_for_substrings(
        path=remaining_string, all_options=all_pols, valid_options=valid_pols
    )


def check_path(
    path: str,
    all_freqs: Iterable[str],
    all_pols: Iterable[str],
    all_layers: Iterable[str],
    all_subswaths: Iterable[str],
    valid_freq_pols: Mapping[str, Iterable[str]],
    valid_layers: Iterable[str] | None = None,
    valid_subswaths: Iterable[str] | None = None,
    rule_exceptions: Iterable[str] = None,
) -> bool:
    """
    Accept or reject a path by frequency, polarity, layer, and subswath rules.

    Parameters
    ----------
    paths : Iterable[str]
        The set of paths to check.
    all_freqs : Iterable[str]
        A set containing all possible frequencies for this product.
        Defaults to the policy default frequency set.
    all_pols : Iterable[str]
        A set containing all possible polarizations for this product.
        Defaults to the policy default polarization set.
    all_layers : Iterable[str]
        A set of all possible layer datasets for this product.
    all_subswaths : Iterable[str]
        A set of all possible subswath datasets for this product.
    valid_freq_pols : Mapping[str, Iterable[str]]
        A mapping of valid frequencies to their valid polarities for this
        product.
    valid_layers : Iterable[str] | None, optional
        A set of all valid layer datasets for this products, or None. If None,
        layer datasets will not be checked. Defaults to None.
    valid_subswaths : Iterable[str] | None, optional
        A set of all valid subswath datasets for this products, or None. If None,
        subswath datasets will not be checked. Defaults to None.
    rule_exceptions : Iterable[str] | None, optional
        A set of regex patterns to consider valid regardless of other rules.
        May not contain placeholder datasets. If None, no exceptions will be
        checked. Defaults to None.

    Returns
    -------
    bool
        True if the path is accepted within the rules, False if not.
    """
    if rule_exceptions is None:
        rule_exceptions = []
    
    for exception_path in rule_exceptions:
        if re.match(exception_path, path):
            return True

    if not check_path_by_freq_pol(
        path=path,
        all_freqs=all_freqs,
        all_pols=all_pols,
        valid_freq_pols=valid_freq_pols,
    ):
        return False

    if valid_subswaths is not None:
        if not check_path_for_substrings(
            path=path,
            all_options=all_subswaths,
            valid_options=valid_subswaths,
        ):
            return False

    if valid_layers is not None:
        if not check_path_for_substrings(
            path=path,
            all_options=all_layers,
            valid_options=valid_layers,
        ):
            return False

    return True


def check_paths(
    paths: Iterable[str],
    valid_freq_pols: Mapping[str, Iterable[str]],
    all_freqs: Iterable[str] = [
        f"frequency{opt}" for opt in nisarqa.NISAR_FREQS
    ],
    all_pols: Iterable[str] = pol_options(),
    all_layers: Iterable[str] = [nisarqa.NISAR_LAYERS],
    all_subswaths: Iterable[str] = subswaths_options(),
    valid_layers: Iterable[str] | None = None,
    valid_subswaths: Iterable[str] | None = None,
    rule_exceptions: Iterable[str] = [],
) -> Tuple[set[str], set[str]]:
    """
    Check a set of HDF5 paths against frequency/polarization, layer, and subswath rules
    and return sets of accepted and rejected paths.

    Parameters
    ----------
    paths : Iterable[str]
        The set of paths to check.
    valid_freq_pols : Mapping[str, Iterable[str]]
        A mapping of valid frequencies to their valid polarities for this product.
    all_freqs : Iterable[str]
        A set containing all possible frequencies for this product.
        Defaults to the policy default frequency set.
    all_pols : Iterable[str]
        A set containing all possible polarizations for this product.
        Defaults to the policy default polarization set.
    all_layers : Iterable[str]
        A set of all possible layer datasets for this product. Defaults to the policy
        default layer set.
    all_subswaths : Iterable[str]
        A set of all possible subswath datasets for this product. Defaults to the policy
        default subswath set.
    valid_layers : Iterable[str] | None, optional
        A set of all valid layer datasets for this products, or None. If None, layer
        datasets will not be checked. Defaults to None.
    valid_subswaths : Iterable[str] | None, optional
        A set of all valid subswath datasets for this products, or None. If None, subswath
        datasets will not be checked. Defaults to None.
    rule_exceptions : Iterable[str], optional
        A set of regex patterns to consider valid regardless of other rules. May contain
        placeholder datasets. Defaults to an empty list.

    Returns
    -------
    accepted: set[str]
        A sorted set of all paths accepted by the rulesets.
    rejected: set[str]
        A sorted set of all paths rejected by the rulesets.
    """
    rule_exceptions = process_excepted_paths(
        valid_freq_pols=valid_freq_pols,
        unprocessed_rule_exceptions=rule_exceptions,
    )

    accepted = set()
    rejected = set()

    for path in paths:
        accepted.add(path) if check_path(
            path=path,
            all_freqs=all_freqs,
            all_pols=all_pols,
            all_layers=all_layers,
            all_subswaths=all_subswaths,
            valid_freq_pols=valid_freq_pols,
            valid_layers=valid_layers,
            valid_subswaths=valid_subswaths,
            rule_exceptions=rule_exceptions,
        ) else rejected.add(path)

    return accepted, rejected


def process_excepted_paths(
    valid_freq_pols: Mapping[str, Iterable[str]],
    unprocessed_rule_exceptions: Iterable[str],
):
    """
    Replace placeholders in the given list of excepted paths with appropriate
    replacements.

    For strings given in the `unprocessed_rule_exceptions` list:
        Any string with "$freq" will require that the string still match one
            of the frequencies given in `valid_freq_pols` at that position.
        Any string with "$pol" will require that the string still match
            one of the frequency/polarity associations given in `valid_freq_pols`.
        Any string with "$lin_pol" will match all circular polarizations.
        Any string with "$circ_pol" will match all circular polarizations.

    Parameters
    ----------
    valid_freq_pols : Mapping[str, Iterable[str]]
        A mapping of valid frequencies to their valid polarities for this product.
    unprocessed_rule_exceptions : Iterable[str]
        The set of unprocessed rule exception strings to format.

    Returns
    -------
    set[str]
        The formatted exception strings.
    """
    rule_exceptions: set[str] = set()

    permutations = []
    for frequency in valid_freq_pols.keys():
        for pol in valid_freq_pols[frequency]:
            permutations.append((frequency, pol))

    permutations_2 = itertools.product(
        permutations, nisarqa.linear_pols(), nisarqa.circular_pols()
    )
    for perm in permutations_2:
        (freq, pol), lin_pol, circ_pol = perm
        for exception_str in unprocessed_rule_exceptions:
            reformatted_exception = Template(exception_str).safe_substitute(
                freq=freq,
                pol=pol,
                lin_pol=lin_pol,
                circ_pol=circ_pol,
            )
            rule_exceptions.add(reformatted_exception)
    return rule_exceptions


__all__ = nisarqa.get_all(__name__, objects_to_skip)
