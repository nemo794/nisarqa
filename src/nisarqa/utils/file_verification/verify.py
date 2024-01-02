from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Optional

import numpy as np

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def verify_file(
    input_file: str | os.PathLike,
    product_type: str,
    product_spec_version: str,
    freq_pols: Mapping[str, Sequence[str]],
    layer_numbers: Optional[Sequence[int]] = None,
) -> None:
    """
    Verify input HDF5 file meets the XML product specifications document.

    Parameters
    ----------
    input_file : path-like
        Filepath to the input product.
    product_type : str
        One of: 'rslc', 'gslc', 'gcov', 'rifg', 'runw', 'gunw', 'roff', 'goff'.
    product_spec_version : str
        Contents of `../identification/productSpecificationVersion` from input
        file. Examples: "0.0.9", "1.0.0"
    freq_pols : Mapping[str, Sequence[str]]
        Dict of the expected polarizations for each frequency. Example:
            { "A" : ["HH", "HV], "B" : ["VV", "VH"] }
    layer_numbers : Sequence[int] or None, optional
        ROFF and GOFF products contain HDF5 Groups referred to as "layer number
        groups" (e.g. `../layer1/..`, `../layer3/..`).
        This parameter should be a sequence of integers in domain [1, 8] of
        the expected layer number groups' numbers in `input_file`.
        If the product type is not ROFF or GOFF, this should be set to None.
        Defaults to `None`.
        Examples: (1,), (1, 3), None
    """
    log = nisarqa.get_logger()

    # Validate inputs
    if product_type.lower() not in nisarqa.LIST_OF_NISAR_PRODUCTS:
        raise ValueError(
            f"`{product_type=}`; must one of: {nisarqa.LIST_OF_NISAR_PRODUCTS}"
        )

    if product_type in ("roff", "goff"):
        if layer_numbers is None:
            raise ValueError(
                f"{product_type=}, so `layer_numbers` cannot be None."
            )
        elif not isinstance(layer_numbers, Sequence):
            msg = f"`{layer_numbers=}` must be a sequence or None."
            raise TypeError(msg)
        elif not all(
            (isinstance(n, int) and (n > 0) and (n < 9)) for n in layer_numbers
        ):
            msg = (
                f"`{layer_numbers=}` must be a sequence of integers in range"
                " [1, 8], or None."
            )
            raise ValueError(msg)
    else:
        if layer_numbers is not None:
            raise ValueError(
                f"`{product_type=}` which should not have layer number groups,"
                f" but input parameter `{layer_numbers=}`. `layer_numbers` is"
                " only used in the case of ROFF and GOFF products. Please"
                " check that this is intended; if not, set `layer_numbers` to"
                " None."
            )

    # Here's how to get the list of all possible polarizations for this product
    # type (returns the covarience terms for GCOV):
    possible_pols = nisarqa.get_possible_pols(product_type=product_type)
    log.info(
        f"{product_type=}, which could include polarizations: {possible_pols}"
    )

    # To use any of the functions in QA, simply call from the nisarqa namespace.
    x = np.bytes_("hello")
    nisarqa.verify_byte_string(x)

    # TODO: Fancy XML-HDF5 input file verification. For now, just practise logging:
    # Rules of Thumb for logging:
    # debug   : pedantic debug statements
    # info    : to let the users know what's up, and status updates on where the code is in execution
    # warning : something slightly off-nominal happened for QA code, or
    #           something could be a cause for concern in the input products,
    #           but that thing could also be totally normal and expected.
    # error   : the input product does not meet the XML product spec, or
    #           some feature in QA needs to be skipped because of an error.
    #           These are items that PGE should always inform ADT about.
    # critical : Do not log as critical. Instead, raise an execption like
    #           usual, and QA code's main() function will catch and log it.

    log.info(f"TEMP: {input_file}")
    log.info(f"TEMP: {product_type}")
    log.info(f"TEMP: {product_spec_version}")
    log.info(f"TEMP: {freq_pols}")
    log.info(f"TEMP: {layer_numbers}")


__all__ = nisarqa.get_all(__name__, objects_to_skip)
