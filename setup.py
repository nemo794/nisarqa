#!/usr/bin/env python3
# NISAR Project - Jet Propulsion Laboratory
# Copyright 2022, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# This software may be subject to U.S. export control laws and regulations.

import glob
import os
import re
from pathlib import Path

from setuptools import Command, find_packages, setup


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Make sure to remove the .egg-info file
        os.system(
            "rm -vrf .scratch_dir ./build ./dist ./*.pyc ./*.tgz ./*.egg-info"
            " ./src/*.egg-info"
        )


def _get_version():
    """Returns the NISAR QA SAS software version from the
    file `src/nisarqa/__init__.py`

    Returns
    -------
    version : str
         NISAR QA SAS software version
    """
    init_file = "src/nisarqa/__init__.py"
    text = Path(init_file).read_text()

    # Get first match of the version number contained in the version file
    # This regex should match a pattern like: __version__ = '3.2.5', but it
    # allows for varying spaces, number of major/minor versions,
    # and quotation mark styles.
    # Note the regex capturing group to extract the version number components.
    match = re.search("__version__\s*=\s*['\"](\d+(\.\d+)*)['\"]", text)

    # Check that the version file contains properly formatted text string
    if match is None:
        raise ValueError(
            f"__init__.py file {init_file} not properly formatted."
            " It should contain text matching e.g.` __version__ = '2.3.4'`"
        )

    return match.group(1)


setup(
    name="nisarqa",
    maintainer="NISAR ADT Team",
    maintainer_email="samantha.c.niemoeller@jpl.nasa.gov",
    description="NISAR ADT Quality Assurance",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license=(
        "Copyright by the California Institute of Technology."
        "All rights reserved"
    ),
    url="http://nisar.jpl.nasa.gov",
    version=_get_version(),
    # Gather all packages located under `src`.
    # (A package is a directory containing an __init__.py file.)
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    test_suite="tests",
    entry_points={"console_scripts": ["nisarqa = nisarqa.__main__:main"]},
    data_files=[
        (
            "product_specs",
            glob.glob(
                os.path.join("src", "parameters", "product_specs", "*.xml")
            ),
        )
    ],
    # Build dependencies
    # Can be removed if/when setup.py is replaced with pyproject.toml
    # See: https://pip.pypa.io/en/stable/reference/build-system/#controlling-setup-requires
    setup_requires=[
        "setuptools",
    ],
    # Runtime dependencies
    install_requires=[
        "cycler",
        "h5py>=3",
        "isce3>=0.16",
        "matplotlib",
        "numpy>=1.20",
        "pillow",
        "python>=3.8",
        "ruamel.yaml",
        "shapely",
    ],
    extras_require={
        "test": ["pytest"],
    },
    cmdclass={
        "clean": CleanCommand,
    },
)
