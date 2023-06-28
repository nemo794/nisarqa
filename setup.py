#!/usr/bin/env python3
# NISAR Project - Jet Propulsion Laboratory
# Copyright 2022, by the California Institute of Technology.
# ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
# This software may be subject to U.S. export control laws and regulations.

import glob
import os
import re

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
            "rm -vrf .scratch_dir ./build ./dist ./*.pyc ./*.tgz ./*.egg-info ./src/*.egg-info"
        )


def _get_version():
    """Returns the NISAR QA SAS software version from the
    file `src/nisarqa/__init__.py`

       Returns
       -------
       version : str
            NISAR QA SAS software version
    """
    init_file = os.path.join('src','nisarqa','__init__.py')

    with open(init_file, 'r') as f:
        text = f.read()

    # Get first match of the version number contained in the version file
    # This regex should match a pattern like: __version__ = '3.2.5', but it
    # allows for varying spaces, number of major/minor versions,
    # and quotation mark styles.
    p = re.search("__version__[ ]*=[ ]*['\"]\d+([.]\d+)*['\"]", text)

    # Check that the version file contains properly formatted text string
    if p is None:
        raise ValueError(
            f"__init__.py file {init_file} not properly formatted."
            " It should contain text matching e.g.` __version__ = '2.3.4'`")

    # Extract just the numeric version number from the string
    p = re.search("\d+([.]\d+)*", p.group(0))

    return p.group(0)    


setup(
    name="nisarqa",
    maintainer="NISAR ADT Team",
    maintainer_email="samantha.c.niemoeller@jpl.nasa.gov",
    description="NISAR ADT Quality Assurance",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Copyright by the California Institute of Technology."
    "All rights reserved",
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
            glob.glob(os.path.join("src", "parameters", "product_specs", "*.xml")),
        )
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "h5py",
        "pillow",
        "ruamel.yaml",
        "cycler",
        "matplotlib",
        "isce3"
    ],
    extras_require={
        "test": ["pytest"],
    },
    cmdclass={
        "clean": CleanCommand,
    },
)
