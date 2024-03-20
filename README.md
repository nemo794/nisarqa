# QualityAssurance
Quality Assurance software for NISAR

For the upcoming [NISAR mission](https://nisar.jpl.nasa.gov/),
eight types of L1/L2 data products will be generated.
This Quality Assurance (QA) software is designed to look at the data products
produced one at a time. For each product, the QA code can:
- Log if the HDF5's structure matches the product spec
- Generate metrics, a PDF report, and a summary CSV describing the quality of the product
- Run CalTools processes on RSLC products
- Produce a browse image PNG with sidecar KML file for that product

# Minimum PreRequisites:
See `environment.yaml` for required packages.

# Operating Instructions:

## Installation

Step 1) Install the conda package manager.

Step 2) Clone the `QualityAssurance` repo and `cd` into the top level directory.
```
git clone git@github-fn.jpl.nasa.gov:NISAR-ADT/QualityAssurance.git
cd QualityAssurance
```

Step 3) Create a conda environment with the correct packages for QA.

By creating a new environment using the `environment.yaml` file, the required 
packages and their versions will all be resolved by conda from the conda-forge 
channel, and thus remain consistent with each other. (Mixing package managers 
has led to import errors during testing.) The `environment.yaml` defaults to 
naming the new conda environment `nisarqa`.
```
conda env create -f environment.yaml
conda activate nisarqa  # activate the new environment
```

Step 4) Install

For standard installation, run:
```
pip install --no-deps .
```

For develop mode, run:
```
pip install --no-deps -e .
```

To test installation, try:
```
nisarqa --version
nisarqa -h
nisarqa dumpconfig -h
```

Warning: Please install via `pip` and with the `--no-deps` flag. Installing
without the `--no-deps` flag fails due to the `isce3` dependency in 
`requirements.txt`, which is used by `pyproject.toml` during installation.
Otherwise, `isce3` would need to be removed from `requirements.txt`,
causing that dependency to be undocumented there.

## Running the QA Code

Run the QA code from the command line using a runconfig.yaml file.

Because the QA code is uniquely written for each product type, each product
type has a unique subcommand.

See `nisarqa -h` and e.g. `nisarqa rslc_qa -h` for usage.

Example commands to process QA:
```
nisarqa rslc_qa <rslc_runconfig.yaml>
nisarqa gslc_qa <gslc_runconfig.yaml>
```

By default, QA outputs the majority of log messages to the log file only.
To additionally stream the log messages to the console, use the verbose flag.
Example:
```
nisarqa gslc_qa <gslc_runconfig.yaml> -v
nisarqa rslc_qa <rslc_runconfig.yaml> --verbose
```


## Runconfig Template w/ default parameters
Because the QA code is uniquely written for each product type, each product
also has a unique runconfig yaml file template and default settings.

The runconfig is where a user specifies the filepath to the input NISAR product,
which workflows to run, what units to use for a given metric, and so forth.

To get a product's example runconfig file that has been populated with
the default parameters, use the `dumpconfig` subcommand.

See `nisarqa dumpconfig -h` for usage.

Example commands:
```
nisarqa dumpconfig rslc
nisarqa dumpconfig rslc --indent 2                # change the indent spacing
nisarqa dumpconfig rslc > my_rslc_runconfig.yaml  # save runconfig to a file
nisarqa dumpconfig gcov
```


## Expected Outputs

For each NISAR product, if all workflows are requested via the `workflows`
parameter in the runconfig, the QA code will generate six output files
and store them in the directory specified by `qa_output_dir` in the runconfig:

1) `BROWSE.png` - RGB browse image for the input NISAR product
2) `BROWSE.kml` - geolocation information for `BROWSE.png`
3) `REPORT.pdf` - graphical summary PDF containing histograms,
                  low-res images of the input datasets, etc.
4) `STATS.h5` - statistical summary HDF5 file containing computed quality
                metrics, the datasets used to generate the plots in 
                `REPORT.pdf`, the QA processing parameters, etc.
5) `SUMMARY.csv` - a high-level PASS/FAIL check summary
6) `LOG.txt` - textual listing of errors encountered during QA processing

These file names are hardcoded; they are not configurable via the
runconfig.


## Copyright
Copyright (c) 2024 California Institute of Technology ("Caltech"). U.S. Government
sponsorship acknowledged.
All rights reserved.