# QualityAssurance
Quality Assurance software for NISAR

For the upcoming [NISAR mission](https://nisar.jpl.nasa.gov/),
eight types of L1/L2 data products will be generated.
This Quality Assurance (QA) software is designed to look at the data products
produced one at a time. For each product, the QA code can:
- Verify the metadata matches the product spec
- Generate metrics, a PDF report, and a summary describing the quality of the product
- Run CalTools processes on RSLC products
- Produce a browse image png for that product

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
via the traditional `python setup.py install` or without the `--no-deps` flag
method fails due to the `isce3` dependency in `setup.py`.
Otherwise, the `isce3` dependency would need to be removed from `setup.py`,
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

- NOTE: `dumpconfig` has only been implemented for RSLC and GSLC at this time.


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


## Test RSLC Data
Here are paths on nisar-adt-dev-3 to various test data sets.
These path will need to be copied into the `qa_input_file` parameter in
the runconfig.

TODO: Update this section (see original QA Code README), and remove the hardlinks.

Multi-Freq, Multi-Pol.
Data Size: Medium-Large
`/home/niemoell/dat/fromJoanne_05022022/rslc_REE_testarea134/output_rslc/rslc.h5`

Real Data that has been manipulated to be like NISAR data
Data Size: Small
`/home/niemoell/dat/qa_test_data_04182022/rslc_ALPSRP037370690/output_rslc/rslc.h5`

Real Data that has been manipulated to be like NISAR data
Data Size: Large, but could complete in original QA Code
`/home/niemoell/dat/qa_test_data_04182022/rslc_ALPSRP271200680/output_rslc/rslc.h5`

DIST1 - Simulated data to look like an actual image
Data Size: Small
`/home/niemoell/dat/qa_test_data_04182022/rslc_DIST1/output_rslc/rslc.h5`

DIST2 - Simulated data to look like an actual image
Data Size: Medium
`/home/niemoell/dat/qa_test_data_04182022/rslc_DIST2/output_rslc/rslc.h5`

REE1 - Simulated data to look like an actual image
Data Size: Small
`/home/niemoell/dat/qa_test_data_04182022/rslc_REE1/output_rslc/rslc.h5`

REE2 - Simulated data to look like an actual image
Data Size: Large, could not complete in original QA Code
`/home/niemoell/dat/qa_test_data_04182022/rslc_REE2/output_rslc/rslc.h5`

Rosamond - Multi-Freq, Multi-Pol. Modified UAVSAR data.
Data Size: Small
`/scratch/gunter/data/NISAR/QualityAssurance/Rosamd_35012_20001_001_200129_L090_CX_03/Rosamd_35012_20001_001_200129_L090_CX_129_03.h5`

Hawaii (Big Island)
`/home/niemoell/dat/UAVSAR_RSLC_testdata_09222022/BigIsl_32905_10003_012_100106_L090_CX_143_02.h5`

## Test GCOV Data
Here are paths on nisar-adt-dev-3 to various test data sets.
These path will need to be copied into the `qa_input_file` parameter in
the runconfig.

Los Angeles
`/home/niemoell/dat/gcov_test_data/may2023/L2_GCOV_LA.h5`

Peru - Only On-Diagonal Terms
`/home/niemoell/dat/gcov_test_data/may2023/L2_GCOV_s1_peru.h5`

Full-Covariance Datasets (includes off-diagonal terms)
The PDF output should include phase histograms for the off-diag terms only
`/home/niemoell/dat/gcov_test_data/may2023/L2_GCOV_FULL_COV_s1_peru.h5`

