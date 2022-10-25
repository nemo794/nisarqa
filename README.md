# QualityAssurance
Quality Assurance software for NISAR
# Minimum PreRequisites:

Python 3.7
TODO

# Operating Instructions:

## Installation

Clone the repo and `cd` into the top level directory.

For standard installation, run:
```
python setup.py install
python setup.py clean
```

For develop mode, run:
```
python setup.py develop
```

##
verify_rslc.py --quality
               --input <path to RSLC files you want to validate>

Specifying the '--quality' flag instructs the code to produce the graphical <fpdf> and statistical <fhdf> output files.

The code only works on NISAR sample RSLC files.  Other file formats are not supported at this time.

## Test RSLC Data

TODO: Update this section (see original QA Code README), and remove the hardlinks.

Multi-Freq, Multi-Pol.
Data Size: Medium-Large
`verify_rslc.py --quality --input /home/niemoell/dat/fromJoanne_05022022/rslc_REE_testarea134/output_rslc/rslc.h5`

Rosemond - Multi-Freq, Multi-Pol. Uses original 'SLC' convention (not 'RSLC')
Data Size: Small
`verify_rslc.py --quality --input /scratch/gunter/data/NISAR/QualityAssurance/Rosamd_35012_20001_001_200129_L090_CX_03/Rosamd_35012_20001_001_200129_L090_CX_129_03.h5`

Real Data that has been manipulated to be like NISAR data
Data Size: Small
`verify_rslc.py --quality --input /home/niemoell/dat/qa_test_data_04182022/rslc_ALPSRP037370690/output_rslc/rslc.h5`

Real Data that has been manipulated to be like NISAR data
Data Size: Large, but could complete in original QA Code
`verify_rslc.py --quality --input /home/niemoell/dat/qa_test_data_04182022/rslc_ALPSRP271200680/output_rslc/rslc.h5`

DIST1 - Simulated data to look like an actual image
Data Size: Small
`verify_rslc.py --quality --input /home/niemoell/dat/qa_test_data_04182022/rslc_DIST1/output_rslc/rslc.h5`

DIST2 - Simulated data to look like an actual image
Data Size: Medium
`verify_rslc.py --quality --input /home/niemoell/dat/qa_test_data_04182022/rslc_DIST2/output_rslc/rslc.h5`

REE1 - Simulated data to look like an actual image
Data Size: Small
`verify_rslc.py --quality --input /home/niemoell/dat/qa_test_data_04182022/rslc_REE1/output_rslc/rslc.h5`

REE2 - Simulated data to look like an actual image
Data Size: Large, could not complete in original QA Code
`verify_rslc.py --quality --input /home/niemoell/dat/qa_test_data_04182022/rslc_REE2/output_rslc/rslc.h5`


# Outputs

This software generates three types of outputs:  

qa.log = textual listing of errors encountered

qa.h5 = statistical summary of file in HDF5 format

qa.pdf = graphical summary of file
  
For example, 'rslc.h5', 'rslc.pdf' and 'rslc.log' are the statistical, graphical and error logs for the RSLC file.  Likewise, 'gcov.h5', 'gcov.pdf' and 'gcov.log' are the outputs for the GCOV file and the hdf, pdf and log files for the GSLC are called 'gslc.h5', 'gslc.pdf' and 'gslc.log' respectively. 
