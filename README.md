# QualityAssurance
Quality Assurance software for NISAR
# Minimum PreRequisites:

Python 3.7
numpy 1.17.2
matplotlib 3.1.1
h5py 2.9.0

# Operating Instructions:

python verify_rslc.py –flog <textual log file of all errors encountered>
                       --fhdf <HDF5 file containing a statistical summary>
                       --fpdf <PDF file containing graphical summary> --validate --quality
                       <path to all RSLC files you want to validate, wildcards accepted>

python verify_gcov.py –flog <textual log file of all errors encountered>
                       --fhdf <HDF5 file containing a statistical summary>
                       --fpdf <PDF file containing graphical summary> --validate --quality
                       <path to all GCOV files you want to validate, wildcards accepted>

Specifying the "--validate" flag instructs the code to check for all errors and output the <flog> file.
Specifying the "--quality" flag instructs the code to produce the graphical <fpdf> and statistical <fhdf> files.
One or both flags may be specified.

The code only works on NISAR sample RSLC and GCOV files.  Other file formats are not supported at this time.

# Sample Files

Two sample files are included (one of each type), both in the "samples" subdirectory.
The RSLC file is named winnip_09002_12055_006_120703_L090_CX_129_02.h5 and the GCOV one is called
NISARP_32039_19049_005_190717_L090_CX_129_03_L2GCOV.h5.

# Outputs

This software generates three types of outputs:  

flog = textual listing of errors encountered

fhdf = statistical summary of file in HDF5 format

fpdf = graphical summary of file
  
This delivery includes the expected output files, located in the "expected" subdirectory.  "rslc.h5", "rslc.pdf" and "rslc.log" are the statistical, graphical and error logs for the RSLC file.  Likewise, "gcov.h5", "gcov.pdf" and "gcov.log" are the outputs for the GCOV file. 

To test the RSLC file run the command:

python verify_rslc.py --fhdf rslc.h5 --fpdf rslc.pdf --flog rslc.log --quality --validate winnip_09002_12055_006_120703_L090_CX_129_02.h5

And to test the GCOV file:

python verify_gcov.py --fhdf gcov.h5 --fpdf gcov.pdf --flog gcov.log --quality --validate NISARP_32039_19049_005_190717_L090_CX_129_03_L2GCOV.h5


