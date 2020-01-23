# QualityAssurance
Quality Assurance software for NISAR
# Minimum PreRequisites:

Python 3.7
numpy 1.17.2
matplotlib 3.1.1
h5py 2.9.0

# Operating Instructions:

python ./verify_slc.py –flog <textual log file of all errors encountered>
                       --fhdf <HDF5 file containing a statistical summary>
                       --fpdf <PDF file containing graphical summary> --validate --quality
                       <path to all files you want to validate, wildcards accepted>

Specifying the "--validate" flag instructs the code to check for all errors and output the <flog> file.
Specifying the "--quality" flag instructs the code to produce the graphical <fpdf> and statistical <fhdf> files.
One or both flags may be specified.

The code only works on NISAR sample SLC files.  Other file formats are not supported at this time.


