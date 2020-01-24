# QualityAssurance
Quality Assurance software for NISAR
# Minimum PreRequisites:

Python 3.7
numpy 1.17.2
matplotlib 3.1.1
h5py 2.9.0

# Operating Instructions:

python verify_slc.py –flog <textual log file of all errors encountered>
                       --fhdf <HDF5 file containing a statistical summary>
                       --fpdf <PDF file containing graphical summary> --validate --quality
                       <path to all files you want to validate, wildcards accepted>

Specifying the "--validate" flag instructs the code to check for all errors and output the <flog> file.
Specifying the "--quality" flag instructs the code to produce the graphical <fpdf> and statistical <fhdf> files.
One or both flags may be specified.

The code only works on NISAR sample SLC files.  Other file formats are not supported at this time.

# Sample Files

Two sample files are being included with this delivery ("good" and "bad").  The "good" file (starts with "SanAnd") passes all my tests, whereas the "bad" one (starts with "winnip") will raise an error about non-uniform time spacing.  Note that both the "good" and "bad" files will raise errors about an invalid track number.

# Outputs

This software generates three types of outputs:  

<flog> = textual listing of errors encountered
<fhdf> = statistical summary of file in HDF5 format
<fpdf> = graphical summary of file
  
This delivery includes the output files: "good.pdf", "good.h5", "good.log" as well as their "bad" counterparts.  

To generate the "good" files run the command:

python verify_slc.py --flog good.log --fhdf good.h5 --fpdf good.pdf --quality --validate SanAnd_05024_18038_006_180730_L090_CX_129_03.h5

And to generate the "bad" files:

python verify_slc.py --flog bad.log --fhdf bad.h5 --fpdf bad.pdf --quality --validate winnip_09002_12055_006_120703_L090_CX_129_02.h5


