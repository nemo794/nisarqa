import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)

def verify_runw(runconfig_file):
    '''
    Verify an RUNW product based on the input file, parameters, etc.
    specified in the input runconfig file.

    This is the main function for running the entire QA workflow for this
    product. It will run based on the options supplied in the 
    input runconfig file.
    The input runconfig file must follow the standard QA runconfig format
    for this product. Run the command line command:
            nisar_qa dumpconfig <product name>
    to generate an example template with default parameters for this product.

    Parameters
    ----------
    runconfig_file : str
        Full filename for an existing QA runconfig file for this NISAR product
    '''

    nisarqa.verify_insar(runconfig_file,'runw')


__all__ = nisarqa.get_all(__name__, objects_to_skip)
