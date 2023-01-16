import nisarqa
from ruamel.yaml import YAML

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)

def verify_gslc(runconfig_file):
    '''
    Verify an GSLC product based on the input file, parameters, etc.
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
    # parse runconfig yaml
    parser = YAML(typ='safe')
    with open(runconfig_file, 'r') as f:
        user_runconfig = parser.load(f)

    # get NISAR product input filename
    input_file = user_runconfig['runconfig']['groups']['product_path_group']['qa_input_file']

    # get NISAR product output directory
    output_dir = user_runconfig['runconfig']['groups']['product_path_group']['qa_output_dir']

    # output stub files
    nisarqa.output_stub_files(output_dir, stub_files='all', input_file=input_file)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
