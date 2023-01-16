import argparse

import nisarqa

objects_to_skip = nisarqa.get_all(__name__)

def parse_cli_args():
    '''
    Parse the command line arguments

    Possible command line arguments:
        nisar_qa --version
        nisar_qa dumpconfig <product type>
        nisar_qa <product type>_qa <runconfig yaml file>

    Examples command line calls for RSLC product:
        nisar_qa --version
        nisar_qa dumpconfig rslc
        nisar_qa rslc_qa runconfig.yaml
    
    Returns
    -------
    args : argparse.Namespace
        The parsed command line arguments
    '''

    list_of_products = ['rslc', 'gslc', 'gcov', 'rifg',
                        'runw', 'gunw', 'roff', 'goff']

    # create the top-level parser
    msg = 'Quality Assurance processing to verify NISAR ' \
          'product files generated by ISCE3'
    parser = argparse.ArgumentParser(
                    description=msg,
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter
                    )

    # --version
    parser.add_argument(
            '-v',
            '--version',
            dest='version',
            action='store_true',
            help='Display nisarqa software version')

    # create sub-parser
    sub_parsers = parser.add_subparsers(help='sub-command help',
                                        dest='command')

    # create the parser for the `dumpconfig` sub-command
    msg = 'Output NISAR QA runconfig template ' \
          'with default values. ' \
          'For usage, see: `nisarqa dumpconfig -h`'
    parser_dumpconfig = sub_parsers.add_parser('dumpconfig', help=msg)

    # Add the required positional argument for the dumpconfig
    parser_dumpconfig.add_argument(
            'product_type',  # positional argument
            choices=list_of_products,
            help='Product type of the default runconfig template')
    
    # create a parser for each `*_qa` subcommands
    msg = 'Run QA for a NISAR %s with runconfig yaml. Usage: `nisarqa %s_qa <runconfig.yaml>`'
    for prod in list_of_products:
        parser_qa = sub_parsers.add_parser(f'{prod}_qa', 
                help=msg % (prod.upper(), prod.lower()))
        parser_qa.add_argument(
                f'runconfig_yaml',
                help=f'NISAR {prod.upper()} product runconfig yaml file')

    # parse args
    args = parser.parse_args()

    return args


__all__ = nisarqa.get_all(__name__, objects_to_skip)
