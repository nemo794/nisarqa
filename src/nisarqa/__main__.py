#!/usr/bin/env python3
# Switch backend to one that doesn't require DISPLAY to be set since we're
# just plotting to file anyway. (Some compute notes do not allow X connections)
# This needs to be set prior to opening any matplotlib objects.
import matplotlib
import pkg_resources

matplotlib.use('Agg')

import nisarqa


def dumpconfig(product_type):
    if product_type == 'rslc':
        nisarqa.RSLCRootParams.dump_runconfig_template()
    else:
        raise Exception(f'{product_type} dumpconfig code not implemented yet.')

    return True

def main():
    # parse the args
    args = nisarqa.parse_cli_args()

    if args.version:
        print(f'nisarqa v{pkg_resources.require("nisarqa")[0].version}')
        return

    subcommand = args.command

    if subcommand == 'dumpconfig':
        dumpconfig(args.product_type)
    elif subcommand == 'rslc_qa':
        nisarqa.rslc.verify_rslc(runconfig_file=args.runconfig_yaml)
    elif subcommand == 'gslc_qa':
        nisarqa.gslc.verify_gslc(runconfig_file=args.runconfig_yaml)
    else:
        raise Exception('Not implemented yet!)')


if __name__ == '__main__':
    main()
