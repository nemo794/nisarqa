import functools
import os
from dataclasses import dataclass, fields

import h5py
import nisarqa
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from PIL import Image

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)

def verify_gslc(user_rncfg):
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
    user_rncfg : dict
        A dictionary whose structure matches an this product's QA runconfig
        yaml file and which contains the parameters needed to run its QA SAS.
    '''

    # Build the RSLCRootParamGroup parameters per the runconfig
    gslc_params = nisarqa.build_root_params(product_type='rslc',
                                            user_rncfg=user_rncfg)
    output_dir = gslc_params.prodpath.qa_output_dir

    print('QA Processing parameters, per runconfig and defaults (runconfig has precedence)')

    gslc_params_names = {
        'input_f': 'Input File Group',
        'prodpath': 'Product Path Group',
        'workflows': 'Workflows',
        'power_img': 'Power Image'
        }

    for params_obj in fields(gslc_params):
        grp_name = gslc_params_names[params_obj.name]
        print(f'  {grp_name} Input Parameters:')

        po = getattr(gslc_params, params_obj.name)
        if po is not None:
            for param in fields(po):
                po2 = getattr(po, param.name)
                if isinstance(po2, bool):
                    print(f'    {param.name}: {po2}')
                else:
                    print(f'    {param.name}: {po2}')

    # Start logger
    # TODO get logger from Brian's code and implement here
    # For now, output the stub log file.
    nisarqa.output_stub_files(output_dir=output_dir, stub_files='log_txt')

    # Create file paths for output files ()
    input_file = gslc_params.input_f.qa_input_file
    msg = f'Starting Quality Assurance for input file: {input_file}' \
            f'\nOutputs to be generated:'

    if gslc_params.workflows.validate or gslc_params.workflows.qa_reports:
        summary_file = os.path.join(output_dir, 'SUMMARY.csv')
        msg += f'\n\tSummary file: {summary_file}'

    if gslc_params.workflows.qa_reports:
        report_file = os.path.join(output_dir, 'REPORT.pdf')
        browse_image = os.path.join(output_dir, 'BROWSE.png')
        browse_kml = os.path.join(output_dir, 'BROWSE.kml')
        stats_file = os.path.join(output_dir, 'STATS.h5')

        msg += f'\n\tReport file: {report_file}' \
               f'\n\tMetrics file: {stats_file}' \
               f'\n\tBrowse Image: {browse_image}' \
               f'\n\tBrowse Image Geolocation file: {browse_kml}'

    print(msg)

    # Parse the file's bands, frequencies, and polarizations.
    # Save data to STATS.h5
    with nisarqa.open_h5_file(input_file, mode='r') as in_file:

        # Note: `pols` contains references to datasets in the open input file.
        # All processing with `pols` must be done within this context manager,
        # or the references will be closed and inaccessible.
        bands, freqs, pols = nisarqa.rslc.get_bands_freqs_pols(in_file)

        # If running these workflows, save the processing parameters and
        # identification group to STATS.h5
        if gslc_params.workflows.qa_reports:

            # This is the first time opening the STATS.h5 file for RSLC
            # workflow, so open in 'w' mode.
            # After this, always open STATS.h5 in 'r+' mode.
            with nisarqa.open_h5_file(stats_file, mode='w') as stats_h5:

                # Save the processing parameters to the stats.h5 file
                # Note: If only the validate workflow is requested,
                # this will do nothing.
                gslc_params.save_params_to_stats_file(h5_file=stats_h5,
                                                      bands=bands)

                # Copy the Product identification group to STATS.h5
                nisarqa.rslc.save_NISAR_identification_group_to_h5(
                        nisar_h5=in_file,
                        stats_h5=stats_h5)

        # Run the requested workflows
        if gslc_params.workflows.validate:
            # TODO Validate file structure
            # (After this, we can assume the file structure for all 
            # subsequent accesses to it)
            # NOTE: Refer to the original get_freq_pol() for the verification 
            # checks. This could trigger a fatal error!

            # These reports will be saved to the SUMMARY.csv file.
            # For now, output the stub file
            nisarqa.output_stub_files(output_dir=output_dir,
                                    stub_files='summary_csv')

        if gslc_params.workflows.qa_reports:

            # TODO qa_reports will add to the SUMMARY.csv file.
            # For now, make sure that the stub file is output
            if not os.path.isfile(summary_file):
                nisarqa.output_stub_files(output_dir=output_dir,
                                        stub_files='summary_csv')

            # TODO qa_reports will create the BROWSE.kml file.
            # For now, make sure that the stub file is output
            nisarqa.output_stub_files(output_dir=output_dir,
                                    stub_files='browse_kml')

            with nisarqa.open_h5_file(stats_file, mode='r+') as stats_h5, \
                PdfPages(report_file) as report_pdf:

                # Save frequency/polarization info to stats file
                nisarqa.rslc.save_nisar_freq_metadata_to_h5(
                                    stats_h5=stats_h5, pols=pols)

                # Generate the GSLC Power Image and Browse Image
                # Note: the `nlooks*` parameters might be updated. TODO comment better.
                nisarqa.rslc.process_power_images(
                                    pols=pols,
                                    params=gslc_params.power_img,
                                    stats_h5=stats_h5,
                                    report_pdf=report_pdf,
                                    browse_filename=browse_image)

                # Generate the GSLC Power and Phase Histograms

                # Process Interferograms

                # Check for invalid values

                # Compute metrics for stats.h5

    print('Successful completion. Check log file for validation warnings and errors.')


__all__ = nisarqa.get_all(__name__, objects_to_skip)
