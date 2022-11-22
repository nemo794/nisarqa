#!/usr/bin/env python3

# Switch backend to one that doesn't require DISPLAY to be set since we're
# just plotting to file anyway. (Some compute notes do not allow X connections)
# This needs to be set prior to opening any matplotlib objects.
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

import nisarqa

def main(args):
    '''
    Main executable script for QA checks of NISAR RSLC products.

    TODO - Once input arguments are defined, document them here.
    '''
    # Verify inputs
    # TODO - complete the verify_inputs() function. Right now it is 'pass'.
    # Decision - if called as a script, this line is redundant.
    nisarqa.verify_inputs(args)

    # Start logger
    # TODO get logger from Brian's code

    # Open the rslc input file's file handle.
    with nisarqa.open_h5_file(args['input_file'], mode='r') as in_file:

        # TODO Validate file structure
        # (After this, we can assume the file structure for all subsequent accesses to it)
        # NOTE: Refer to the original 'get_bands()' to check that in_file
        # contains metadata, swaths, Identification groups, and that it is SLC/RSLC compliant.
        #   These should trigger a fatal error!
        # NOTE: Refer to the original get_freq_pol() for the verification checks.
        #   This could trigger a fatal error!

        # fin.get_start_time()
        # val_failures = []
        # if args['validate'] and False:
        #     # TODO:
        #     # -> If logic error, fail the entire program.
        #     # -> If validation error, then log those to terminal/log, and continue and do the quality outputs.

        #     # Validate file
        #     xml_file = args['xml_file']
        #     print('Validating file %s with xml spec %s' % (rslc_file, xml_file))
        #     logger.log_message(logging_base.LogFilterInfo, \
        #                     'Validating file %s with xml spec %s' % (rslc_file, xml_file))

        #     fin.find_missing_datasets(xml_file)
        #     fin.check_identification()
        #     for band in fin.bands:
        #         fin.check_frequencies(band, fin.FREQUENCIES[band])
        #     fin.check_time()
        #     fin.check_slant_range()
        #     fin.check_subswaths_bounds()

        # If --quality flag was included, check the images for NaN's, create plots, etc.
        if args['quality']:
            msg = f'Generating Quality reports {args["stats_file"]} and {args["plots_file"]} for file {in_file}'
            print(msg)
            # logger.log_message(logging_base.LogFilterInfo, msg)

            # Get the file's bands, frequencies, and polarizations.
            bands, freqs, pols = nisarqa.rslc.get_bands_freqs_pols(in_file)

            # Open file handles for output stats.h5 and graphs.pdf files
            with nisarqa.open_h5_file(args['stats_file'], mode='w') as stats_file, \
                     PdfPages(args['plots_file']) as plots_file:

                # Store the parameters into a well-defined data structure
                # TODO - Move these hardcoded values into a yaml runconfig
                core_params = nisarqa.rslc.CoreQAParams(
                                plots_pdf=plots_file,
                                stats_h5=stats_file,
                                bands=list(pols),
                                browse_image_dir='.',
                                browse_image_prefix='',
                                tile_shape=(1024,1024))

                pow_img_params = nisarqa.rslc.RSLCPowerImageParams.from_parent(
                                core=core_params,
                                nlooks_freqa=None,
                                nlooks_freqb=None, 
                                linear_units=True,
                                middle_percentile=95.0,
                                num_mpix=4.0,
                                gamma=0.5)

                hist_params = nisarqa.rslc.RSLCHistogramParams.from_parent(
                                core=core_params,
                                decimation_ratio=(8,8),
                                phs_in_radians=True,
                                pow_histogram_start=-80,
                                pow_histogram_endpoint=20)

                # Save parameters to stats.h5 file
                pow_img_params.save_processing_params_to_h5('/QA/processing')
                hist_params.save_processing_params_to_h5('/QA/processing')
                nisarqa.rslc.save_NISAR_identification_group_to_h5(
                                nisar_h5=in_file,
                                stats_h5=stats_file,
                                path_to_group='/identification')
                nisarqa.rslc.save_NISAR_freq_metadata_to_h5(
                                stats_h5=stats_file,
                                path_to_group='/QA/data',
                                pols=pols)

                # Generate the RSLC Power Image
                nisarqa.rslc.process_power_images(
                                pols=pols,
                                params=pow_img_params)

                # Generate the RSLC Power and Phase Histograms
                nisarqa.rslc.process_power_and_phase_histograms(
                                pols=pols,
                                params=hist_params)

                # Process Interferograms

                # Generate Spectra

                # Check for invalid values

                # Compute metrics for stats.h5

        # If --caltools flag was included, check the images for NaN's, create plots, etc.
        if args['caltools']:

            msg = f'Generating Caltools reports for file {in_file}'
            print(msg)
            # logger.log_message(logging_base.LogFilterInfo, msg)

            # Open file handle for output stats.h5 file
            # If file was already been created during --quality step,
            # open in append mode.
            mode = 'a' if args['quality'] else 'w'
            with nisarqa.open_h5_file(args['stats_file'], mode=mode) as stats_file:

                # If QA quality metrics were not generated, then generate the identification group
                if not args['quality']:
                    _, _, pols = nisarqa.rslc.get_bands_freqs_pols(in_file)

                    # Save parameters to stats.h5 file
                    nisarqa.rslc.save_NISAR_identification_group_to_h5(
                                nisar_h5=in_file,
                                stats_h5=stats_file,
                                path_to_group='/identification')

                # Store the parameters into a well-defined data structure
                # TODO - Move these hardcoded values into a yaml runconfig
                core_params = nisarqa.caltools.CoreCalToolsParams(
                                stats_h5=stats_file,
                                bands=list(pols))

                abscal_params = nisarqa.caltools.AbsCalParams.from_parent(
                                core=core_params,
                                corner_reflector_filename='../corner_reflectors.txt',
                                attr1=3.0)                                

                nesz_params = nisarqa.caltools.NESZParams.from_parent(
                                core=core_params,
                                attr1=5.0)                                

                pta_params = nisarqa.caltools.PointTargetAnalyzerParams.from_parent(
                                core=core_params,
                                corner_reflector_filename='../corner_reflectors.txt',
                                attr1=4.5)                                

                # Save processing parameters to stats.h5 file
                abscal_params.save_processing_params_to_h5('/absoluteCalibrationFactor/processing')
                nesz_params.save_processing_params_to_h5('/NESZ/processing')
                pta_params.save_processing_params_to_h5('/pointTargetAnalyzer/processing')

                # Run Absolute Calibration Factor tool
                nisarqa.caltools.run_absolute_cal_factor(
                                params=abscal_params)

                # Run NESZ tool
                nisarqa.caltools.run_nesz(
                                params=nesz_params)

                # Run Point Analyzer tool
                nisarqa.caltools.run_point_target_analyzer(
                                params=pta_params)



    # logger.log_message(logging_base.LogFilterInfo, 'Runtime = %i seconds' % (time.time() - time1))
    # logger.close()

    print('Successful completion. Check log file for validation warnings and errors.')


if __name__ == '__main__':
    args = nisarqa.parse_args('rslc')
    main(args)
