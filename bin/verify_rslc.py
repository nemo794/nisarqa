#!/usr/bin/env python3

# Switch backend to one that doesn't require DISPLAY to be set since we're
# just plotting to file anyway. (Some compute notes do not allow X connections)
# This needs to be set prior to opening any matplotlib objects.
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

import nisarqa

def main(args):
    """
    Main executable script for QA checks of NISAR RSLC products.

    TODO - Once input arguments are defined, document them here.
    """
    # Verify inputs
    # TODO - complete the verify_inputs() function. Right now it is "pass".
    # Decision - if called as a script, this line is redundant.
    nisarqa.verify_inputs(args)

    # Start logger
    # TODO get logger from Brian's code

    # Open the rslc input file's file handle.
    with nisarqa.open_h5_file(args['input_file'], mode='r') as in_file:

        # TODO Validate file structure
        # (After this, we can assume the file structure for all subsequent accesses to it)
        # NOTE: Refer to the original "get_bands()" to check that in_file
        # contains metadata, swaths, Identification groups, and that it is SLC/RSLC compliant.
        #   These should trigger a fatal error!
        # NOTE: Refer to the original get_freq_pol() for the verification checks.
        #   This could trigger a fatal error!

        # fin.get_start_time()
        # val_failures = []
        # if args["validate"] and False:
        #     # TODO:
        #     # -> If logic error, fail the entire program.
        #     # -> If validation error, then log those to terminal/log, and continue and do the quality outputs.

        #     # Validate file
        #     xml_file = args['xml_file']
        #     print("Validating file %s with xml spec %s" % (rslc_file, xml_file))
        #     logger.log_message(logging_base.LogFilterInfo, \
        #                     "Validating file %s with xml spec %s" % (rslc_file, xml_file))

        #     fin.find_missing_datasets(xml_file)
        #     fin.check_identification()
        #     for band in fin.bands:
        #         fin.check_frequencies(band, fin.FREQUENCIES[band])
        #     fin.check_time()
        #     fin.check_slant_range()
        #     fin.check_subswaths_bounds()

        # If --quality flag was included, check the images for NaN's, create plots, etc.
        if args["quality"]:
            msg = f"Generating Quality reports {args['stats_file']} and {args['plots_file']} for file {in_file}"
            print(msg)
            # logger.log_message(logging_base.LogFilterInfo, msg)

            # Open file handles for output stats.h5 and graphs.pdf files
            with nisarqa.open_h5_file(args['stats_file'], mode='w') as stats_file, \
                     PdfPages(args["plots_file"]) as plots_file:

                # Get the file's bands, frequencies, and polarizations.
                # This will also create the mask_ok and log invalid pixels.
                bands, freqs, pols = nisarqa.rslc.get_bands_freq_pols(in_file)

                # Generate the RSLC Power Image
                nisarqa.rslc.process_power_image(pols=pols,
                                         plots_pdf=plots_file,
                                         nlooks_freqA=None,
                                         nlooks_freqB=None, 
                                         linear_units=True,
                                         middle_percentile=95.0,
                                         num_MPix=4.0,
                                         highlight_inf_pixels=True,
                                         browse_image_dir=".",
                                         browse_image_prefix=None,
                                         tile_shape=(16384,16384))

                # # Create output stats.h5 and graphs.pdf files
                # fin.create_images(time_step=args["time_step"], range_step=args["range_step"])
                # fin.check_for_invalid_values()
                # fin.check_images(fpdf_out, fhdf_out)


    # logger.log_message(logging_base.LogFilterInfo, "Runtime = %i seconds" % (time.time() - time1))
    # logger.close()

    print("Successful completion. Check log file for validation warnings and errors.")


if __name__ == "__main__":
    args = nisarqa.parse_args('rslc')
    main(args)
