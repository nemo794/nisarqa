import os

import h5py
from matplotlib.backends.backend_pdf import PdfPages

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def verify_gslc(
    user_rncfg: dict[str, dict], console_verbosity: str = "quiet"
) -> None:
    """
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
    user_rncfg : nested dict
        A dictionary whose structure matches this product's QA runconfig
        YAML file and which contains the parameters needed to run its QA SAS.]
    console_verbosity : int, optional
        Minimum level of log messages to stream to console (stderr). Options:
            "quiet"    : (default) Almost none. (log messages prior to log
                         file setup, etc.)
            "critical" : A serious error, indicating that the program itself
                         may be unable to continue running.
            "error"    : Due to a more serious problem, the software has not
                         been able to perform some function.
            "warning"  : An indication that something unexpected happened, or
                         that a problem might occur in the near future
                         (e.g. ‘disk space low’). The software is still working
                         as expected.
            "info"     : Confirmation that things are working as expected.
            "debug"    : Detailed information, typically only of interest to a
                         developer trying to diagnose a problem.
        Note: after the log file is setup, all levels of log messages will
        always be output to the log file. `verbosity` is a mechanism for users
        to additionally see log messages stream to console in real time.
    """
    log = nisarqa.get_logger()
    log.info("Begin parsing of runconfig for user-provided QA parameters.")

    # Build the GSLCRootParamGroup parameters per the runconfig
    try:
        root_params = nisarqa.build_root_params(
            product_type="gslc", user_rncfg=user_rncfg
        )
    except nisarqa.ExitEarly:
        # No workflows were requested. Exit early.
        log.info(
            "All `workflows` set to `False` in the runconfig, "
            "so no QA outputs will be generated. This is not an error."
        )
        return

    # Start logging in the log file
    out_dir = root_params.get_output_dir()
    log_file_txt = out_dir / root_params.get_log_filename()
    log.info(
        f"Parsing of runconfig for QA parameters complete. Complete log"
        f" continues in the output log file."
    )
    nisarqa.set_logger_handler(
        log_file=log_file_txt, console_verbosity=console_verbosity
    )

    # Log the values of the parameters.
    # Currently, this prints to stdout. Once the logger is implemented,
    # it should log the values directly to the log file.
    root_params.log_parameters()

    # For readibility, store output filenames in variables.
    # Depending on which workflows are set to True, not all filename
    # variables will be used.
    input_file = root_params.input_f.qa_input_file
    out_dir = root_params.get_output_dir()
    browse_file_png = out_dir / root_params.get_browse_png_filename()
    browse_file_kml = out_dir / root_params.get_kml_browse_filename()
    report_file = out_dir / root_params.get_report_pdf_filename()
    stats_file = out_dir / root_params.get_stats_h5_filename()
    summary_file = out_dir / root_params.get_summary_csv_filename()

    msg = f"Starting Quality Assurance for input file: {input_file}"
    print(msg)
    log.info(msg)

    if root_params.workflows.validate:
        msg = f"Beginning validation of input file against XML Product Spec..."
        log.info(msg)

        # TODO Validate file structure
        # (After this, we can assume the file structure for all
        # subsequent accesses to it)
        # NOTE: Refer to the original get_freq_pol() for the verification
        # checks. This could trigger a fatal error!

        # These reports will be saved to the SUMMARY.csv file.
        # For now, output the stub file
        nisarqa.output_stub_files(output_dir=out_dir, stub_files="summary_csv")
        msg = f"Input file validation PASS/FAIL checks saved: {summary_file}"
        log.info(msg)
        msg = "Input file validation complete."
        print(msg)
        log.info(msg)

    if root_params.workflows.qa_reports:
        log.info(f"Beginning `qa_reports` processing...")

        product = nisarqa.GSLC(input_file)

        # TODO qa_reports will add to the SUMMARY.csv file.
        # For now, make sure that the stub file is output
        if not os.path.isfile(summary_file):
            nisarqa.output_stub_files(
                output_dir=root_params.get_output_dir(),
                stub_files="summary_csv",
            )
            log.info(f"PASS/FAIL checks saved to {summary_file}")
            msg = "PASS/FAIL checks complete."
            print(msg)
            log.info(msg)

        log.info(f"Beginning processing of browse KML...")
        nisarqa.write_latlonquad_to_kml(
            llq=product.get_browse_latlonquad(),
            output_dir=root_params.get_output_dir(),
            kml_filename=root_params.get_kml_browse_filename(),
            png_filename=root_params.get_browse_png_filename(),
        )
        log.info(f"Browse image kml file saved to {browse_file_kml}")

        with h5py.File(stats_file, mode="w") as stats_h5, PdfPages(
            report_file
        ) as report_pdf:
            # Save the processing parameters to the stats.h5 file
            root_params.save_processing_params_to_stats_h5(
                h5_file=stats_h5, band=product.band
            )
            log.info(f"QA Processing Parameters saved to {stats_file}")

            nisarqa.rslc.copy_identification_group_to_stats_h5(
                product=product, stats_h5=stats_h5
            )
            log.info(f"Input file Identification group copied to {stats_file}")

            # Save frequency/polarization info to stats file
            nisarqa.rslc.save_nisar_freq_metadata_to_h5(
                stats_h5=stats_h5, product=product
            )

            input_raster_represents_power = False
            name_of_backscatter_content = (
                r"GSLC Backscatter Coefficient ($\beta^0$)"
            )

            # Generate the GSLC Power Image and Browse Image
            nisarqa.rslc.process_backscatter_imgs_and_browse(
                product=product,
                params=root_params.backscatter_img,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
                browse_filename=browse_file_png,
            )
            log.info("Processing of Backscatter images complete.")
            log.info(f"Browse image PNG file saved to {browse_file_png}")

            # Generate the GSLC Power and Phase Histograms
            nisarqa.rslc.process_backscatter_and_phase_histograms(
                product=product,
                params=root_params.histogram,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
            )
            log.info("Processing of backscatter and phase histograms complete.")

            # Process Interferograms

            # Check for invalid values

            # Compute metrics for stats.h5

            log.info(f"PDF reports saved to {report_file}")
            log.info(f"HDF5 statistics saved to {stats_file}")
            log.info(f"CSV Summary PASS/FAIL checks saved to {summary_file}")
            msg = "`qa_reports` processing complete."
            print(msg)
            log.info(msg)

    msg = (
        "QA SAS complete. For details, warnings, and errors see output log"
        " file."
    )
    print(msg)
    log.info(msg)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
