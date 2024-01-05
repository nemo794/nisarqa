from __future__ import annotations

import os
from collections.abc import Mapping

import h5py
from matplotlib.backends.backend_pdf import PdfPages

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def verify_offset(
    user_rncfg: Mapping[str, Mapping], product_type: str, verbose: bool = False
) -> None:
    """
    Verify a ROFF or GOFF product per provided runconfig.

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
        YAML file and which contains the parameters needed to run its QA SAS.
    product_type : str
        One of: 'roff' or 'goff'
    verbose : bool, optional
        True to stream log messages to console (stderr) in addition to the
        log file. False to only stream to the log file. (Initial log messages
        during setup will stream to console regardless.) Defaults to False.
    """
    if product_type not in ("roff", "goff"):
        raise ValueError(f"{product_type=}, must be one of 'roff' or 'goff'.")

    log = nisarqa.get_logger()
    log.info("Begin parsing of runconfig for user-provided QA parameters.")

    # Build the *RootParamGroup parameters per the runconfig
    try:
        root_params = nisarqa.build_root_params(
            product_type=product_type, user_rncfg=user_rncfg
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
        "Parsing of runconfig for QA parameters complete. Complete log"
        " continues in the output log file."
    )
    nisarqa.set_logger_handler(log_file=log_file_txt, verbose=verbose)

    # Log the values of the parameters.
    root_params.log_parameters()

    # For readibility, store output filenames in variables.
    input_file = root_params.input_f.qa_input_file
    out_dir = root_params.get_output_dir()
    browse_file_png = out_dir / root_params.get_browse_png_filename()
    browse_file_kml = out_dir / root_params.get_kml_browse_filename()
    report_file = out_dir / root_params.get_report_pdf_filename()
    stats_file = out_dir / root_params.get_stats_h5_filename()
    summary_file = out_dir / root_params.get_summary_csv_filename()

    msg = f"Starting Quality Assurance for input file: {input_file}"
    log.info(msg)
    if not verbose:
        print(msg)

    if product_type == "roff":
        product = nisarqa.ROFF(input_file)
    else:
        assert product_type == "goff"
        product = nisarqa.GOFF(input_file)

    # Initialize the PASS/FAIL checks summary file
    summary = nisarqa.GetSummary(summary_file)

    # Input product could be opened via the product reader.
    summary.check_can_open_input_file(result="PASS")

    if root_params.workflows.validate:
        msg = f"Beginning validation of input file against XML Product Spec..."
        log.info(msg)

        # Build list of polarizations
        freq_pol: dict[str, list[str]] = {}
        for freq in product.list_of_frequencies:
            freq_pol[freq] = product.get_list_of_polarizations(freq=freq)

        nisarqa.verify_file_against_xml(
            input_file=product.filepath,
            product_type=product.product_type.lower(),
            product_spec_version=product.product_spec_version,
            freq_pols=freq_pol,
            # TODO - create a new function in the product reader
            # called "get_list_of_layers(freq)" to extract the contents of
            # the `listOfLayers` dataset
            layer_numbers=product.available_layer_numbers,
        )

        msg = "Input file validation complete."
        log.info(msg)
        if not verbose:
            print(msg)

    if root_params.workflows.qa_reports:
        log.info("Beginning processing of `qa_reports` items...")

        log.info(f"Beginning processing of browse KML...")
        nisarqa.write_latlonquad_to_kml(
            llq=product.get_browse_latlonquad(),
            output_dir=out_dir,
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
            product.save_qa_metadata_to_h5(stats_h5=stats_h5)

            # Generate along track + slant range browse image, quiver plots,
            # and side-by-side plots for PDF
            nisarqa.process_az_and_slant_rg_offsets_from_offset_product(
                product=product,
                params=root_params.quiver,
                report_pdf=report_pdf,
                stats_h5=stats_h5,
                browse_png=browse_file_png,
            )

            nisarqa.process_az_and_slant_rg_variances_from_offset_product(
                product=product,
                params=root_params.variances,
                report_pdf=report_pdf,
                stats_h5=stats_h5,
            )

            nisarqa.process_cross_variance_and_surface_peak(
                product=product,
                params=root_params.cross_variance,
                report_pdf=report_pdf,
                stats_h5=stats_h5,
            )

        log.info(f"PDF reports saved to {report_file}")
        log.info(f"HDF5 statistics saved to {stats_file}")
        log.info(f"CSV Summary PASS/FAIL checks saved to {summary_file}")
        msg = "`qa_reports` processing complete."
        log.info(msg)
        if not verbose:
            print(msg)

    msg = (
        "QA SAS complete. For details, warnings, and errors see output log"
        " file."
    )
    log.info(msg)
    if not verbose:
        print(msg)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
