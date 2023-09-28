from __future__ import annotations

import os
import warnings

from matplotlib.backends.backend_pdf import PdfPages

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def verify_rifg(user_rncfg):
    """
    Verify an RIFG product per provided runconfig.

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
        A dictionary whose structure matches this product's QA runconfig
        YAML file and which contains the parameters needed to run its QA SAS.
    """
    # Build the RIFGRootParamGroup parameters per the runconfig
    try:
        root_params = nisarqa.build_root_params(
            product_type="rifg", user_rncfg=user_rncfg
        )
    except nisarqa.ExitEarly:
        # No workflows were requested. Exit early.
        print(
            "All `workflows` set to `False` in the runconfig, "
            "so no QA outputs will be generated. This is not an error."
        )
        return

    # Start logger
    # TODO get logger from Brian's code and implement here
    # For now, output the stub log file.
    nisarqa.output_stub_files(
        output_dir=root_params.get_output_dir(), stub_files="log_txt"
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

    print(f"Starting Quality Assurance for input file: {input_file}")

    if root_params.workflows.validate:
        print("Beginning input file validation...")

        # TODO Validate file structure
        # (After this, we can assume the file structure for all
        # subsequent accesses to it)
        # NOTE: Refer to the original get_freq_pol() for the verification
        # checks. This could trigger a fatal error!

        # These reports will be saved to the SUMMARY.csv file.
        # For now, output the stub file
        nisarqa.output_stub_files(output_dir=out_dir, stub_files="summary_csv")
        print(f"Input file validation PASS/FAIL checks saved: {summary_file}")
        print("Input file validation complete.")

    if root_params.workflows.qa_reports:
        print("Beginning processing of `qa_reports` items...")

        product = nisarqa.RIFG(input_file)

        # TODO qa_reports will add to the SUMMARY.csv file.
        # For now, make sure that the stub file is output
        if not os.path.isfile(summary_file):
            nisarqa.output_stub_files(
                output_dir=root_params.get_output_dir(),
                stub_files="summary_csv",
            )
            print(f"PASS/FAIL checks saved to: {summary_file}")

        try:
            nisarqa.write_latlonquad_to_kml(
                llq=nisarqa.get_latlonquad(product=product),
                output_dir=root_params.get_output_dir(),
                kml_filename=root_params.get_kml_browse_filename(),
                png_filename=root_params.get_browse_png_filename(),
            )
        except ValueError:
            warnings.warn("Could not create KML file. Update product reader.")
            nisarqa.output_stub_files(
                output_dir=out_dir, stub_files="browse_kml"
            )

        print("Processing of browse image kml complete.")
        print(f"Browse image kml file saved to {browse_file_kml}")

        with nisarqa.open_h5_file(stats_file, mode="w") as stats_h5, PdfPages(
            report_file
        ) as report_pdf:
            # Save the processing parameters to the stats.h5 file
            root_params.save_processing_params_to_stats_h5(
                h5_file=stats_h5, band=product.band
            )
            print(f"QA Processing Parameters saved to {stats_file}")

            nisarqa.rslc.copy_identification_group_to_stats_h5(
                product=product, stats_h5=stats_h5
            )
            print(f"Input file Identification group copied to {stats_file}")

            # Save frequency/polarization info to stats file
            product.save_qa_metadata_to_h5(stats_h5=stats_h5)

            # Save Browse Image
            nisarqa.make_hsi_browse_wrapped(
                product=product,
                params=root_params.hsi,
                browse_png=browse_file_png,
            )

            # Save HSI images to PDF
            nisarqa.hsi_images_to_pdf_wrapped(
                product=product, report_pdf=report_pdf
            )

    print(
        "Successful completion of QA SAS. Check log file for validation"
        " warnings and errors."
    )


def verify_runw(user_rncfg):
    """
    Verify an RUNW product per provided runconfig.

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
        A dictionary whose structure matches this product's QA runconfig
        YAML file and which contains the parameters needed to run its QA SAS.
    """
    # Build the RUNWRootParamGroup parameters per the runconfig
    try:
        root_params = nisarqa.build_root_params(
            product_type="runw", user_rncfg=user_rncfg
        )
    except nisarqa.ExitEarly:
        # No workflows were requested. Exit early.
        print(
            "All `workflows` set to `False` in the runconfig, "
            "so no QA outputs will be generated. This is not an error."
        )
        return

    # Start logger
    # TODO get logger from Brian's code and implement here
    # For now, output the stub log file.
    nisarqa.output_stub_files(
        output_dir=root_params.get_output_dir(), stub_files="log_txt"
    )

    # Log the values of the parameters.
    # Currently, this prints to stdout. Once the logger is implemented,
    # it should log the values directly to the log file.
    root_params.log_parameters()

    # For readibility, store output filenames in variables.
    input_file = root_params.input_f.qa_input_file
    out_dir = root_params.get_output_dir()
    browse_file_png = out_dir / root_params.get_browse_png_filename()
    browse_file_kml = out_dir / root_params.get_kml_browse_filename()
    report_file = out_dir / root_params.get_report_pdf_filename()
    stats_file = out_dir / root_params.get_stats_h5_filename()
    summary_file = out_dir / root_params.get_summary_csv_filename()

    print(f"Starting Quality Assurance for input file: {input_file}")

    if root_params.workflows.validate:
        print("Beginning input file validation...")

        # TODO Validate file structure
        # (After this, we can assume the file structure for all
        # subsequent accesses to it)
        # NOTE: Refer to the original get_freq_pol() for the verification
        # checks. This could trigger a fatal error!

        # These reports will be saved to the SUMMARY.csv file.
        # For now, output the stub file
        nisarqa.output_stub_files(output_dir=out_dir, stub_files="summary_csv")
        print(f"Input file validation PASS/FAIL checks saved: {summary_file}")
        print("Input file validation complete.")

    if root_params.workflows.qa_reports:
        print("Beginning processing of `qa_reports` items...")

        product = nisarqa.RUNW(input_file)

        # TODO qa_reports will add to the SUMMARY.csv file.
        # For now, make sure that the stub file is output
        if not os.path.isfile(summary_file):
            nisarqa.output_stub_files(
                output_dir=root_params.get_output_dir(),
                stub_files="summary_csv",
            )
            print(f"PASS/FAIL checks saved to: {summary_file}")

        try:
            nisarqa.write_latlonquad_to_kml(
                llq=nisarqa.get_latlonquad(product=product),
                output_dir=root_params.get_output_dir(),
                kml_filename=root_params.get_kml_browse_filename(),
                png_filename=root_params.get_browse_png_filename(),
            )
        except ValueError:
            warnings.warn("Could not create KML file. Update product reader.")
            nisarqa.output_stub_files(
                output_dir=out_dir, stub_files="browse_kml"
            )

        print("Processing of browse image kml complete.")
        print(f"Browse image kml file saved to {browse_file_kml}")

        with nisarqa.open_h5_file(stats_file, mode="w") as stats_h5, PdfPages(
            report_file
        ) as report_pdf:
            # Save the processing parameters to the stats.h5 file
            root_params.save_processing_params_to_stats_h5(
                h5_file=stats_h5, band=product.band
            )
            print(f"QA Processing Parameters saved to {stats_file}")

            nisarqa.rslc.copy_identification_group_to_stats_h5(
                product=product, stats_h5=stats_h5
            )
            print(f"Input file Identification group copied to {stats_file}")

            # Save frequency/polarization info to stats file
            product.save_qa_metadata_to_h5(stats_h5=stats_h5)

            # Save Browse Image
            nisarqa.make_hsi_browse_unwrapped(
                product=product,
                params=root_params.hsi,
                browse_png=browse_file_png,
            )

            # Save HSI images to PDF
            nisarqa.hsi_images_to_pdf_unwrapped(
                product=product,
                report_pdf=report_pdf,
                rewrap=root_params.hsi.rewrap,
            )

    print(
        "Successful completion of QA SAS. Check log file for validation"
        " warnings and errors."
    )


def verify_gunw(user_rncfg):
    """
    Verify a GUNW product per provided runconfig.

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
        A dictionary whose structure matches this product's QA runconfig
        YAML file and which contains the parameters needed to run its QA SAS.
    """
    # Build the GUNWRootParamGroup parameters per the runconfig
    try:
        root_params = nisarqa.build_root_params(
            product_type="gunw", user_rncfg=user_rncfg
        )
    except nisarqa.ExitEarly:
        # No workflows were requested. Exit early.
        print(
            "All `workflows` set to `False` in the runconfig, "
            "so no QA outputs will be generated. This is not an error."
        )
        return

    # Start logger
    # TODO get logger from Brian's code and implement here
    # For now, output the stub log file.
    nisarqa.output_stub_files(
        output_dir=root_params.get_output_dir(), stub_files="log_txt"
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

    print(f"Starting Quality Assurance for input file: {input_file}")

    if root_params.workflows.validate:
        print("Beginning input file validation...")

        # TODO Validate file structure
        # (After this, we can assume the file structure for all
        # subsequent accesses to it)
        # NOTE: Refer to the original get_freq_pol() for the verification
        # checks. This could trigger a fatal error!

        # These reports will be saved to the SUMMARY.csv file.
        # For now, output the stub file
        nisarqa.output_stub_files(output_dir=out_dir, stub_files="summary_csv")
        print(f"Input file validation PASS/FAIL checks saved: {summary_file}")
        print("Input file validation complete.")

    if root_params.workflows.qa_reports:
        print("Beginning processing of `qa_reports` items...")

        product = nisarqa.GUNW(input_file)

        # TODO qa_reports will add to the SUMMARY.csv file.
        # For now, make sure that the stub file is output
        if not os.path.isfile(summary_file):
            nisarqa.output_stub_files(
                output_dir=root_params.get_output_dir(),
                stub_files="summary_csv",
            )
            print(f"PASS/FAIL checks saved to: {summary_file}")

        nisarqa.write_latlonquad_to_kml(
            llq=nisarqa.get_latlonquad(product=product),
            output_dir=root_params.get_output_dir(),
            kml_filename=root_params.get_kml_browse_filename(),
            png_filename=root_params.get_browse_png_filename(),
        )
        print("Processing of browse image kml complete.")
        print(f"Browse image kml file saved to {browse_file_kml}")

        with nisarqa.open_h5_file(stats_file, mode="w") as stats_h5, PdfPages(
            report_file
        ) as report_pdf:
            # Save the processing parameters to the stats.h5 file
            root_params.save_processing_params_to_stats_h5(
                h5_file=stats_h5, band=product.band
            )
            print(f"QA Processing Parameters saved to {stats_file}")

            nisarqa.rslc.copy_identification_group_to_stats_h5(
                product=product, stats_h5=stats_h5
            )
            print(f"Input file Identification group copied to {stats_file}")

            # Save frequency/polarization info to stats file
            product.save_qa_metadata_to_h5(stats_h5=stats_h5)

            # Save Browse Image
            nisarqa.make_hsi_browse_unwrapped(
                product=product,
                params=root_params.hsi,
                browse_png=browse_file_png,
            )

            # Save HSI images to PDF
            nisarqa.hsi_images_to_pdf_unwrapped(
                product=product,
                report_pdf=report_pdf,
                rewrap=root_params.hsi.rewrap,
            )

    print(
        "Successful completion of QA SAS. Check log file for validation"
        " warnings and errors."
    )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
