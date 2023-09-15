import os
import warnings

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def verify_gcov(user_rncfg):
    """
    Verify an GCOV product based on the input file, parameters, etc.
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
        A nested dict whose structure matches this product's QA runconfig
        yaml file and which contains the parameters needed to run its QA SAS.
    """

    # Build the GCOVRootParamGroup parameters per the runconfig
    try:
        root_params = nisarqa.build_root_params(
            product_type="gcov", user_rncfg=user_rncfg
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

    # For readibility, store possible output filenames in variables.
    input_file = root_params.input_f.qa_input_file
    out_dir = root_params.get_output_dir()
    browse_file_png = out_dir / root_params.get_browse_png_filename()
    browse_file_kml = out_dir / root_params.get_kml_browse_filename()
    report_file = out_dir / root_params.get_report_pdf_filename()
    stats_file = out_dir / root_params.get_stats_h5_filename()
    summary_file = out_dir / root_params.get_summary_csv_filename()

    print(f"Starting Quality Assurance for input file: {input_file}")

    if root_params.workflows.validate:
        # TODO Validate file structure
        # (After this, we can assume the file structure for all
        # subsequent accesses to it)
        # NOTE: Refer to the original 'get_bands()' to check that in_file
        # contains metadata, swaths, Identification groups, and that it
        # is SLC/RSLC compliant. These should trigger a fatal error!
        # NOTE: Refer to the original get_freq_pol() for the verification
        # checks. This could trigger a fatal error!

        # These reports will be saved to the SUMMARY.csv file.
        # For now, output the stub file
        nisarqa.output_stub_files(output_dir=out_dir, stub_files="summary_csv")
        print(f"Input file validation PASS/FAIL checks saved: {summary_file}")
        print(f"Input file validation complete.")

        # TODO - this GCOV validation check should be integrated into
        # the actual product validation. For now, we'll leave it here.
        product = nisarqa.GCOV(input_file)
        for freq in product.freqs:
            for pol in product.get_pols(freq=freq):
                if pol in nisarqa.GCOV_DIAG_POLS:
                    continue
                elif pol in nisarqa.GCOV_OFF_DIAG_POLS:
                    warnings.warn(
                        f"GCOV product contains off-diagonal term {pol}.",
                        RuntimeWarning,
                    )

    if root_params.workflows.qa_reports:
        print("Beginning processing of `qa_reports` items...")

        product = nisarqa.GCOV(input_file)

        # TODO qa_reports will add to the SUMMARY.csv file.
        # For now, make sure that the stub file is output
        if not os.path.isfile(summary_file):
            nisarqa.output_stub_files(
                output_dir=out_dir,
                stub_files="summary_csv",
            )
            print(
                f"Input file validation PASS/FAIL checks saved: {summary_file}"
            )
            print(f"Input file validation complete.")

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
            root_params.save_params_to_stats_h5(
                h5_file=stats_h5, band=product.band
            )
            print(f"QA Processing Parameters saved to {stats_file}")

            nisarqa.rslc.copy_identification_group_to_stats_h5(
                product=product, stats_h5=stats_h5
            )
            print(f"Input file Identification group copied to {stats_file}")

            # Save frequency/polarization info from `pols` to stats file
            nisarqa.rslc.save_nisar_freq_metadata_to_h5(
                product=product, stats_h5=stats_h5
            )

            input_raster_represents_power = True
            name_of_backscatter_content = (
                r"GCOV Backscatter Coefficient ($\gamma^0$)"
            )

            # Generate the Backscatter Image and Browse Image
            nisarqa.rslc.process_backscatter_imgs_and_browse(
                product=product,
                params=root_params.backscatter_img,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
                browse_filename=browse_file_png,
            )
            print("Processing of Backscatter images complete.")
            print(f"Browse image PNG file saved to {browse_file_png}")

            # Generate the Backscatter and Phase Histograms
            nisarqa.rslc.process_backscatter_and_phase_histograms(
                product=product,
                params=root_params.histogram,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
            )
            print("Processing of backscatter and phase histograms complete.")

            # Check for invalid values

            # Compute metrics for stats.h5

            print(f"PDF reports saved to {report_file}")
            print(f"HDF5 statistics saved to {stats_file}")
            print(f"CSV Summary PASS/FAIL checks saved to {summary_file}")
            print("`qa_reports` processing complete.")

    print(
        "Successful completion of QA SAS. Check log file for validation"
        " warnings and errors."
    )


# Dear Reviewer: PLEASE DELETE THIS COMMENT!
# These two functions were incorporated into the GCOV Product reader.

__all__ = nisarqa.get_all(__name__, objects_to_skip)
