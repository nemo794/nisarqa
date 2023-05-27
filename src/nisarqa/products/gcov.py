import os
from dataclasses import fields

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
        gcov_params = nisarqa.build_root_params(
            product_type="gcov", user_rncfg=user_rncfg
        )
    except nisarqa.ExitEarly as e:
        # No workflows were requested. Exit early.
        print(
            "All `workflows` were set to `False` in the runconfig. No QA "
            "processing will be performed. Exiting..."
        )
        return

    output_dir = gcov_params.prodpath.qa_output_dir

    # Print final processing parameters that will be used for QA, for debugging
    print(
        "QA Processing parameters, per runconfig and defaults (runconfig has precedence)"
    )

    gcov_params_names = {
        "input_f": "Input File Group",
        "prodpath": "Product Path Group",
        "workflows": "Workflows",
        "backscatter_img": "Backscatter Image",
        "histogram": "Histogram",
    }

    for params_obj in fields(gcov_params):
        grp_name = gcov_params_names[params_obj.name]
        print(f"  {grp_name} Input Parameters:")

        po = getattr(gcov_params, params_obj.name)
        if po is not None:
            for param in fields(po):
                po2 = getattr(po, param.name)
                if isinstance(po2, bool):
                    print(f"    {param.name}: {po2}")
                else:
                    print(f"    {param.name}: {po2}")

    output_dir = gcov_params.prodpath.qa_output_dir

    # Start logger
    # TODO get logger from Brian's code and implement here
    # For now, output the stub log file.
    nisarqa.output_stub_files(output_dir=output_dir, stub_files="log_txt")

    # Create file paths for output files ()
    input_file = gcov_params.input_f.qa_input_file
    msg = (
        f"Starting Quality Assurance for input file: {input_file}"
        f"\nOutputs to be generated:"
    )
    if gcov_params.workflows.validate or gcov_params.workflows.qa_reports:
        summary_file = os.path.join(output_dir, "SUMMARY.csv")
        msg += f"\n\tSummary file: {summary_file}"

    if gcov_params.workflows.qa_reports:
        stats_file = os.path.join(output_dir, "STATS.h5")
        msg += f"\n\tMetrics file: {stats_file}"
        report_file = os.path.join(output_dir, "REPORT.pdf")
        browse_image = os.path.join(output_dir, "BROWSE.png")
        browse_kml = os.path.join(output_dir, "BROWSE.kml")

        msg += (
            f"\n\tReport file: {report_file}"
            f"\n\tBrowse Image: {browse_image}"
            f"\n\tBrowse Image Geolocation file: {browse_kml}"
        )
    print(msg)

    # Begin QA workflows
    with nisarqa.open_h5_file(input_file, mode="r") as in_file:
        # Run the requested workflows
        if gcov_params.workflows.validate:
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
            nisarqa.output_stub_files(output_dir=output_dir, stub_files="summary_csv")

        if not gcov_params.workflows.qa_reports:
            print(
                "Successful completion. Check log file for validation warnings and errors."
            )
            return

        # First time opening the STATS.h5 file, so open in 'w' mode.
        with nisarqa.open_h5_file(stats_file, mode="w") as stats_h5, PdfPages(
            report_file
        ) as report_pdf:
            # Note: `pols` contains references to datasets in the open input file.
            # All processing with `pols` must be done within this context manager,
            # or the references will be closed and inaccessible.
            pols = nisarqa.rslc.get_pols(in_file)

            # Save frequency/polarization info from `pols` to stats file
            nisarqa.rslc.save_nisar_freq_metadata_to_h5(stats_h5=stats_h5, pols=pols)

            # Save the processing parameters to the stats.h5 file
            gcov_params.save_params_to_stats_file(
                h5_file=stats_h5, bands=tuple(pols.keys())
            )

            # Copy the Product identification group to STATS.h5
            nisarqa.rslc.save_NISAR_identification_group_to_h5(
                nisar_h5=in_file, stats_h5=stats_h5
            )

            # TODO qa_reports will add to the SUMMARY.csv file.
            # For now, make sure that the stub file is output
            if not os.path.isfile(summary_file):
                nisarqa.output_stub_files(
                    output_dir=output_dir, stub_files="summary_csv"
                )

            # TODO qa_reports will create the BROWSE.kml file.
            # For now, make sure that the stub file is output
            nisarqa.output_stub_files(output_dir=output_dir, stub_files="browse_kml")

            input_raster_represents_power = True
            name_of_backscatter_content = "GCOV Backscatter Coefficient (gamma0)"

            # Generate the Backscatter Image and Browse Image
            nisarqa.rslc.process_backscatter_imgs_and_browse(
                pols=pols,
                params=gcov_params.backscatter_img,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                product_type="gcov",
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
                browse_filename=browse_image,
            )

            # Generate the Backscatter and Phase Histograms
            nisarqa.rslc.process_backscatter_and_phase_histograms(
                pols=pols,
                params=gcov_params.histogram,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
            )

            # Check for invalid values

            # Compute metrics for stats.h5

    print("Successful completion. Check log file for validation warnings and errors.")


__all__ = nisarqa.get_all(__name__, objects_to_skip)
