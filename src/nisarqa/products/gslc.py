import os
from dataclasses import fields

from matplotlib.backends.backend_pdf import PdfPages

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def verify_gslc(user_rncfg):
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
    user_rncfg : dict
        A dictionary whose structure matches an this product's QA runconfig
        yaml file and which contains the parameters needed to run its QA SAS.
    """

    # Build the GSLCRootParamGroup parameters per the runconfig
    try:
        root_params = nisarqa.build_root_params(
            product_type="gslc", user_rncfg=user_rncfg
        )
    except nisarqa.ExitEarly as e:
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
    browse_file_png = (
        root_params.get_output_dir() / root_params.get_browse_png_filename()
    )
    browse_file_kml = (
        root_params.get_output_dir() / root_params.get_kml_browse_filename()
    )
    report_file = (
        root_params.get_output_dir() / root_params.get_report_pdf_filename()
    )
    stats_file = (
        root_params.get_output_dir() / root_params.get_stats_h5_filename()
    )
    summary_file = (
        root_params.get_output_dir() / root_params.get_summary_csv_filename()
    )

    print(f"Starting Quality Assurance for input file: {input_file}")

    if root_params.workflows.validate:
        print(f"Beginning input file validation...")

        # TODO Validate file structure
        # (After this, we can assume the file structure for all
        # subsequent accesses to it)
        # NOTE: Refer to the original get_freq_pol() for the verification
        # checks. This could trigger a fatal error!

        # These reports will be saved to the SUMMARY.csv file.
        # For now, output the stub file
        nisarqa.output_stub_files(
            output_dir=root_params.get_output_dir(),
            stub_files="summary_csv",
        )

        print(
            f"Input file validation PASS/FAIL checks saved to {summary_file}"
        )
        print(f"Input file validation complete.")

    if root_params.workflows.qa_reports:
        print(f"Beginning `qa_reports` processing...")

        # TODO qa_reports will add to the SUMMARY.csv file.
        # For now, make sure that the stub file is output
        if not os.path.isfile(summary_file):
            nisarqa.output_stub_files(
                output_dir=root_params.get_output_dir(),
                stub_files="summary_csv",
            )

        # TODO qa_reports will create the BROWSE.kml file.
        # For now, make sure that the stub file is output
        nisarqa.output_stub_files(
            output_dir=root_params.get_output_dir(),
            stub_files="browse_kml",
        )
        print("Processing of browse image kml complete.")
        print(f"Browse image kml file saved to {browse_file_kml}")

        with nisarqa.open_h5_file(
            input_file, mode="r"
        ) as in_file, nisarqa.open_h5_file(
            stats_file, mode="w"
        ) as stats_h5, PdfPages(
            report_file
        ) as report_pdf:
            # Note: `pols` contains references to datasets in the open input file.
            # All processing with `pols` must be done within this context manager,
            # or the references will be closed and inaccessible.
            print("Beginning processing of `qa_reports` items...")

            pols = nisarqa.rslc.get_pols(in_file)

            # Save the processing parameters to the stats.h5 file
            root_params.save_params_to_stats_file(
                h5_file=stats_h5, bands=tuple(pols.keys())
            )

            # Copy the Product identification group to STATS.h5
            nisarqa.rslc.save_NISAR_identification_group_to_h5(
                nisar_h5=in_file, stats_h5=stats_h5
            )
            print(f"Input file Identification group copied to {stats_file}")

            # Save frequency/polarization info to stats file
            nisarqa.rslc.save_nisar_freq_metadata_to_h5(
                stats_h5=stats_h5, pols=pols
            )
            print(f"QA Processing Parameters saved to {stats_file}")

            input_raster_represents_power = False
            name_of_backscatter_content = (
                r"GSLC Backscatter Coefficient ($\beta^0$)"
            )

            # Generate the GSLC Power Image and Browse Image
            nisarqa.rslc.process_backscatter_imgs_and_browse(
                pols=pols,
                params=root_params.backscatter_img,
                product_type="gslc",
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
                browse_filename=browse_file_png,
            )
            print("Processing of power images complete.")
            print(f"Browse image PNG file saved to {browse_file_png}")

            # Generate the GSLC Power and Phase Histograms
            nisarqa.rslc.process_backscatter_and_phase_histograms(
                pols=pols,
                params=root_params.histogram,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                input_raster_represents_power=input_raster_represents_power,
            )
            print("Processing of power and phase histograms complete.")

            # Process Interferograms

            # Check for invalid values

            # Compute metrics for stats.h5

            print(f"PDF reports saved to {report_file}")
            print(f"HDF5 statistics saved to {stats_file}")
            print(f"CSV Summary PASS/FAIL checks saved to {summary_file}")
            print("`qa_reports` processing complete.")

    print(
        "Successful completion of QA SAS. Check log file for validation warnings and errors."
    )


def save_geocoded_backscatter_img_to_pdf(
    img_arr,
    img,
    params,
    report_pdf,
    plot_title_prefix="Backscatter Coefficient",
    colorbar_formatter=None,
):
    """
    Annotate and save a Geocoded Backscatter Image to `report_pdf`.

    Parameters
    ----------
    img_arr : numpy.ndarray
        2D image array to be saved. All image correction, multilooking, etc.
        needs to have previously been applied
    img : GeoRaster
        The GeoRaster object that corresponds to `img`. The metadata
        from this will be used for annotating the image plot.
    params : BackscatterImageParamGroup
        A structure containing the parameters for processing
        and outputting the backscatter image(s).
    report_pdf : PdfPages
        The output pdf file to append the backscatter image plot to
    plot_title_prefix : str, optional
        Prefix for the title of the backscatter plots.
        Suggestions: "RSLC Backscatter Coefficient (beta-0)" or
        "GCOV Backscatter Coefficient (gamma-0)"
        Defaults to "Backscatter Coefficient"
    colorbar_formatter : matplotlib.ticker.FuncFormatter or None, optional
        Tick formatter function to define how the numeric value
        associated with each tick on the colorbar axis is formatted
        as a string. This function must take exactly two arguments:
        `x` for the tick value and `pos` for the tick position,
        and must return a `str`. The `pos` argument is used
        internally by matplotlib.
        If None, then default tick values will be used. Defaults to None.
        See: https://matplotlib.org/2.0.2/examples/pylab_examples/custom_ticker1.html
    """

    # Plot and Save Backscatter Image to graphical summary pdf
    title = f"{plot_title_prefix}\n(scale={params.backscatter_units}%s)\n{img.name}"
    if params.gamma is None:
        title = title % ""
    else:
        title = title % rf", $\gamma$-correction={params.gamma}"

    # TODO: double-check that start and stop were parsed correctly from the metadata

    nisarqa.rslc.img2pdf(
        img_arr=img_arr,
        title=title,
        ylim=[nisarqa.m2km(img.y_start), nisarqa.m2km(img.y_stop)],
        xlim=[nisarqa.m2km(img.x_start), nisarqa.m2km(img.x_stop)],
        colorbar_formatter=colorbar_formatter,
        ylabel="Northing (km)",
        xlabel="Easting (km)",
        plots_pdf=report_pdf,
    )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
