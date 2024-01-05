from __future__ import annotations

import os
from collections.abc import Mapping
from typing import overload

import h5py
from matplotlib.backends.backend_pdf import PdfPages

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def verify_igram(
    user_rncfg: Mapping[str, Mapping], product_type: str, verbose: bool = False
) -> None:
    """
    Verify a RIFG, RUNW, or GUNW product per provided runconfig.

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
        One of: "rifg", "runw", or "gunw".
    verbose : bool, optional
        True to stream log messages to console (stderr) in addition to the
        log file. False to only stream to the log file. (Initial log messages
        during setup will stream to console regardless.) Defaults to False.
    """
    supported_product_types = ("rifg", "runw", "gunw")
    if product_type not in supported_product_types:
        raise ValueError(
            f"{product_type=}, must be one of {supported_product_types}."
        )

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
    log.info(msg)
    if not verbose:
        print(msg)

    if product_type == "rifg":
        product = nisarqa.RIFG(input_file)
    elif product_type == "runw":
        product = nisarqa.RUNW(input_file)
    else:
        assert product_type == "gunw"
        product = nisarqa.GUNW(input_file)

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

            if isinstance(product, nisarqa.UnwrappedGroup):
                nisarqa.process_phase_image_unwrapped(
                    product=product,
                    params=root_params.unw_phs_img,
                    report_pdf=report_pdf,
                    stats_h5=stats_h5,
                )

            if isinstance(product, nisarqa.WrappedGroup):
                nisarqa.process_phase_image_wrapped(
                    product=product,
                    report_pdf=report_pdf,
                    stats_h5=stats_h5,
                )

            if isinstance(product, nisarqa.UnwrappedGroup):
                nisarqa.process_ionosphere_phase_screen(
                    product=product,
                    report_pdf=report_pdf,
                    stats_h5=stats_h5,
                )

            # Plot azimuth offsets and slant range offsets (RIFG, RUNW, & GUNW)
            nisarqa.process_az_and_slant_rg_offsets_from_igram_product(
                product=product,
                report_pdf=report_pdf,
                stats_h5=stats_h5,
            )

            # Save Browse Image and HSI Plots
            process_hsi(
                product=product,
                params=root_params.hsi,
                browse_png=browse_file_png,
                report_pdf=report_pdf,
                wrapped_hsi=True if product_type == "rifg" else False,
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


@overload
def process_hsi(
    product: nisarqa.WrappedGroup,
    params: nisarqa.HSIImageParamGroup,
    browse_png: str | os.PathLike,
    report_pdf: PdfPages,
    wrapped_hsi: bool,
) -> None: ...


@overload
def process_hsi(
    product: nisarqa.UnwrappedGroup,
    params: nisarqa.UNWHSIImageParamGroup,
    browse_png: str | os.PathLike,
    report_pdf: PdfPages,
    wrapped_hsi: bool,
) -> None: ...


def process_hsi(product, params, browse_png, report_pdf, wrapped_hsi):
    """
    Thin wrapper to make HSI browse PNGs and PDFs for interferogram products.

    Based on the input parameters, this function calls
    `make_hsi_browse_[un]wrapped()` and `hsi_images_to_pdf_[un]wrapped().

    Parameters
    ----------
    product : nisarqa.WrappedGroup or nisarqa.UnwrappedGroup
        Input NISAR product.
    params : nisarqa.HSIImageParamGroup or nisarqa.UNWHSIImageParamGroup
        A structure containing the parameters for creating the HSI image.
    browse_png : path-like
        Filename (with path) for the browse image PNG.
    report_pdf : PdfPages
        The output PDF file to append the HSI image plot to.
    wrapped_hsi : bool
        True to produce a plot of the wrapped interferogram in the product,
        False to produce a plot of the unwrapped interferogram in the product.
        As of R3.4, RIFG only contains the wrapped interferogram group,
        RUNW only contains the unwrapped interferogram group,
        and GUNW contains both.
        If `wrapped_hsi` conflicts with the type of input product, a
        TypeError will be raised.
    """
    if wrapped_hsi:
        if not isinstance(product, nisarqa.WrappedGroup):
            raise TypeError(
                f"`product` type is {type(product)}, must be WrappedGroup"
                " because `wrapped_hsi` is set to True."
            )
        if not isinstance(params, nisarqa.HSIImageParamGroup):
            raise TypeError(
                f"`params` type is {type(params)}, must be HSIImageParamGroup"
                " because `wrapped_hsi` is set to True."
            )
    else:
        if not isinstance(product, nisarqa.UnwrappedGroup):
            raise TypeError(
                f"`product` type is {type(product)}, must be"
                " UnwrappedGroup because `wrapped_hsi` is set to False."
            )
        if not isinstance(params, nisarqa.UNWHSIImageParamGroup):
            raise TypeError(
                f"`params` type is {type(params)}, must be"
                " UNWHSIImageParamGroup because `wrapped_hsi` is set to False."
            )

    if wrapped_hsi:
        nisarqa.make_hsi_browse_wrapped(
            product=product,
            params=params,
            browse_png=browse_png,
        )

        nisarqa.hsi_images_to_pdf_wrapped(
            product=product, report_pdf=report_pdf
        )
    else:
        nisarqa.make_hsi_browse_unwrapped(
            product=product,
            params=params,
            browse_png=browse_png,
        )
        nisarqa.hsi_images_to_pdf_unwrapped(
            product=product,
            report_pdf=report_pdf,
            rewrap=params.rewrap,
        )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
