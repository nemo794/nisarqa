from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from typing import Optional

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from numpy.typing import ArrayLike
from PIL import Image
from scipy import constants

import nisarqa

# List of objects from the import statements that
# should not be included when importing this module
objects_to_skip = nisarqa.get_all(name=__name__)


def verify_rslc(user_rncfg):
    """
    Verify an RSLC product based on the input file, parameters, etc.
    specified in the input runconfig file.

    This is the main function for running the entire QA workflow. It will
    run based on the options supplied in the input runconfig file.
    The input runconfig file must follow the standard RSLC QA runconfig
    format. Run the command line command 'nisar_qa dumpconfig rslc'
    for an example template with default parameters (where available).

    Parameters
    ----------
    user_rncfg : dict
        A dictionary whose structure matches this product's QA runconfig
        YAML file and which contains the parameters needed to run its QA SAS.
    """

    # Build the RSLCRootParamGroup parameters per the runconfig
    try:
        root_params = nisarqa.build_root_params(
            product_type="rslc", user_rncfg=user_rncfg
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
    out_dir = root_params.get_output_dir()
    nisarqa.output_stub_files(output_dir=out_dir, stub_files="log_txt")

    # Log the values of the parameters.
    # Currently, this prints to stdout. Once the logger is implemented,
    # it should log the values directly to the log file.
    root_params.log_parameters()

    # For readibility, store output filenames in variables.
    # Depending on which workflows are set to True, not all filename
    # variables will be used.
    input_file = root_params.input_f.qa_input_file
    browse_file_png = out_dir / root_params.get_browse_png_filename()
    browse_file_kml = out_dir / root_params.get_kml_browse_filename()
    report_file = out_dir / root_params.get_report_pdf_filename()
    stats_file = out_dir / root_params.get_stats_h5_filename()
    summary_file = out_dir / root_params.get_summary_csv_filename()

    print(f"Starting Quality Assurance for input file: {input_file}")

    # Begin QA workflows

    # Run validate first because it checks the product spec
    if root_params.workflows.validate:
        print(f"Beginning input file validation...")
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
        nisarqa.output_stub_files(
            output_dir=out_dir,
            stub_files="summary_csv",
        )
        print(f"Input file validation PASS/FAIL checks saved to {summary_file}")
        print(f"Input file validation complete.")

    # If running these workflows, save the processing parameters and
    # identification group to STATS.h5
    if (
        root_params.workflows.qa_reports
        or root_params.workflows.abs_cal
        or root_params.workflows.noise_estimation
        or root_params.workflows.point_target
    ):
        # This is the first time opening the STATS.h5 file for RSLC
        # workflow, so open in 'w' mode.
        # After this, always open STATS.h5 in 'r+' mode.
        with nisarqa.open_h5_file(stats_file, mode="w") as stats_h5:
            product = nisarqa.RSLC(input_file)

            # Save the processing parameters to the stats.h5 file
            # Note: If only the validate workflow is requested,
            # this will do nothing.
            root_params.save_processing_params_to_stats_h5(
                h5_file=stats_h5, band=product.band
            )
            print(f"QA Processing Parameters saved to {stats_file}")

            nisarqa.rslc.copy_identification_group_to_stats_h5(
                product=product, stats_h5=stats_h5
            )
            print(f"Input file Identification group copied to {stats_file}")

    if root_params.workflows.qa_reports:
        print(f"Beginning `qa_reports` processing...")
        # TODO qa_reports will add to the SUMMARY.csv file.
        # For now, make sure that the stub file is output
        if not os.path.isfile(summary_file):
            nisarqa.output_stub_files(
                output_dir=out_dir,
                stub_files="summary_csv",
            )

        nisarqa.write_latlonquad_to_kml(
            llq=product.get_browse_latlonquad(),
            output_dir=out_dir,
            kml_filename=root_params.get_kml_browse_filename(),
            png_filename=root_params.get_browse_png_filename(),
        )
        print("Processing of browse image kml complete.")
        print(f"Browse image kml file saved to {browse_file_kml}")

        with nisarqa.open_h5_file(stats_file, mode="r+") as stats_h5, PdfPages(
            report_file
        ) as report_pdf:
            product = nisarqa.RSLC(filepath=input_file)

            # Save frequency/polarization info to stats file
            save_nisar_freq_metadata_to_h5(stats_h5=stats_h5, product=product)

            input_raster_represents_power = False
            name_of_backscatter_content = (
                r"RSLC Backscatter Coefficient ($\beta^0$)"
            )

            # Generate the RSLC Backscatter Image and Browse Image
            process_backscatter_imgs_and_browse(
                product=product,
                params=root_params.backscatter_img,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
                browse_filename=browse_file_png,
            )
            print("Processing of backscatter images complete.")
            print(f"Browse image PNG file saved to {browse_file_png}")

            # Generate the RSLC Power and Phase Histograms
            process_backscatter_and_phase_histograms(
                product=product,
                params=root_params.histogram,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
                plot_title_prefix=name_of_backscatter_content,
                input_raster_represents_power=input_raster_represents_power,
            )
            print("Processing of backscatter and phase histograms complete.")

            # Process Interferograms

            # Generate Spectra
            process_range_spectra(
                product=product,
                params=root_params.range_spectra,
                stats_h5=stats_h5,
                report_pdf=report_pdf,
            )

            # Check for invalid values

            # Compute metrics for stats.h5

            print(f"PDF reports saved to {report_file}")
            print(f"HDF5 statistics saved to {stats_file}")
            print(f"CSV Summary PASS/FAIL checks saved to {summary_file}")
            print("`qa_reports` processing complete.")

    if root_params.workflows.abs_cal:
        print("Beginning Absolute Radiometric Calibration CalTool...")

        # Run Absolute Radiometric Calibration tool
        nisarqa.caltools.run_abscal_tool(
            abscal_params=root_params.abs_cal,
            dyn_anc_params=root_params.anc_files,
            input_filename=input_file,
            stats_filename=stats_file,
        )
        print(
            "Absolute Radiometric Calibration CalTool results saved to"
            f" {stats_file}"
        )
        print("Absolute Radiometric Calibration CalTool complete.")

    if root_params.workflows.noise_estimation:
        print("Beginning Noise Estimation Tool CalTool...")

        # Run NET tool
        nisarqa.caltools.run_noise_estimation_tool(
            params=root_params.noise_estimation,
            input_filename=input_file,
            stats_filename=stats_file,
        )
        print(f"Noise Estimation Tool CalTool results saved to {stats_file}")
        print("Noise Estimation Tool CalTool complete.")

    if root_params.workflows.point_target:
        print("Beginning Point Target Analyzer CalTool...")

        # Run Point Target Analyzer tool
        nisarqa.caltools.run_pta_tool(
            pta_params=root_params.pta,
            dyn_anc_params=root_params.anc_files,
            input_filename=input_file,
            stats_filename=stats_file,
        )
        print(f"Point Target Analyzer CalTool results saved to {stats_file}")
        print("Point Target Analyzer CalTool complete.")

    print(
        "Successful completion of QA SAS. Check log file for validation"
        " warnings and errors."
    )


# TODO - move to generic NISAR module
def copy_identification_group_to_stats_h5(
    product: nisarqa.NisarProduct, stats_h5: h5py.File
) -> None:
    """
    Copy the identification group from the input NISAR file
    to the STATS.h5 file.

    Parameters
    ----------
    product : nisarqa.NisarProduct
        Instance of a NisarProduct
    stats_h5 : h5py.File
        Handle to an h5 file where the identification metadata
        should be saved
    """

    src_grp_path = product.identification_path
    dest_grp_path = nisarqa.STATS_H5_IDENTIFICATION_GROUP % product.band

    with nisarqa.open_h5_file(product.filepath, "r") as in_file:
        if dest_grp_path in stats_h5:
            # The identification group already exists, so copy each
            # dataset, etc. individually
            for item in in_file[src_grp_path]:
                item_path = f"{dest_grp_path}/{item}"
                in_file.copy(in_file[item_path], stats_h5, item_path)
        else:
            # Copy entire identification metadata from input file to stats.h5
            in_file.copy(in_file[src_grp_path], stats_h5, dest_grp_path)


# TODO - move to generic NISAR module
def save_nisar_freq_metadata_to_h5(
    product: nisarqa.NonInsarProduct, stats_h5: h5py.File
) -> None:
    """
    Populate the `stats_h5` HDF5 file with a list of each available
    frequency's polarizations.

    If `pols` contains values for Frequency A, then this dataset will
    be created in `stats_h5`:
        /science/<band>/QA/data/frequencyA/listOfPolarizations

    If `pols` contains values for Frequency B, then this dataset will
    be created in `stats_h5`:
        /science/<band>/QA/data/frequencyB/listOfPolarizations

    * Note: The paths are pulled from the global nisarqa.STATS_H5_QA_FREQ_GROUP.
    If the value of that global changes, then the path for the
    `listOfPolarizations` dataset(s) will change accordingly.

    Parameters
    ----------
    product : nisarqa.NisarProduct
        Input NISAR product
    stats_h5 : h5py.File
        Handle to an h5 file where the list(s) of polarizations should be saved
    """
    # Populate data group's metadata
    for freq in product.freqs:
        list_of_pols = product.get_pols(freq=freq)
        grp_path = nisarqa.STATS_H5_QA_FREQ_GROUP % (product.band, freq)
        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name="listOfPolarizations",
            ds_data=list_of_pols,
            ds_description=(
                f"Polarizations for Frequency {freq} "
                "discovered in input NISAR product by QA code"
            ),
        )


def process_backscatter_imgs_and_browse(
    product: nisarqa.NonInsarProduct,
    params: nisarqa.BackscatterImageParamGroup,
    stats_h5: h5py.File,
    report_pdf: PdfPages,
    browse_filename: str,
    input_raster_represents_power: bool = False,
    plot_title_prefix: str = "Backscatter Coefficient",
) -> None:
    """
    Generate Backscatter Image plots for the `report_pdf` and
    corresponding browse image product.

    Parameters
    ----------
    product : nisarqa.NonInsarProduct
        The input NISAR product
    params : BackscatterImageParamGroup
        A dataclass containing the parameters for processing
        and outputting backscatter image(s) and browse image.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output PDF file to append the backscatter image plot to
    browse_filename : str
        Filename (with path) for the browse image PNG.
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either power or root power.
        If `True`, then QA SAS assumes the input data already represents
        power and uses the pixels' magnitudes for computations.
        If `False`, then QA SAS assumes the input data represents root power
        aka magnitude and will handle the full computation to power using
        the formula:  power = abs(<magnitude>)^2 .
        Defaults to False (root power).
    plot_title_prefix : str
        Prefix for the title of the backscatter plots.
        Suggestions: "RSLC Backscatter Coefficient (beta-0)" or
        "GCOV Backscatter Coefficient (gamma-0)".
        Defaults to "Backscatter Coefficient".
    """

    # Select which layers will be needed for the browse image.
    # Multilooking takes a long time, but each multilooked polarization image
    # should be less than ~4 MB (per the current requirements for NISAR),
    # so it's ok to store the necessary multilooked Backscatter Images in memory.
    # to combine them later into the Browse image. The memory costs are
    # less than the costs for re-computing the multilooking.
    layers_for_browse = product.get_layers_for_browse()

    # At the end of the loop below, the keys of this dict should exactly
    # match the set of TxRx polarizations needed to form the browse image
    pol_imgs_for_browse = {}

    # Process each image in the dataset

    for freq in product.freqs:
        for pol in product.get_pols(freq=freq):
            # Open the *SARRaster image

            with product.get_raster(freq=freq, pol=pol) as img:
                multilooked_img = get_multilooked_backscatter_img(
                    img=img,
                    params=params,
                    stats_h5=stats_h5,
                    input_raster_represents_power=input_raster_represents_power,
                )

                corrected_img, orig_vmin, orig_vmax = apply_image_correction(
                    img_arr=multilooked_img, params=params
                )

                if params.gamma is not None:
                    inverse_func = functools.partial(
                        invert_gamma_correction,
                        gamma=params.gamma,
                        vmin=orig_vmin,
                        vmax=orig_vmax,
                    )

                    colorbar_formatter = FuncFormatter(
                        lambda x, pos: "{:.3f}".format(inverse_func(x))
                    )

                else:
                    colorbar_formatter = None

                # Label and Save Backscatter Image to PDF
                with product.get_raster(freq=freq, pol=pol) as img:
                    title = (
                        f"{plot_title_prefix}\n"
                        f"(scale={params.backscatter_units}%s)\n{img.name}"
                    )
                    if params.gamma is None:
                        title = title % ""
                    else:
                        title = title % rf", $\gamma$-correction={params.gamma}"

                    nisarqa.rslc.img2pdf_grayscale(
                        img_arr=corrected_img,
                        title=title,
                        ylim=img.y_axis_limits,
                        xlim=img.x_axis_limits,
                        colorbar_formatter=colorbar_formatter,
                        ylabel=img.y_axis_label,
                        xlabel=img.x_axis_label,
                        plots_pdf=report_pdf,
                    )

                # If this backscatter image is needed to construct the browse image...
                if (freq in layers_for_browse) and (
                    pol in layers_for_browse[freq]
                ):
                    # ...keep the multilooked, color-corrected image in memory
                    pol_imgs_for_browse[pol] = corrected_img

    # Construct the browse image
    product.save_browse(pol_imgs=pol_imgs_for_browse, filepath=browse_filename)


# TODO - move to generic location
def get_multilooked_backscatter_img(
    img, params, stats_h5, input_raster_represents_power=False
):
    """
    Generate the multilooked Backscatter Image array for a single
    polarization image.

    Parameters
    ----------
    img : RadarRaster or GeoRaster
        The raster to be processed
    params : BackscatterImageParamGroup
        A structure containing the parameters for processing
        and outputting the backscatter image(s).
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either power or root power.
        If `True`, then QA SAS assumes the input data already represents
        power and uses the pixels' magnitudes for computations.
        If `False`, then QA SAS assumes the input data represents root power
        aka magnitude and will handle the full computation to power using
        the formula:  power = abs(<magnitude>)^2 .
        Defaults to False (root power).

    Returns
    -------
    out_img : numpy.ndarray
        The multilooked Backscatter Image
    """

    nlooks_freqa_arg = params.nlooks_freqa
    nlooks_freqb_arg = params.nlooks_freqb

    # Get the window size for multilooking
    if (img.freq == "A" and nlooks_freqa_arg is None) or (
        img.freq == "B" and nlooks_freqb_arg is None
    ):
        nlooks = nisarqa.compute_square_pixel_nlooks(
            img.data.shape,
            sample_spacing=(
                np.abs(img.y_axis_spacing),
                np.abs(img.x_axis_spacing),
            ),
            longest_side_max=params.longest_side_max,
        )

    elif img.freq == "A":
        nlooks = nlooks_freqa_arg
    elif img.freq == "B":
        nlooks = nlooks_freqb_arg
    else:
        raise ValueError(
            f"frequency is '{img.freq}', but only 'A' or 'B' are valid options."
        )

    # Save the final nlooks to the HDF5 dataset
    grp_path = nisarqa.STATS_H5_QA_PROCESSING_GROUP % img.band
    dataset_name = f"backscatterImageNlooksFreq{img.freq.upper()}"

    if isinstance(img, nisarqa.RadarRaster):
        axes = "[<azimuth>,<range>]"
    elif isinstance(img, nisarqa.GeoRaster):
        axes = "[<Y direction>,<X direction>]"
    else:
        raise TypeError(
            f"Input `img` must be RadarRaster or GeoRaster. It is {type(img)}"
        )

    # Create the nlooks dataset
    if dataset_name in stats_h5[grp_path]:
        assert tuple(stats_h5[grp_path][dataset_name][...]) == tuple(nlooks)
    else:
        nisarqa.create_dataset_in_h5group(
            h5_file=stats_h5,
            grp_path=grp_path,
            ds_name=dataset_name,
            ds_data=nlooks,
            ds_units="unitless",
            ds_description=(
                f"Number of looks along {axes} axes of "
                f"Frequency {img.freq.upper()} image arrays "
                "for multilooking the backscatter and browse images."
            ),
        )

    print(f"\nMultilooking Image {img.name} with shape: {img.data.shape}")
    print("Y direction (azimuth) ground spacing: ", img.y_axis_spacing)
    print("X direction (range) ground spacing: ", img.x_axis_spacing)
    print("Beginning Multilooking with nlooks window shape: ", nlooks)

    # Multilook
    out_img = nisarqa.compute_multilooked_backscatter_by_tiling(
        arr=img.data,
        nlooks=nlooks,
        input_raster_represents_power=input_raster_represents_power,
        tile_shape=params.tile_shape,
    )

    print(f"Multilooking Complete. Multilooked image shape: {out_img.shape}")

    return out_img


def apply_image_correction(img_arr, params):
    """
    Apply image correction in `img_arr` as specified in `params`.

    Image correction is applied in the following order:
        Step 1: Per `params.percentile_for_clipping`, clip the image array's outliers
        Step 2: Per `params.linear_units`, convert from linear units to dB
        Step 3: Per `params.gamma`, apply gamma correction

    Parameters
    ----------
    img_arr : numpy.ndarray
        2D image array to have image correction applied to.
        For example, for RSLC this is the multilooked image array.
    params : BackscatterImageParamGroup
        A structure containing the parameters for processing
        and outputting the backscatter image(s).

    Returns
    -------
    out_img : numpy.ndarray
        2D image array. If any image correction was specified via `params`
        and applied to `img_arr`, this returned array will include that
        image correction.
    vmin, vmax : float
        The min and max of the image array (excluding Nan), as computed
        after Step 2 but before Step 3. These can be used to set
        colorbar tick mark values; by computing vmin and vmax prior to
        gamma correction, the tick marks values retain physical meaning.
    """

    # Step 1: Clip the image array's outliers
    img_arr = clip_array(
        img_arr, percentile_range=params.percentile_for_clipping
    )

    # Step 2: Convert from linear units to dB
    if not params.linear_units:
        img_arr = nisarqa.pow2db(img_arr)

    # Get the vmin and vmax prior to applying gamma correction.
    # These can later be used for setting the colorbar's
    # tick mark values.
    vmin = np.nanmin(img_arr)
    vmax = np.nanmax(img_arr)

    # Step 3: Apply gamma correction
    if params.gamma is not None:
        img_arr = apply_gamma_correction(img_arr, gamma=params.gamma)

    return img_arr, vmin, vmax


def clip_array(arr, percentile_range=(0.0, 100.0)):
    """
    Clip input array to the provided percentile range.

    NaN values are excluded from the computation of the percentile.

    Parameters
    ----------
    arr : array_like
        Input array
    percentile_range : pair of numeric, optional
        Defines the percentile range of the `arr`
        that the colormap covers. Must be in the range [0.0, 100.0],
        inclusive.
        Defaults to (0.0, 100.0) (no clipping).

    Returns
    -------
    out_img : numpy.ndarray
        A copy of the input array with the values outside of the
        range defined by `percentile_range` clipped.
    """
    for p in percentile_range:
        nisarqa.verify_valid_percentile(p)
    if len(percentile_range) != 2:
        raise ValueError(f"{percentile_range=} must have length of 2")

    # Get the value of the e.g. 5th percentile and the 95th percentile
    vmin, vmax = np.nanpercentile(arr, percentile_range)

    # Clip the image data and return
    return np.clip(arr, a_min=vmin, a_max=vmax)


def apply_gamma_correction(img_arr, gamma):
    """
    Apply gamma correction to the input array.

    Function will normalize the array and apply gamma correction.
    The returned output array will remain in range [0,1].

    Parameters
    ----------
    img_arr : array_like
        Input array
    gamma : float
        The gamma correction parameter.
        Gamma will be applied as follows:
            array_out = normalized_array ^ gamma
        where normalized_array is a copy of `img_arr` with values
        scaled to the range [0,1].

    Returns
    -------
    out_img : numpy.ndarray
        Copy of `img_arr` with the specified gamma correction applied.
        Due to normalization, values in `out_img` will be in range [0,1].

    Also See
    --------
    invert_gamma_correction : inverts this function
    """
    # Normalize to range [0,1]
    out_img = nisarqa.normalize(img_arr)

    # Apply gamma correction
    out_img = np.power(out_img, gamma)

    return out_img


def invert_gamma_correction(img_arr, gamma, vmin, vmax):
    """
    Invert the gamma correction to the input array.

    Function will normalize the array and apply gamma correction.
    The returned output array will remain in range [0,1].

    Parameters
    ----------
    img_arr : array_like
        Input array
    gamma : float
        The gamma correction parameter.
        Gamma will be inverted as follows:
            array_out = img_arr ^ (1/gamma)
        The array will then be rescaled as follows, to "undo" normalization:
            array_out = (array_out * (vmax - vmin)) + vmin
    vmin, vmax : float
        The min and max of the source image array BEFORE gamma correction
        was applied.

    Returns
    -------
    out : numpy.ndarray
        Copy of `img_arr` with the specified gamma correction inverted
        and scaled to range [<vmin>, <vmax>]

    Also See
    --------
    apply_gamma_correction : inverts this function
    """
    # Invert the power
    out = np.power(img_arr, 1 / gamma)

    # Invert the normalization
    out = (out * (vmax - vmin)) + vmin

    return out


def plot_to_grayscale_png(img_arr, filepath):
    """
    Save the image array to a 1-channel grayscale PNG with transparency.

    Finite pixels will have their values scaled to 1-255. Non-finite pixels
    will be set to 0 and made to appear transparent in the PNG.
    The pixel value of 0 is reserved for the transparent pixels.

    Parameters
    ----------
    img_arr : array_like
        2D Image to plot
    filepath : str
        Full filepath the browse image product.

    Notes
    -----
    This function does not add a full alpha channel to the output png.
    It instead uses "cheap transparency" (palette-based transparency)
    to keep file size smaller.
    See: http://www.libpng.org/pub/png/book/chapter08.html#png.ch08.div.5.4
    """

    # Only use 2D arrays
    if len(np.shape(img_arr)) != 2:
        raise ValueError("Input image array must be 2D.")

    img_arr, transparency_val = prep_arr_for_png_with_transparency(img_arr)

    # Save as grayscale image using PIL.Image. 'L' is grayscale mode.
    # (Pyplot only saves png's as RGB, even if cmap=plt.cm.gray)
    im = Image.fromarray(img_arr, mode="L")
    im.save(filepath, transparency=transparency_val)  # default = 72 dpi


def plot_to_rgb_png(red, green, blue, filepath):
    """
    Combine and save RGB channel arrays to a browse PNG with transparency.

    Finite pixels will have their values scaled to 1-255. Non-finite pixels
    will be set to 0 and made to appear transparent in the PNG.
    The pixel value of 0 is reserved for the transparent pixels.

    Parameters
    ----------
    red, green, blue : numpy.ndarray
        2D arrays that will be mapped to the red, green, and blue
        channels (respectively) for the PNG. These three arrays must have
        identical shape.
    filepath : str
        Full filepath for where to save the browse image PNG.

    Notes
    -----
    This function does not add a full alpha channel to the output png.
    It instead uses "cheap transparency" (palette-based transparency)
    to keep file size smaller.
    See: http://www.libpng.org/pub/png/book/chapter08.html#png.ch08.div.5.4
    """

    # Only use 2D arrays
    for arr in (red, green, blue):
        if len(np.shape(arr)) != 2:
            raise ValueError("Input image array must be 2D.")

    # Concatenate into uint8 RGB array.
    nrow, ncol = np.shape(red)
    rgb_arr = np.zeros((nrow, ncol, 3), dtype=np.uint8)

    # transparency_val will be the same from all calls to this function;
    # only need to capture it once.
    rgb_arr[:, :, 0], transparency_val = prep_arr_for_png_with_transparency(red)
    rgb_arr[:, :, 1] = prep_arr_for_png_with_transparency(green)[0]
    rgb_arr[:, :, 2] = prep_arr_for_png_with_transparency(blue)[0]

    # make a tuple with length 3, where each entry denotes the transparent
    # value for R, G, and B channels (respectively)
    transparency_val = (transparency_val,) * 3

    im = Image.fromarray(rgb_arr, mode="RGB")

    im.save(filepath, transparency=transparency_val)  # default = 72 dpi


def prep_arr_for_png_with_transparency(img_arr):
    """
    Prepare a 2D image array for use in a uint8 PNG with palette-based
    transparency.

    Normalizes and then scales the array values to 1-255. Non-finite pixels
    (aka invalid pixels) are set to 0.

    Parameters
    ----------
    img_arr : array_like
        2D Image to plot

    Returns
    -------
    out : numpy.ndarray with dtype numpy.uint8
        Copy of the input image array that has been prepared for use in
        a PNG file.
        Input image array values were normalized to [0,1] and then
        scaled to [1,255]. Non-finite pixels are set to 0.
    transparency_value : int
        The pixel value denoting non-finite (invalid) pixels. This is currently always 0.

    Notes
    -----
    For PNGs with palette-based transparency, one value in 0-255 will need
    to be assigned to be the fill value (i.e. the value that will appear
    as transparent). For unsigned integer data, it's conventional to use
    the largest representable value. (For signed integer data you usually
    want the most negative value.)
    However, when using RGB mode + palette-based transparency in Python's
    PIL library, if a pixel in only e.g. one color channel is invalid,
    but the corresponding pixel in other channels is valid, then the
    resulting PNG image will make the color for the first channel appear
    dominant. For example, for a given pixel in an RGB image. If a red
    channel's value for that pixel is 255 (invalid), while the green and
    blue channels' values are 123 and 67 (valid), then in the output RGB
    that pixel will appear bright red -- even if the `transparency` parameter
    is assigned correctly. So, instead we'll use 0 to represent invalid
    pixels, so that the resulting PNG "looks" more representative of the
    underlying data.
    """

    # Normalize to range [0,1]. If the array is already normalized,
    # this should have no impact.
    out = nisarqa.normalize(img_arr)

    # After normalization to range [0,1], scale to 1-255 for unsigned int8
    # Reserve the value 0 for use as the transparency value.
    #   out = (<normalized array> * (target_max - target_min)) + target_min
    out = (np.uint8(out * (255 - 1))) + 1

    # Set transparency value so that the "alpha" is added to the image
    transparency_value = 0

    # Denote invalid pixels with 255, so that they output as transparent
    out[~np.isfinite(img_arr)] = transparency_value

    return out, transparency_value


# TODO - move to generic plotting.py
def img2pdf_grayscale(
    img_arr: ArrayLike,
    plots_pdf: PdfPages,
    title: Optional[str] = None,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    colorbar_formatter: Optional[FuncFormatter] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Plot the image array in grayscale, add a colorbar, and append to the PDF.

    Parameters
    ----------
    img_arr : array_like
        Image to plot in grayscale
    plots_pdf : PdfPages
        The output PDF file to append the backscatter image plot to
    title : str or None, optional
        The full title for the plot
    xlim, ylim : sequence of numeric or None, optional
        Lower and upper limits for the axes ticks for the plot.
        Format: xlim=[<x-axis lower limit>, <x-axis upper limit>],
                ylim=[<y-axis lower limit>, <y-axis upper limit>]
    colorbar_formatter : matplotlib.ticker.FuncFormatter or None, optional
        Tick formatter function to define how the numeric value
        associated with each tick on the colorbar axis is formatted
        as a string. `FuncFormatter`s take exactly two arguments:
        `x` for the tick value and `pos` for the tick position,
        and must return a `str`. The `pos` argument is used
        internally by Matplotlib.
        If None, then default tick values will be used. Defaults to None.
        See: https://matplotlib.org/2.0.2/examples/pylab_examples/custom_ticker1.html
        (Wrapping the function with FuncFormatter is optional.)
    xlabel, ylabel : str or None, optional
        Axes labels for the x-axis and y-axis (respectively)
    """

    # Instantiate the figure object
    # (Need to instantiate it outside of the plotting function
    # in order to later modify the plot for saving purposes.)
    f = plt.figure(figsize=nisarqa.FIG_SIZE_ONE_PLOT_PER_PAGE)
    ax = plt.gca()

    # grayscale
    cmap = plt.cm.gray

    # Plot the img_arr image.
    ax_img = ax.imshow(X=img_arr, cmap=cmap)

    # Add Colorbar
    cbar = plt.colorbar(ax_img, ax=ax)

    if colorbar_formatter is not None:
        cbar.ax.yaxis.set_major_formatter(colorbar_formatter)

    ## Label the plot
    format_axes_ticks_and_labels(
        ax=ax,
        xlim=xlim,
        ylim=ylim,
        img_arr_shape=np.shape(img_arr),
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )

    # Make sure axes labels do not get cut off
    f.tight_layout()

    # Append figure to the output PDF
    plots_pdf.savefig(f)

    # Close the plot
    plt.close(f)


def format_axes_ticks_and_labels(
    ax: Axes,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    img_arr_shape: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    """
    Format the tick marks and labels on the given axes object.

    Parameters
    ----------
    ax : matplotlib.axis.Axes
        The axes object to be modified. This axes' tick marks, labels, and
        title will be formatted as specified by the parameters provided to
        this function. `ax` should be the full size of `fig`.
    xlim, ylim : sequence of numeric or None, optional
        Lower and upper limits for the axes ticks for the plot.
        Format: xlim=[<x-axis lower limit>, <x-axis upper limit>],
                ylim=[<y-axis lower limit>, <y-axis upper limit>]
        If `xlim` is None, the x-axes ticks and labels will not be modified.
        Similar for `ylim`. (They are handled independently.) Defaults to None.
    img_arr_shape : pair of ints or None, optional
        The shape of the image which will be placed on `ax`. In practise, this
        establishes the aspect ratio for the axes.
        Required if `xlim` or `ylim` are specified; otherwise will be ignored.
    title : str or None, optional
        The full title for the plot. Defaults to None (no title added).
    xlabel, ylabel : str or None, optional
        Axes labels for the x-axis and y-axis (respectively).
        Defaults to None (no labels added).
    """

    # If xlim or ylim are not provided, let Matplotlib auto-assign the ticks.
    # Otherwise, dynamically calculate and set the ticks w/ labels for
    # the x-axis and/or y-axis.
    # (Attempts to set the limits by using the `extent` argument for
    # matplotlib.imshow() caused significantly distorted images.
    # So, compute and set the ticks w/ labels manually.)
    if xlim is not None or ylim is not None:
        if img_arr_shape is None:
            raise ValueError("Must provide `img_arr_shape` input.")

        # Set the density of the ticks on the figure
        ticks_per_inch = 2.5

        # Get the dimensions of the figure object in inches
        fig_w, fig_h = ax.get_figure().get_size_inches()

        # Get the dimensions of the image array in pixels
        W = img_arr_shape[1]
        H = img_arr_shape[0]

        # Update variables to the actual, displayed image size
        # (The actual image will have a different aspect ratio
        # than the Matplotlib figure window's aspect ratio.
        # But, altering the Matplotlib figure window's aspect ratio
        # will lead to inconsistently-sized pages in the output PDF.)
        if H / W >= fig_h / fig_w:
            # image will be limited by its height, so
            # it will not use the full width of the figure
            fig_w = W * (fig_h / H)
        else:
            # image will be limited by its width, so
            # it will not use the full height of the figure
            fig_h = H * (fig_w / W)

    if xlim is not None:
        # Compute num of xticks to use
        num_xticks = int(ticks_per_inch * fig_w)

        # Always have a minimum of 2 labeled ticks
        num_xticks = max(num_xticks, 2)

        # Specify where we want the ticks, in pixel locations.
        xticks = np.linspace(0, img_arr_shape[1], num_xticks)
        ax.set_xticks(xticks)

        # Specify what those pixel locations correspond to in data coordinates.
        # By default, np.linspace is inclusive of the endpoint
        xticklabels = [
            "{:.1f}".format(i)
            for i in np.linspace(start=xlim[0], stop=xlim[1], num=num_xticks)
        ]
        ax.set_xticklabels(xticklabels, rotation=45)

    if ylim is not None:
        # Compute num of yticks to use
        num_yticks = int(ticks_per_inch * fig_h)

        # Always have a minimum of 2 labeled ticks
        num_yticks = max(num_yticks, 2)

        # Specify where we want the ticks, in pixel locations.
        yticks = np.linspace(0, img_arr_shape[0], num_yticks)
        ax.set_yticks(yticks)

        # Specify what those pixel locations correspond to in data coordinates.
        # By default, np.linspace is inclusive of the endpoint
        yticklabels = [
            "{:.1f}".format(i)
            for i in np.linspace(start=ylim[0], stop=ylim[1], num=num_yticks)
        ]
        ax.set_yticklabels(yticklabels)

    # Label the Axes
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # Add title
    if title is not None:
        ax.set_title(title)


def process_backscatter_and_phase_histograms(
    product: nisarqa.NonInsarProduct,
    params: nisarqa.HistogramParamGroup,
    stats_h5: h5py.File,
    report_pdf: PdfPages,
    plot_title_prefix: str = "Backscatter Coefficient",
    input_raster_represents_power: bool = False,
):
    """
    Generate the Backscatter and Phase Histograms and save their plots
    to the graphical summary PDF file.

    Backscatter histogram will be computed in decibel units.
    Phase histogram defaults to being computed in radians,
    configurable to be computed in degrees by setting
    `params.phs_in_radians` to False.
    NaN values will be excluded from Histograms.

    Parameters
    ----------
    product : nisarqa.NonInsarProduct
        Input NISAR product
    params : HistogramParams
        A structure containing the parameters for processing
        and outputting the backscatter and phase histograms.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output PDF file to append the backscatter image plot to
    plot_title_prefix : str, optional
        Prefix for the title of the backscatter plots.
        Suggestions: "RSLC Backscatter Coefficient (beta-0)" or
        "GCOV Backscatter Coefficient (gamma-0)".
        Defaults to "Backscatter Coefficient".
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either power or root power.
        If `True`, then QA SAS assumes the input data already represents
        power and uses the pixels' magnitudes for computations.
        If `False`, then QA SAS assumes the input data represents root power
        aka magnitude and will handle the full computation to power using
        the formula:  power = abs(<magnitude>)^2 .
        Defaults to False (root power).
    """

    # Generate and store the histograms

    for freq in product.freqs:
        generate_backscatter_image_histogram_single_freq(
            product=product,
            freq=freq,
            params=params,
            stats_h5=stats_h5,
            report_pdf=report_pdf,
            input_raster_represents_power=input_raster_represents_power,
            plot_title_prefix=plot_title_prefix,
        )

        generate_phase_histogram_single_freq(
            product=product,
            freq=freq,
            params=params,
            stats_h5=stats_h5,
            report_pdf=report_pdf,
        )


def generate_backscatter_image_histogram_single_freq(
    product: nisarqa.NonInsarProduct,
    freq: str,
    params: nisarqa.HistogramParamGroup,
    stats_h5: h5py.File,
    report_pdf: PdfPages,
    plot_title_prefix: str = "Backscatter Coefficient",
    input_raster_represents_power: bool = False,
) -> None:
    """
    Generate Backscatter Image Histogram for a single frequency.

    The histogram's plot will be appended to the graphical
    summary file `report_pdf`, and its data will be
    stored in the statistics .h5 file `stats_h5`.
    Backscatter histogram will be computed in decibel units.

    Parameters
    ----------
    product : nisarqa.NonInsarProduct
        Input NISAR product
    freq : str
        Frequency name for the histograms to be processed,
        e.g. 'A' or 'B'
    params : HistogramParamGroup
        A structure containing the parameters for processing
        and outputting the histograms.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output PDF file to append the backscatter image plot to
    plot_title_prefix : str
        Prefix for the title of the backscatter plots.
        Suggestions: "RSLC Backscatter Coefficient (beta-0)" or
        "GCOV Backscatter Coefficient (gamma-0)"
        Defaults to "Backscatter Coefficient"
    input_raster_represents_power : bool, optional
        The input dataset rasters associated with these histogram parameters
        should have their pixel values represent either power or root power.
        If `True`, then QA SAS assumes the input data already represents
        power and uses the pixels' magnitudes for computations.
        If `False`, then QA SAS assumes the input data represents root power
        aka magnitude and will handle the full computation to power using
        the formula:  power = abs(<magnitude>)^2 .
        Defaults to False (root power).
    """

    print(f"Generating Backscatter Image Histograms for Frequency {freq}...")

    # Open one figure+axes.
    # Each band+frequency will have a distinct plot, with all of the
    # polarization images for that setup plotted together on the same plot.
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=nisarqa.FIG_SIZE_ONE_PLOT_PER_PAGE
    )

    # Use custom cycler for accessibility
    ax.set_prop_cycle(nisarqa.CUSTOM_CYCLER)

    def img_prep(arr):
        # Convert to backscatter.
        # For Backscatter Histogram, do not mask out zeros.
        power = (
            np.abs(arr)
            if input_raster_represents_power
            else nisarqa.arr2pow(arr)
        )

        return nisarqa.pow2db(power)

    for pol_name in product.get_pols(freq=freq):
        with product.get_raster(freq=freq, pol=pol_name) as pol_data:
            # Get histogram probability density
            hist_density = nisarqa.compute_histogram_by_tiling(
                arr=pol_data.data,
                bin_edges=params.backscatter_bin_edges,
                data_prep_func=img_prep,
                density=True,
                decimation_ratio=params.decimation_ratio,
                tile_shape=params.tile_shape,
            )

            # Save to stats.h5 file
            grp_path = f"{nisarqa.STATS_H5_QA_FREQ_GROUP}/{pol_name}/" % (
                product.band,
                freq,
            )

            # Save Backscatter Histogram Counts to HDF5 file
            backscatter_units = params.get_units_from_hdf5_metadata(
                "backscatter_bin_edges"
            )

            nisarqa.create_dataset_in_h5group(
                h5_file=stats_h5,
                grp_path=grp_path,
                ds_name="backscatterHistogramDensity",
                ds_data=hist_density,
                ds_units=f"1/{backscatter_units}",
                ds_description=(
                    "Normalized density of the backscatter image histogram"
                ),
            )

            # Add backscatter histogram density to the figure
            add_hist_to_axis(
                ax,
                counts=hist_density,
                edges=params.backscatter_bin_edges,
                label=pol_name,
            )

    # Label the Backscatter Image Figure
    title = (
        f"{plot_title_prefix} Histograms\n{product.band}-band Frequency {freq}"
    )
    ax.set_title(title)

    ax.legend(loc="upper right")
    ax.set_xlabel(f"Backscatter ({backscatter_units})")
    ax.set_ylabel(f"Density (1/{backscatter_units})")

    # Per ADT, let the top limit float for Backscatter Histogram
    ax.set_ylim(bottom=0.0)
    ax.grid()

    # Save complete plots to graphical summary PDF file
    report_pdf.savefig(fig)

    # Close figure
    plt.close(fig)

    print(f"Backscatter Image Histograms for Frequency {freq} complete.")


def generate_phase_histogram_single_freq(
    product: nisarqa.NonInsarProduct,
    freq: str,
    params: nisarqa.HistogramParamGroup,
    stats_h5: h5py.File,
    report_pdf: PdfPages,
) -> None:
    """
    Generate Phase Histograms for a single frequency.

    The histograms' plots will be appended to the graphical
    summary file `report_pdf`, and their data will be
    stored in the statistics .h5 file `stats_h5`.
    Phase histogram defaults to being computed in radians,
    configurable to be computed in degrees per `params.phs_in_radians`.
    NOTE: Only if the dtype of a polarization raster is complex-valued
    (e.g. complex32) will it be included in the Phase histogram(s).
    NaN values will be excluded from the histograms.

    Parameters
    ----------
    product : nisarqa.NonInsarProduct
        The input NISAR product
    freq : str
        Frequency name for the histograms to be processed,
        e.g. 'A' or 'B'
    params : HistogramParamGroup
        A structure containing the parameters for processing
        and outputting the backscatter and phase histograms.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to
    report_pdf : PdfPages
        The output PDF file to append the backscatter image plot to
    """

    band = product.band

    # flag for if any phase histogram densities are generated
    # (We expect this flag to be set to True if any polarization contains
    # phase information. But for example, if a GCOV product only has
    # on-diagonal terms which are real-valued and lack phase information,
    # this will remain False.)
    save_phase_histogram = False

    print(f"Generating Phase Histograms for Frequency {freq}...")

    # Open one figure+axes.
    # Each band+frequency will have a distinct plot, with all of the
    # polarization images for that setup plotted together on the same plot.
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=nisarqa.FIG_SIZE_ONE_PLOT_PER_PAGE
    )

    # Use custom cycler for accessibility
    ax.set_prop_cycle(nisarqa.CUSTOM_CYCLER)

    def img_prep(arr):
        # Remove zero values (and nans) in case of 0 magnitude vectors, etc.
        # Note: There will be no need to clip phase values; the output of
        # np.angle() is always in the range (-pi, pi] (or (-180, 180]).
        if params.phs_in_radians:
            return np.angle(arr[np.abs(arr) >= 1.0e-05], deg=False)
        else:
            # phase in degrees
            return np.angle(arr[np.abs(arr) >= 1.0e-05], deg=True)

    for pol_name in product.get_pols(freq=freq):
        with product.get_raster(freq=freq, pol=pol_name) as pol_data:
            # Only create phase histograms for complex datasets. Examples of
            # complex datasets include RSLC, GSLC, and GCOV off-diagonal rasters.
            # Note: Need to use `np.issubdtype` instead of `np.iscomplexobj`
            # due to e.g. RSLC and GSLC datasets of type ComplexFloat16Decoder.
            if not np.issubdtype(pol_data.data, np.complexfloating):
                continue

            save_phase_histogram = True

            # Get histogram probability densities
            hist_density = nisarqa.compute_histogram_by_tiling(
                arr=pol_data.data,
                bin_edges=params.phs_bin_edges,
                data_prep_func=img_prep,
                density=True,
                decimation_ratio=params.decimation_ratio,
                tile_shape=params.tile_shape,
            )

            # Save to stats.h5 file
            freq_path = nisarqa.STATS_H5_QA_FREQ_GROUP % (band, freq)
            grp_path = f"{freq_path}/{pol_name}/"

            phs_units = params.get_units_from_hdf5_metadata("phs_bin_edges")

            nisarqa.create_dataset_in_h5group(
                h5_file=stats_h5,
                grp_path=grp_path,
                ds_name="phaseHistogramDensity",
                ds_data=hist_density,
                ds_units=f"1/{phs_units}",
                ds_description="Normalized density of the phase histogram",
            )

            # Add phase histogram density to the figure
            add_hist_to_axis(
                ax,
                counts=hist_density,
                edges=params.phs_bin_edges,
                label=pol_name,
            )

    # Label and output the Phase Histogram Figure
    if save_phase_histogram:
        ax.set_title(f"{band} Frequency {freq} Phase Histograms")
        ax.legend(loc="upper right")
        ax.set_xlabel(f"Phase ({phs_units})")
        ax.set_ylabel(f"Density (1/{phs_units})")
        if params.phs_in_radians:
            ax.set_ylim(bottom=0.0, top=0.5)
        else:
            ax.set_ylim(bottom=0.0, top=0.01)
        ax.grid()

        # Save complete plots to graphical summary PDF file
        report_pdf.savefig(fig)

        # Close figure
        plt.close(fig)
    else:
        # Remove unused dataset from STATS.h5 because no phase histogram was
        # generated.

        # Get param attribute for the extraneous group
        metadata = nisarqa.HistogramParamGroup.get_attribute_metadata(
            "phs_bin_edges"
        )

        # Get the instance of the HDF5Attrs object for this parameter
        hdf5_attrs_instance = metadata["hdf5_attrs_func"](params)

        # Form the path in output STATS.h5 file to the group to be deleted
        path = hdf5_attrs_instance.group_path % band
        path += f"/{hdf5_attrs_instance.name}"

        # Delete the unnecessary dataset
        if path in stats_h5:
            del stats_h5[path]

    print(f"Histograms for Frequency {freq} complete.")


def add_hist_to_axis(axis, counts, edges, label):
    """Add the plot of the given counts and edges to the
    axis object. Points will be centered in each bin,
    and the plot will be denoted `label` in the legend.
    """
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    axis.plot(bin_centers, counts, label=label)


def process_range_spectra(
    product: nisarqa.RSLC,
    params: nisarqa.RangeSpectraParamGroup,
    stats_h5: h5py.File,
    report_pdf: PdfPages,
) -> None:
    """
    Generate the RSLC Range Spectra plot(s) and save to PDF and stats.h5.

    Generate the RSLC Range Spectra; save the plot
    to the graphical summary .pdf file and the data to the
    statistics .h5 file.

    Power Spectral Density (PSD) is computed in decibels referenced to 1/hertz (dB re 1/Hz) units,
    and Frequency in Hz or MHz (specified per `params.hz_to_mhz`).

    Parameters
    ----------
    product : nisarqa.RSLC
        Input RSLC product.
    params : nisarqa.RangeSpectraParamGroup
        A structure containing the parameters for processing
        and outputting the range spectra.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to.
    report_pdf : PdfPages
        The output PDF file to append the range spectra plots plot to.
    """

    # Generate and store the range spectra plots
    for freq in product.freqs:
        generate_range_spectra_single_freq(
            product, freq, params, stats_h5, report_pdf
        )


def generate_range_spectra_single_freq(
    product: nisarqa.RSLC,
    freq: str,
    params: nisarqa.RangeSpectraParamGroup,
    stats_h5: h5py.File,
    report_pdf: PdfPages,
) -> None:
    """
    Generate the RSLC Range Spectra for a single frequency.

    Generate the RSLC Range Spectra; save the plot
    to the graphical summary .pdf file and the data to the
    statistics .h5 file.

    Power Spectral Density (PSD) is computed in decibels referenced to 1/hertz (dB re 1/Hz) units,
    and Frequency in Hz or MHz (specified per `params.hz_to_mhz`).

    Parameters
    ----------
    product : nisarqa.RSLC
        Input RSLC product.
    freq : str
        Frequency name for the range power spectra to be processed,
        e.g. 'A' or 'B'
    params : RangeSpectraParamGroup
        A structure containing the parameters for processing
        and outputting the spectra.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to.
    report_pdf : PdfPages
        The output PDF file to append the range spectra plots plot to.
    """

    print(f"Generating Range Spectra for Frequency {freq}...")

    # Plot the range spectra using strictly increasing sample frequencies
    # (no discontinuity).
    fft_shift = True

    # Get the FFT spacing
    # Because `freq` is fixed, and all polarizations within
    # the same frequency will have the same `fft_freqs`.
    # So, we only need to do this computation one time.
    first_pol = product.get_pols(freq=freq)[0]
    with product.get_raster(freq, first_pol) as img:
        # Compute the sample rate
        # c/2 for radar energy round-trip; units for `sample_rate` will be Hz
        dr = product.get_slant_range_spacing(freq)
        sample_rate = (constants.c / 2.0) / dr

        fft_freqs = nisarqa.generate_fft_freqs(
            num_samples=img.data.shape[1],
            sampling_rate=sample_rate,
            fft_shift=fft_shift,
        )

        proc_center_freq = product.get_processed_center_frequency(freq)

        if params.hz_to_mhz:
            fft_freqs = nisarqa.hz2mhz(fft_freqs)
            proc_center_freq = nisarqa.hz2mhz(proc_center_freq)
            freq_units = "MHz"
        else:
            freq_units = "Hz"

    # Save x-axis values to stats.h5 file
    nisarqa.create_dataset_in_h5group(
        h5_file=stats_h5,
        grp_path=nisarqa.STATS_H5_QA_FREQ_GROUP % (product.band, freq),
        ds_name="rangeSpectraFrequencies",
        ds_data=fft_freqs,
        ds_units=freq_units,
        ds_description=(
            f"Frequency coordinates for Frequency {freq} range power spectra."
        ),
    )

    # Plot the Range Power Spectra for each pol onto the same axes.
    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=nisarqa.FIG_SIZE_ONE_PLOT_PER_PAGE
    )

    # Use custom cycler for accessibility
    ax.set_prop_cycle(nisarqa.CUSTOM_CYCLER)

    rng_spec_units = "dB re 1/Hz"

    for pol in product.get_pols(freq):
        with product.get_raster(freq=freq, pol=pol) as img:
            # Get the Range Spectra
            # (The returned array is in dB re 1/Hz)
            rng_spectrum = nisarqa.compute_range_spectra_by_tiling(
                arr=img.data,
                sampling_rate=sample_rate,
                az_decimation=params.az_decimation,
                tile_height=params.tile_height,
                fft_shift=fft_shift,
            )

            # Save normalized range power spectra values to stats.h5 file
            nisarqa.create_dataset_in_h5group(
                h5_file=stats_h5,
                grp_path=nisarqa.STATS_H5_QA_POL_GROUP
                % (product.band, freq, pol),
                ds_name="rangePowerSpectralDensity",
                ds_data=rng_spectrum,
                ds_units=rng_spec_units,
                ds_description=(
                    "Normalized range power spectral density for Frequency"
                    f" {freq}, Polarization {pol}."
                ),
            )

            # Add this power spectrum to the figure
            ax.plot(fft_freqs, rng_spectrum, label=pol)

    # Label the Plot
    ax.set_title(f"Range Power Spectra for Frequency {freq}\n")
    ax.set_xlabel(f"Frequency rel. {proc_center_freq} {freq_units}")

    ax.set_ylabel(f"Power Spectral Density ({rng_spec_units})")

    ax.legend(loc="upper right")
    ax.grid()

    # Save complete plots to graphical summary pdf file
    report_pdf.savefig(fig)

    # Close the plot
    plt.close()

    print(f"Range Power Spectra for Frequency {freq} complete.")


__all__ = nisarqa.get_all(__name__, objects_to_skip)
