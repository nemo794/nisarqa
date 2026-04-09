from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union, overload

import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import nisarqa

from ..plotting_utils import (
    downsample_img_to_size_of_axes,
    format_axes_ticks_and_labels,
    format_cbar_ticks_for_multiples_of_pi,
    plot_2d_array_and_save_to_png,
)
from ..processing_utils import get_phase_array
from .histograms import process_two_histograms

objects_to_skip = nisarqa.get_all(name=__name__)


def process_phase_image_wrapped(
    product: nisarqa.WrappedGroup,
    params_wrapped_igram: nisarqa.ThresholdParamGroup,
    params_coh_mag: nisarqa.ThresholdParamGroup,
    report_pdf: PdfPages,
    stats_h5: h5py.File,
) -> None:
    """
    Process the wrapped groups' phase and coherence magnitude layers.

    Appends plots of each layer to the report PDF file.

    Parameters
    ----------
    product : nisarqa.WrappedGroup
        Input NISAR product.
    params_wrapped_igram : nisarqa.ThresholdParamGroup
        A structure containing the parameters for checking the percentage
        of invalid pixels in the wrapped interferogram layer.
    params_coh_mag : nisarqa.ThresholdParamGroup
        A structure containing the parameters for checking the percentage
        of invalid pixels in the coherence magnitude layer.
    report_pdf : matplotlib.backends.backend_pdf.PdfPages
        Output PDF file to append the phase and coherence magnitude plots to.
    stats_h5 : h5py.File
        The output file to save QA metrics, etc. to.
    """
    for freq in product.freqs:
        for pol in product.get_pols(freq=freq):
            with (
                product.get_wrapped_igram(freq=freq, pol=pol) as complex_img,
                product.get_wrapped_coh_mag(freq=freq, pol=pol) as coh_img,
            ):
                # Compute Statistics first, in case of malformed layers
                # (which could cause plotting to fail)
                nisarqa.compute_and_save_basic_statistics(
                    raster=complex_img,
                    params=params_wrapped_igram,
                    stats_h5=stats_h5,
                )
                nisarqa.compute_and_save_basic_statistics(
                    raster=coh_img,
                    params=params_coh_mag,
                    stats_h5=stats_h5,
                )

                plot_wrapped_phase_image_and_coh_mag_to_pdf(
                    complex_raster=complex_img,
                    coh_raster=coh_img,
                    report_pdf=report_pdf,
                )

                # Plot Histograms
                process_two_histograms(
                    raster1=complex_img,
                    raster2=coh_img,
                    r1_xlabel="InSAR Phase",
                    r2_xlabel="Coherence Magnitude",
                    name_of_histogram_pair="Wrapped Phase Image Group",
                    report_pdf=report_pdf,
                    stats_h5=stats_h5,
                    sharey=False,
                    r1_data_prep_func=np.angle,
                )


@overload
def plot_wrapped_phase_image_and_coh_mag_to_pdf(
    complex_raster: nisarqa.RadarRaster,
    coh_raster: nisarqa.RadarRaster,
    report_pdf: PdfPages,
) -> None: ...


@overload
def plot_wrapped_phase_image_and_coh_mag_to_pdf(
    complex_raster: nisarqa.GeoRaster,
    coh_raster: nisarqa.GeoRaster,
    report_pdf: PdfPages,
) -> None: ...


def plot_wrapped_phase_image_and_coh_mag_to_pdf(
    complex_raster,
    coh_raster,
    report_pdf,
):
    """
    Plot a complex raster and coherence magnitude layers side-by-side on PDF.

    Parameters
    ----------
    complex_raster : nisarqa.RadarRaster or nisarqa.GeoRaster
        *Raster of complex interferogram data. This should correspond to
        `coh_raster`.
    coh_raster : nisarqa.GeoRaster or nisarqa.RadarRaster
        *Raster for the coherence magnitude raster. This should correspond to
        `complex_raster`.
    report_pdf : matplotlib.backends.backend_pdf.PdfPages
        Output PDF file to append the phase and coherence magnitude plots to.
    """

    # Validate that the pertinent metadata in the rasters is equal.
    nisarqa.compare_raster_metadata(
        complex_raster, coh_raster, almost_identical=False
    )

    phs_img, cbar_min_max = get_phase_array(
        phs_or_complex_raster=complex_raster,
        make_square_pixels=True,
        rewrap=None,
    )

    coh_img = nisarqa.decimate_raster_array_to_square_pixels(coh_raster)

    fig, (ax1, ax2) = plt.subplots(
        ncols=2,
        nrows=1,
        constrained_layout="tight",
        figsize=nisarqa.FIG_SIZE_TWO_PLOTS_PER_PAGE,
        sharey=True,
    )

    # Decimate to fit nicely on the figure.
    phs_img = downsample_img_to_size_of_axes(
        ax=ax1, arr=phs_img, mode="decimate"
    )
    coh_img = downsample_img_to_size_of_axes(
        ax=ax2, arr=coh_img, mode="decimate"
    )

    # Construct title for the overall PDF page. (`*raster.name` has a format
    # like "RIFG_L_A_interferogram_HH_wrappedInterferogram". We need to
    # remove the final layer name of e.g. "_wrappedInterferogram".)
    name = "_".join(complex_raster.name.split("_")[:-1])
    title = f"Wrapped Phase Image Group\n{name}"
    fig.suptitle(title)

    # Add the wrapped phase image plot
    im1 = ax1.imshow(
        phs_img,
        aspect="equal",
        cmap="twilight_shifted",
        interpolation="none",
        vmin=cbar_min_max[0],
        vmax=cbar_min_max[1],
    )

    format_axes_ticks_and_labels(
        ax=ax1,
        xlim=complex_raster.x_axis_limits,
        ylim=complex_raster.y_axis_limits,
        img_arr_shape=np.shape(phs_img),
        xlabel=complex_raster.x_axis_label,
        ylabel=complex_raster.y_axis_label,
        title=complex_raster.name.split("_")[-1],  # use only the layer's name
    )

    # Add a colorbar to the figure
    cax1 = fig.colorbar(im1)
    cax1.ax.set_ylabel(
        ylabel="InSAR Phase (radians)", rotation=270, labelpad=10.0
    )

    format_cbar_ticks_for_multiples_of_pi(
        cbar_min=cbar_min_max[0], cbar_max=cbar_min_max[1], cax=cax1
    )

    # Add the coh mag layer corresponding to the wrapped phase image plot
    im2 = ax2.imshow(
        coh_img,
        aspect="equal",
        cmap="gray",
        interpolation="none",
        vmin=0.0,
        vmax=1.0,
    )

    # No y-axis label nor ticks. This is the right side plot; y-axis is shared.
    format_axes_ticks_and_labels(
        ax=ax2,
        xlim=coh_raster.x_axis_limits,
        img_arr_shape=np.shape(coh_img),
        xlabel=coh_raster.x_axis_label,
        # Use only the final layer name for `title`. (`coh_raster.name` has
        # format like: "RUNW_L_A_interferogram_HH_unwrappedPhase".)
        title=coh_raster.name.split("_")[-1],
    )

    # Add a colorbar to the figure
    cax2 = fig.colorbar(im2)
    cax2.ax.set_ylabel(
        ylabel="Coherence Magnitude", rotation=270, labelpad=10.0
    )

    # Append figure to the output PDF
    report_pdf.savefig(fig)

    # Close the plot
    plt.close(fig)


def make_wrapped_phase_browse(
    product: nisarqa.WrappedGroup,
    *,
    freq: str,
    pol: str,
    params: Any,  # will be type-narrowed in the function
    out_dir: str | os.Pathlike,
    browse_filename: str,
    kml_filename: str,
    dem_file: str | os.PathLike | None = None,
) -> None:
    """
    Create and save the wrapped interferogram browse products as PNG+KML.

    Parameters
    ----------
    product : nisarqa.WrappedGroup
        Input NISAR product.
    freq, pol : str
        The frequency and polarization (respectively) pair for the wrapped
        interferogram to save as a PNG.
    params : nisarqa.IgramBrowseParamGroup and
            nisarqa.L1RadarBrowse4326ParamGroup or nisarqa.L2GeoBrowse4326ParamGroup
        A structure containing the processing parameters for the browse PNG.
        Must be an instance of IgramBrowseParamGroup and (via multiple
        inheritance) also an instance of either:
            L1RadarBrowse4326ParamGroup or L2GeoBrowse4326ParamGroup
    png_filepath : path-like
        Filename (with path) for the image PNG.
    out_dir : path-like
        The directory to write the output PNG and KML file(s) to. This
        directory must already exist.
    browse_filename : str
        The basename of the output browse image PNG file. The file will be
        created in `out_dir`. Example: "BROWSE.png".
    kml_filename : str
        The basename of the output browse image KML file. The file will be
        created in `out_dir`. Example: "BROWSE.kml".
    dem_file : path-like or None, optional
        Digital Elevation Model (DEM) file path in a GDAL-compatible raster
        format. Will be ignored if `params.output_browse_4326`
        is False or if `product` is a Level-2 Geocoded product.
        Used for Level-1 products when geocoding the EPSG 4326 browse; if None,
        a zero-height DEM will be used.
        Defaults to None.
    longest_side_max : int or None, optional
        The maximum number of pixels allowed for the longest side of the final
        2D multilooked image. If None, the longest edge of `arr` will be used.
        Defaults to None.
    """
    log = nisarqa.get_logger()

    # XXX - Should improve the function's type annotation, but there's not
    # a good syntax for multiple inheritance in combination with a Union.
    # So, use type narrowing:
    if not isinstance(params, nisarqa.IgramBrowseParamGroup):
        raise TypeError(f"{type(params)=}, must be IgramBrowseParamGroup")
    t = nisarqa.L1RadarBrowse4326ParamGroup | nisarqa.L2GeoBrowse4326ParamGroup
    if not isinstance(params, t):
        msg = f"{type(params)=}, must be L1RadarBrowse4326ParamGroup or L2GeoBrowse4326ParamGroup"
        raise TypeError(msg)

    with product.get_wrapped_igram(freq=freq, pol=pol) as igram_r:
        phase, cbar_min_max = get_phase_array(
            phs_or_complex_raster=igram_r,
            make_square_pixels=False,  # we'll do this while downsampling
            rewrap=None,
        )

        ky, kx = plot_2d_array_and_save_to_png(
            arr=phase,
            cmap="twilight_shifted",
            sample_spacing=(igram_r.y_ground_spacing, igram_r.x_ground_spacing),
            longest_side_max=params.longest_side_max,
            png_filepath=Path(out_dir, browse_filename),
            vmin=cbar_min_max[0],
            vmax=cbar_min_max[1],
        )

        browse_grid = igram_r.grid.downsample(
            y_stride=ky, x_stride=kx, mode="decimate"
        )

        llq_kwargs = {
            "output_dir": out_dir,
            "kml_filename": kml_filename,
            "png_filename": browse_filename,
        }
        if not product.is_geocoded:
            llq_kwargs["orbit"] = product.get_orbit(ref_or_sec="reference")
            llq_kwargs["wavelength"] = product.wavelength(freq=freq)
            llq_kwargs["look_side"] = product.look_direction
            llq_kwargs["dem_file"] = dem_file

        browse_grid.save_kml(**llq_kwargs)

        if params.output_browse_4326:
            if product.is_geocoded:
                # Level-2: Reproject using GDAL
                geocoded_arr, qa_geogrid_4326 = nisarqa.reproject_geo_raster(
                    image_array=phase,
                    fill_value=igram_r.fill_value,
                    geogrid=browse_grid,
                    output_epsg=4326,
                    longest_side_max=params.longest_side_max,
                    resample=params.resample,
                )
            else:
                # Level-1: Geocode using ISCE3
                browse_radargrid = browse_grid.get_isce3_radar_grid_parameters(
                    wavelength=product.wavelength(freq=freq),
                    look_side=product.look_direction,
                )

                isce3_geogrid = nisarqa.compute_geogrid(
                    bounding_polygon=product.bounding_polygon,
                    epsg=4326,  # lon/lat
                    longest_side_max=params.longest_side_max,
                    margin_in_km=params.margin_in_km,
                )

                geocoded_arr = nisarqa.geocode_radar_raster(
                    radar_array=phase,
                    radargrid=browse_radargrid,
                    orbit=product.get_orbit(ref_or_sec="reference"),
                    geogrid=isce3_geogrid,
                    dem_file=dem_file,
                    resample=params.resample,
                )

                qa_geogrid_4326 = nisarqa.GeoGrid.from_isce3_geo_grid(
                    isce3_geogrid=isce3_geogrid
                )

            # Save EPSG 4326 browse PNG
            browse_filename_4326 = str(browse_filename).replace(
                ".png", "_4326.png"
            )
            browse_path_4326 = Path(out_dir, browse_filename_4326)
            plot_2d_array_and_save_to_png(
                arr=geocoded_arr,
                cmap="twilight_shifted",
                sample_spacing=None,  # geocoded_arr already on square pixels
                longest_side_max=None,  # geocoded_arr already correct shape
                png_filepath=browse_path_4326,
                vmin=cbar_min_max[0],
                vmax=cbar_min_max[1],
            )

            # Generate EPSG 4326 KML
            kml_filename_4326 = str(kml_filename).replace(".kml", "_4326.kml")
            qa_geogrid_4326.save_kml(
                output_dir=out_dir,
                kml_filename=kml_filename_4326,
                png_filename=browse_filename_4326,
            )

            log.info(f"EPSG 4326 browse PNG saved to {browse_path_4326}")
            log.info(
                f"EPSG 4326 browse KML saved to {Path(out_dir, kml_filename_4326)}"
            )


__all__ = nisarqa.get_all(__name__, objects_to_skip)
