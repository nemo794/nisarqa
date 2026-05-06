from __future__ import annotations

import os
from typing import Any, overload

import isce3
import numpy as np

import nisarqa

from ..plotting_utils import plot_2d_array_and_save_to_png

objects_to_skip = nisarqa.get_all(name=__name__)


@overload
def _make_phase_browse(
    phase: np.ndarray,
    grid: nisarqa.RadarGrid,
    *,
    cbar_min_max: tuple[float, float],
    sample_spacing: tuple[float, float],
    longest_side_max: int | None,
    resample: str,
    browse_paths: nisarqa.BrowseOutputPaths,
    save_latlon_browse: bool,
    orbit: isce3.core.Orbit,
    wavelength: float,
    look_side: isce3.core.LookSide | str,
    dem_file: str | os.PathLike | None,
) -> None: ...


@overload
def _make_phase_browse(
    phase: np.ndarray,
    grid: nisarqa.GeoGrid,
    *,
    cbar_min_max: tuple[float, float],
    sample_spacing: tuple[float, float],
    longest_side_max: int | None,
    resample: str,
    browse_paths: nisarqa.BrowseOutputPaths,
    save_latlon_browse: bool,
    fill_value: float,
) -> None: ...


def _make_phase_browse(
    phase: np.ndarray,
    grid: nisarqa.GeoGrid | nisarqa.RadarGrid,
    *,
    cbar_min_max: tuple[float, float],
    sample_spacing: tuple[float, float],
    longest_side_max: int | None,
    resample: str,
    browse_paths: nisarqa.BrowseOutputPaths,
    save_latlon_browse: bool,
    # Level-1 (RadarGrid) parameters:
    orbit: Any | None = None,
    wavelength: float | None = None,
    look_side: Any | None = None,
    dem_file: str | os.PathLike | None = None,
    # Level-2 (GeoGrid) parameters:
    fill_value: float | None = None,
) -> None:
    """
    Helper function to create and save phase browse products as PNG+KML.

    This function handles both Level-1 (radar coordinates) and Level-2 (geocoded)
    phase arrays. It decimates the input phase array to square pixels, saves
    the primary browse PNG+KML, and optionally creates an EPSG 4326 lat/lon
    version of the browse products.

    Parameters
    ----------
    phase : np.ndarray
        Input phase array. This will be decimated per `sample_spacing`
        and `longest_side_max` prior to saving into browse products.
    grid : nisarqa.GeoGrid or nisarqa.RadarGrid
        Coordinate grid corresponding to input phase array.
        This will be decimated per `sample_spacing` and `longest_side_max`
        prior to use for generating the browse products.
    cbar_min_max : tuple[float, float]
        The (min, max) values for the colorbar range.
        Format: (vmin, vmax)
    sample_spacing : tuple[float, float]
        The Y direction ground sample spacing and X direction ground
        sample spacing of the phase array, in the same units (e.g. meters).
        For radar-domain products, Y direction corresponds to azimuth,
        and X direction corresponds to range.
        Only the magnitude (absolute value) of the sample spacing is used.
        Format: (dy, dx)
    longest_side_max : int or None
        Maximum number of pixels for the longest side of the decimated array.
        If None, no maximum is enforced.
    resample : str
        Resampling method for EPSG 4326 (lat/lon) browse generation.
        Common options: 'bilinear', 'cubic', 'nearest', etc.
    browse_paths : nisarqa.BrowseOutputPaths
        Container with output directory and browse/KML filenames.
    save_latlon_browse : bool
        If False, only save the primary browse PNG+KML in the input phase
        array's native grid.
        If True, additionally save a version of the browse PNG+KML in
        EPSG 4326 (lat/lon).
    orbit : isce3.core.Orbit or None, optional
        **Required for Level-1 (RadarGrid). Ignored for Level-2.**
        The trajectory of the radar antenna phase center.
        Defaults to None.
    wavelength : float or None, optional
        **Required for Level-1 (RadarGrid). Ignored for Level-2.**
        The radar central wavelength, in meters, corresponding to the
        input phase array.
        Defaults to None.
    look_side : isce3.core.LookSide or {'left', 'right'} or None, optional
        **Required for Level-1 (RadarGrid). Ignored for Level-2.**
        The look direction of the radar (left-looking or right-looking).
        Defaults to None.
    dem_file : str, os.PathLike, or None, optional
        **Required for Level-1 (RadarGrid). Ignored for Level-2.**
        Digital Elevation Model (DEM) file path in a GDAL-compatible raster
        format. Used when geocoding the EPSG 4326 (lat/lon) browse.
        If None, a zero-height DEM will be used.
        Defaults to None.
    fill_value : float or None, optional
        **Required for Level-2 (GeoGrid). Ignored for Level-1.**
        The fill value for the input phase array.
        Defaults to None.
    """
    log = nisarqa.get_logger()

    # Decimate the phase array to square pixels
    decimated_phase, ky, kx = (
        nisarqa.decimate_array_to_square_pixels_with_strides(
            arr=phase,
            y_axis_spacing=sample_spacing[0],
            x_axis_spacing=sample_spacing[1],
            longest_side_max=longest_side_max,
        )
    )

    # Save primary browse PNG
    plot_2d_array_and_save_to_png(
        arr=decimated_phase,
        cmap="twilight_shifted",
        png_filepath=browse_paths.primary_browse_path,
        vmin=cbar_min_max[0],
        vmax=cbar_min_max[1],
    )

    # Downsample the grid to match the decimated phase array
    browse_grid = grid.downsample(y_stride=ky, x_stride=kx, mode="decimate")

    # `decimated_phase` and `browse_grid` were downsampled independently.
    # Assert they have the same length/width.
    assert (
        np.shape(decimated_phase)[0] == np.shape(browse_grid.y_pixel_centers)[0]
    ), (
        f"Must be equal: {np.shape(decimated_phase)[0]=} but"
        f" {np.shape(browse_grid.y_pixel_centers)[0]=}."
    )
    assert np.shape(decimated_phase)[1] == len(browse_grid.x_pixel_centers), (
        f"Must be equal: {np.shape(decimated_phase)[1]=} but"
        f" {len(browse_grid.x_pixel_centers)=}."
    )

    # Save KML for the primary browse PNG
    if isinstance(grid, nisarqa.GeoGrid):
        # Level-2: Geocoded product
        browse_grid.save_kml(browse_paths=browse_paths)
    else:
        # Level-1: Radar product
        assert isinstance(grid, nisarqa.RadarGrid)
        browse_grid.save_kml(
            browse_paths=browse_paths,
            orbit=orbit,
            wavelength=wavelength,
            look_side=look_side,
            dem_file=dem_file,
        )

    # Optionally create EPSG 4326 browse PNG and KML
    if save_latlon_browse:
        if isinstance(grid, nisarqa.GeoGrid):
            # Level-2: Reproject using GDAL
            geocoded_arr, qa_geogrid_4326 = nisarqa.reproject_geo_raster(
                image_array=decimated_phase,
                fill_value=fill_value,
                geogrid=browse_grid,
                output_epsg=4326,
                resample=resample,
            )
        else:
            # Level-1: Geocode using ISCE3
            geocoded_arr, qa_geogrid_4326 = nisarqa.geocode_radar_raster(
                radar_array=decimated_phase,
                radargrid=browse_grid,
                orbit=orbit,
                wavelength=wavelength,
                look_side=look_side,
                epsg=4326,
                dem_file=dem_file,
                resample=resample,
            )

        # Save EPSG 4326 browse PNG
        suffix = nisarqa.LATLON_SUFFIX
        png_4326_path = browse_paths.get_browse_path(suffix=suffix)

        plot_2d_array_and_save_to_png(
            arr=geocoded_arr,
            cmap="twilight_shifted",
            png_filepath=png_4326_path,
            vmin=cbar_min_max[0],
            vmax=cbar_min_max[1],
        )
        log.info(f"EPSG 4326 (lat/lon) browse PNG saved to {png_4326_path}")

        # Save EPSG 4326 KML
        qa_geogrid_4326.save_kml(browse_paths=browse_paths, suffix=suffix)

        kml_4326_path = browse_paths.get_kml_path(suffix=suffix)
        log.info(f"EPSG 4326 (lat/lon) browse KML saved to {kml_4326_path}")


__all__ = nisarqa.get_all(__name__, objects_to_skip)
