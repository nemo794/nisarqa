from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

import isce3
import numpy as np
from numpy.typing import ArrayLike
from osgeo import gdal, osr

import nisarqa

objects_to_skip = nisarqa.get_all(__name__)


@contextmanager
def dem_file_manager(
    dem_file: str | os.PathLike | None = None,
) -> Generator[pathlib.Path, None, None]:
    """
    Context manager for DEM file handling.

    If a DEM file is provided, yields it without cleanup.
    If None, creates a temporary zero-height DEM (global coverage in EPSG 4326
    with 1 degree resolution) and automatically cleans it up on exit.

    Parameters
    ----------
    dem_file : path-like or None, optional
        Digital Elevation Model (DEM) file path. If None, a temporary
        zero-height DEM will be created (global coverage in EPSG 4326 with
        1 degree resolution) and automatically cleaned up when exiting the
        context manager. Defaults to None.

    Yields
    ------
    pathlib.Path
        Path to the DEM file (either the provided DEM or the temporary
        zero-height DEM with global coverage in EPSG 4326 with 1 degree
        resolution).

    Examples
    --------
    >>> # Use temporary zero-height DEM (auto-cleanup)
    >>> with dem_file_manager(dem_file=None) as dem_path:
    ...     dem = isce3.io.Raster(str(dem_path))
    ...     # ... use dem ...
    ...     # Temporary DEM automatically deleted after this block

    >>> # Use provided DEM (no cleanup)
    >>> with dem_file_manager(dem_file="/path/to/real_dem.tif") as dem_path:
    ...     dem = isce3.io.Raster(str(dem_path))
    ...     # ... use dem ...
    ...     # Provided DEM is NOT deleted
    """
    if dem_file is None:
        # Create temporary zero-height DEM
        temp_dem_path = _create_zero_height_dem()
        try:
            yield temp_dem_path
        finally:
            # Cleanup temporary DEM
            temp_dem_path.unlink(missing_ok=True)
    else:
        # Use provided DEM (no cleanup)
        yield pathlib.Path(dem_file)


def _create_zero_height_dem() -> pathlib.Path:
    """
    Create a zero-height DEM TIF file in the scratch directory.

    Creates a GeoTIFF file containing an array of zeros with global coverage
    in EPSG 4326 (lon/lat) with 1 degree resolution.

    The file is uniquely named with a timestamp to avoid collisions. This is
    useful for geocoding operations where a DEM is required but high accuracy
    is not critical (e.g., for browse image generation).

    Returns
    -------
    pathlib.Path
        Path to the created zero-height DEM file with global coverage
        in EPSG 4326 with 1 degree resolution, located in the nisarqa
        global scratch directory.

    Notes
    -----
    This is a private helper function. The file is NOT automatically cleaned up.
    Use dem_file_manager() context manager for automatic cleanup.
    """
    # Create a uniquely-named file in the scratch directory
    scratch = nisarqa.get_global_scratch_dir()
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

    dem_file = scratch / f"zero_height_dem_{utc_now}.tif"

    # 1. Define resolution
    res = 1.0  # 1 degree resolution

    # Calculate dimensions (360x180 for 1-degree resolution)
    width = int(360 / res)
    height = int(180 / res)

    # 2. Create the dataset
    driver = gdal.GetDriverByName("GTiff")
    # GDT_Int16 is common for DEMs; use GDT_Float32 for precision
    ds = driver.Create(str(dem_file), width, height, 1, gdal.GDT_Int16)

    # 3. Set GeoTransform:
    #   [Upper Left X, X Resolution, Rotation,
    #       Upper Left Y, Rotation, Y Resolution]
    # Note: Y resolution must be negative for North-up images
    ds.SetGeoTransform([-180, res, 0, 90, 0, -res])

    # 4. Set Spatial Reference (WGS 84)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())

    # 5. Fill with zeros
    band = ds.GetRasterBand(1)
    zeros = np.zeros((height, width), dtype=np.int16)
    band.WriteArray(zeros)

    # Finalize and save
    band.FlushCache()
    ds = None

    return dem_file


def geocode_radar_raster(
    radar_array: ArrayLike,
    radargrid: nisarqa.RadarGrid,
    orbit: isce3.core.Orbit,
    wavelength: float,
    look_side: isce3.core.LookSide | str,
    epsg: int,
    dem_file: str | os.PathLike | None = None,
    resample: str = "bilinear",
) -> np.ndarray:
    """
    Geocode a range-Doppler grid image array onto a given geogrid.

    This function geocodes data from range-Doppler coordinates
    to geographic/projected coordinates.

    Parameters
    ----------
    radar_array : array-like
        Input range-Doppler image array (e.g. NISAR Level-1) to be geocoded.
        Must be real-valued (float).
    radargrid : isce3.product.RadarGridParameters
        ISCE3 radargrid parameters specifying the range-Doppler grid associated
        with the input `radar_array`.
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center.
    wavelength : float
        The radar central wavelength, in meters.
    look_side : isce3.core.LookSide or {'left', 'right'}
        The look direction of the radar (left-looking or right-looking).
    epsg : int
        The EPSG code for the output geocoded raster. Supports any valid EPSG
        (e.g., 4326 for lat/lon, 32610 for UTM Zone 10N, 3413 for polar
        stereographic).
    dem_file : path-like or None, optional
        Path to a DEM file; required for accurate geolocation of the pixels.
        If None, a temporary zero-height DEM will be used (global coverage
        in EPSG 4326 with 1 degree resolution).
        Defaults to None.
    resample : str, optional
        Resampling method for ISCE3 geocoding. Options: 'sinc', 'bilinear',
        'bicubic', 'nearest', 'biquintic'. Default: 'bilinear'.

    Returns
    -------
    geocoded_array : numpy.ndarray
        2D array of geocoded data. The length of the longest side of
        `geocoded_array` will be no greater than the length of the
        longest side of the input `radar_array`.
    output_geogrid : nisarqa.GeoGrid
        GeoGrid object describing the coordinate system and grid of the
        geocoded array.

    Warnings
    --------
    This function is not tested for large, full-size NISAR rasters.
    Recommend only using it to geocode relatively small rasters, such as
    browse image arrays. To geocode full-size NISAR rasters, suggest using
    ISCE3 directly.
    """

    if np.iscomplexobj(radar_array):
        raise ValueError(
            f"{type(radar_array)=} which is complex-valued. Only real-valued"
            " data currently supported."
        )

    # Use scratch directory with temporary files for ISCE3 raster I/O.
    # GDAL's in-memory option has potential security issues.
    scratch = nisarqa.get_global_scratch_dir()

    gdal_dtype = gdal.GDT_Float64
    output_dtype = np.float64
    # Create a uniquely-named string for filenaming
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

    # Setup temporary file paths
    input_file = scratch / f"input_{utc_now}.tif"
    output_file = scratch / f"output_{utc_now}.tif"

    try:
        # Use context manager for DEM (handles both provided and temporary DEM)
        with dem_file_manager(dem_file) as dem_filepath:
            dem = isce3.io.Raster(str(dem_filepath))

            # Setup temporary input raster file, so that it can be converted to
            # the correct format for ISCE3 geocoding
            input_ds = gdal.GetDriverByName("GTiff").Create(
                input_file,
                radar_array.data.shape[1],
                radar_array.data.shape[0],
                1,
                gdal_dtype,
            )

            # Ensure float type
            raster_array = radar_array.astype(np.float64)

            # Get lon/lat corners from radar grid
            llq = radargrid.get_latlonquad(
                orbit=orbit,
                wavelength=wavelength,
                look_side=look_side,
                dem_file=dem_filepath,
            )

            # Transform lon/lat corners to target projection
            proj = isce3.core.make_projection(epsg)
            corners_proj = []
            for corner in [llq.ul, llq.ur, llq.ll, llq.lr]:
                lon_rad = np.deg2rad(corner.lon)
                lat_rad = np.deg2rad(corner.lat)
                # Forward transform: lon/lat -> target projection
                # Returns coordinates in projection's native units
                # (degrees for EPSG 4326, meters for UTM, etc.)
                x, y, z = proj.forward([lon_rad, lat_rad, 0])
                corners_proj.append((x, y))

            # Get bounding box in target projection's native units
            xs = [c[0] for c in corners_proj]
            ys = [c[1] for c in corners_proj]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)

            width = maxx - minx
            height = maxy - miny

            # Determine resolution based on the longest side of the array.
            # Handle width and height independently. For example, in lon/lat
            # coordinates, 1 degree of latitude is not equivalent to 1 degree
            # of longitude, particularly towards the polar regions.
            resolution_x = width / max(np.shape(radar_array))
            resolution_y = height / max(np.shape(radar_array))

            # Convert nisarqa.RadarGrid to isce3.product.RadarGridParameters
            isce3_radargrid = radargrid.get_isce3_radar_grid_parameters(
                wavelength=wavelength,
                look_side=look_side,
            )

            input_ds.GetRasterBand(1).WriteArray(raster_array)
            input_ds.FlushCache()
            input_ds = None  # Close
            input_raster = isce3.io.Raster(str(input_file))

            # Set up geocoding object
            geocode_obj = isce3.geocode.GeocodeFloat64()

            geocode_obj.orbit = orbit
            geocode_obj.ellipsoid = isce3.core.WGS84_ELLIPSOID
            geocode_obj.doppler = isce3.core.LUT2d()  # Zero-Doppler for NISAR
            geocode_obj.threshold_geo2rdr = 1.0e-8
            geocode_obj.numiter_geo2rdr = 25
            geocode_obj.data_interpolator = resample

            # Call geogrid() with custom EPSG and pixel spacing, otherwise
            # update_geogrid() will use the DEM's EPSG and pixel spacing.
            # (This is not ideal in case e.g. we use the zero-height DEM.)
            # But, use NaN for start positions and 0 for dimensions to signal
            # to update_geogrid() that it should compute those values.
            geocode_obj.geogrid(
                x_start=np.nan,  # Will be computed by update_geogrid
                y_start=np.nan,  # Will be computed by update_geogrid
                x_spacing=resolution_x,
                y_spacing=-resolution_y,  # Y posting should be negative
                width=0,  # Will be computed by update_geogrid
                length=0,  # Will be computed by update_geogrid
                epsg=epsg,
            )

            # Now call update_geogrid - it will compute bounds, and also
            # handle any antimeridian crossings
            geocode_obj.update_geogrid(isce3_radargrid, dem)

            # Create temporary output raster file
            output_ds = gdal.GetDriverByName("GTiff").Create(
                output_file,
                geocode_obj.geogrid_width,
                geocode_obj.geogrid_length,
                1,
                gdal_dtype,
            )

            # Set output file's projection and geotransform
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            output_ds.SetProjection(srs.ExportToWkt())

            output_geotransform = [
                geocode_obj.geogrid_start_x,
                geocode_obj.geogrid_spacing_x,
                0,
                geocode_obj.geogrid_start_y,
                0,
                geocode_obj.geogrid_spacing_y,
            ]
            output_ds.SetGeoTransform(output_geotransform)

            output_ds.FlushCache()
            output_ds = None  # Close
            output_raster = isce3.io.Raster(str(output_file), update=True)

            # Perform geocoding
            geocode_obj.geocode(
                radar_grid=isce3_radargrid,
                input_raster=input_raster,
                output_raster=output_raster,
                dem_raster=dem,
                output_mode=isce3.geocode.GeocodeOutputMode.INTERP,
            )

            # Explicitly close the ISCE3 Rasters
            input_raster = None
            output_raster = None
            dem = None

            # Read geocoded result
            output_ds = gdal.Open(output_file)
            geocoded_array = (
                output_ds.GetRasterBand(1).ReadAsArray().astype(output_dtype)
            )
            output_gt = list(output_ds.GetGeoTransform())
            output_ds = None

            # Create GeoGrid for the output
            # GDAL geotransform uses upper-left corner convention,
            # but nisarqa.GeoGrid uses pixel center convention.
            # Convert from corner to center coordinates:
            # GT(0) = x-coordinate of upper-left corner of upper-left pixel
            # GT(3) = y-coordinate of upper-left corner of upper-left pixel
            # GT(1) = pixel width (x spacing)
            # GT(5) = pixel height (y spacing, negative for north-up)

            # Calculate pixel center coordinates
            reproj_height, reproj_width = geocoded_array.shape

            x_coords = (
                output_gt[0] + (np.arange(reproj_width) + 0.5) * output_gt[1]
            )
            y_coords = (
                output_gt[3] + (np.arange(reproj_height) + 0.5) * output_gt[5]
            )

            output_geogrid = nisarqa.GeoGrid(
                epsg=epsg,
                x_axis_posting=output_gt[1],
                x_coordinates=x_coords,
                y_axis_posting=output_gt[5],
                y_coordinates=y_coords,
            )

        # DEM context manager has exited - temporary DEM cleaned up if needed
        return geocoded_array, output_geogrid

    finally:
        # Delete temp input/output files (always executes, even if exception occurred)
        input_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)


def reproject_geo_raster(
    image_array: np.ndarray,
    fill_value: float,
    geogrid: nisarqa.GeoGrid,
    *,
    output_epsg: int,
    resample: str = "average",
) -> tuple[np.ndarray, nisarqa.GeoGrid]:
    """
    Reproject a geocoded raster array.

    Parameters
    ----------
    image_array : numpy.ndarray
        2D array of image data to be reprojected.
        Should be real-valued (float). Complex data is not supported.
    fill_value : float or None
        Fill value for `image_array`.
    geogrid : nisarqa.GeoGrid
        GeoGrid defining the coordinate system and spacing of `image_array`.
    output_epsg : int
        EPSG to reproject the raster to. Example: 4326 for lat/lon.
    resample : str, optional
        Resampling algorithm for GDAL reprojection. Options: 'near', 'bilinear',
        'cubic', 'cubicspline', 'lanczos', 'average', 'mode'.
        Default: 'average'.

    Returns
    -------
    reprojected_array : numpy.ndarray
        2D array of reprojected data in the output EPSG.
        The output dimensions will match the input dimensions (height and width
        in pixels are preserved), which may introduce geometric distortion when
        reprojecting between coordinate systems with different scale factors.
    output_geogrid : nisarqa.GeoGrid
        GeoGrid object describing the coordinate system and grid of the
        reprojected array. Uses pixel center convention.

    Warnings
    --------
    This function is designed for visual browse image generation and
    prioritizes predictable output dimensions and visual appearance.
    The output dimensions are constrained to match the input dimensions, which
    may introduce slight distortion when reprojecting between coordinate systems
    (e.g., from UTM meters to EPSG 4326 degrees, or in regions near the poles).

    This function is not optimized for large, full-size NISAR rasters.
    Recommend only using it to reproject relatively small rasters, such as
    browse image arrays.
    """
    if np.iscomplexobj(image_array):
        raise ValueError(
            f"image_array is complex-valued. Only real-valued data"
            " currently supported."
        )

    if np.iscomplexobj(fill_value):
        raise TypeError(
            f"fill_value must be real-valued, but received complex: {fill_value}. "
            "Convert to float (e.g., using np.real()) before calling this function."
        )

    # Use scratch directory with temporary files for ISCE3 raster I/O.
    # GDAL's in-memory option has potential security issues.
    scratch = nisarqa.get_global_scratch_dir()
    # Create a uniquely-named string for filenaming
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

    # Ensure float type
    raster_array = image_array.astype(np.float64)

    source_epsg = geogrid.epsg

    # Calculate geotransform for the source raster
    # GDAL geotransform: [x_origin, x_pixel_size, 0, y_origin, 0, y_pixel_size]
    # GeoGrid uses pixel centers, so we need x_start/y_start which are the edges
    geotransform = [
        geogrid.x_start,
        geogrid.x_axis_posting,
        0,
        geogrid.y_start,
        0,
        geogrid.y_axis_posting,
    ]

    # Setup temporary file paths
    source_file = scratch / f"input_{utc_now}.tif"
    reprojected_file = scratch / f"reprojected_{utc_now}.tif"

    try:
        # Create temporary input source GeoTIFF
        driver = gdal.GetDriverByName("GTiff")
        src_height, src_width = raster_array.shape
        source_ds = driver.Create(
            source_file, src_width, src_height, 1, gdal.GDT_Float64
        )

        # Set projection and geotransform
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(source_epsg)
        source_ds.SetProjection(srs.ExportToWkt())
        source_ds.SetGeoTransform(geotransform)

        # Write data
        source_ds.GetRasterBand(1).WriteArray(raster_array)
        # Make sure to set the fill value
        # Reason: GDAL uses all non-NODATA pixels during resampling.
        # If the fill value is not designated, then the fill value pixels will
        # (unfortunately) be used in the resampling.
        source_ds.GetRasterBand(1).SetNoDataValue(float(fill_value))

        source_ds.FlushCache()
        source_ds = None  # Close

        # Specify warp options; handle antimeridian crossings.
        warp_options = {
            "srcSRS": f"EPSG:{source_epsg}",
            "resampleAlg": resample,
            "format": "GTiff",
            "dstNodata": fill_value,
            # Constrain output dimensions to match input dimensions.
            # This is intentional for browse image generation to ensure
            # predictable output shapes and improve visual appearance
            # for the browse products (particularly in polar regions),
            # regardless of the source and target projections.
            # This means that the output's x spacing and y spacing will no
            # longer be approx. equal (i.e. output won't have ~square pixels).
            "height": src_height,
            "width": src_width,
        }

        if output_epsg == 4326 and geogrid.crosses_antimeridian:
            # For antimeridian crossing, use +lon_wrap=180 to shift the
            # coordinate system center to 180° (dateline) instead of 0°
            # (prime meridian).
            # This avoids the discontinuity at -180°/+180° and allows GDAL to
            # properly handle data that spans the dateline.
            warp_options["dstSRS"] = "+proj=longlat +datum=WGS84 +lon_wrap=180"
        else:
            # Standard reprojection for non-dateline-crossing data
            warp_options["dstSRS"] = f"EPSG:{output_epsg}"

        gdal.Warp(
            reprojected_file,
            source_file,
            options=gdal.WarpOptions(**warp_options),
        )

        # Open reprojected image
        reprojected_ds = gdal.Open(reprojected_file)
        reproj_data = reprojected_ds.GetRasterBand(1).ReadAsArray()
        reproj_gt = list(reprojected_ds.GetGeoTransform())
        reprojected_ds = None  # Close

        # Create GeoGrid for the output
        # GDAL geotransform uses upper-left corner convention,
        # but nisarqa.GeoGrid uses pixel center convention.
        # Convert from corner to center coordinates:
        # GT(0) = x-coordinate of upper-left corner of upper-left pixel
        # GT(3) = y-coordinate of upper-left corner of upper-left pixel
        # GT(1) = pixel width (x spacing)
        # GT(5) = pixel height (y spacing, negative for north-up)

        # Calculate pixel center coordinates
        reproj_height, reproj_width = reproj_data.shape

        x_coords = reproj_gt[0] + (np.arange(reproj_width) + 0.5) * reproj_gt[1]
        y_coords = (
            reproj_gt[3] + (np.arange(reproj_height) + 0.5) * reproj_gt[5]
        )

        output_geogrid = nisarqa.GeoGrid(
            epsg=output_epsg,
            x_axis_posting=reproj_gt[1],
            x_coordinates=x_coords,
            y_axis_posting=reproj_gt[5],
            y_coordinates=y_coords,
        )

        return reproj_data, output_geogrid

    finally:
        # Delete temp files (always executes, even if exception occurred)
        source_file.unlink(missing_ok=True)
        reprojected_file.unlink(missing_ok=True)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
