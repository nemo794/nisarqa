from __future__ import annotations

import os
import pathlib
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone

import isce3
import numpy as np
from nisar.workflows.stage_dem import apply_margin_to_geographic_box
from osgeo import gdal, osr
from shapely.wkt import loads

import nisarqa

objects_to_skip = nisarqa.get_all(__name__)


def compute_geogrid(
    bounding_polygon: str,
    epsg: int,
    longest_side_max: int = 2048,
    margin_in_km: int = 5,
) -> isce3.product.GeoGridParameters:
    """
    Compute the geogrid parameters for a given bounding polygon and image size.

    Parameters
    ----------
    bounding_polygon : str
        WKT string representing the approximate bounding polygon
        of the radar swath.
    epsg : int
        EPSG of the output geogrid. Example: 4326 for lat/lon.
    margin_in_km : float, optional
        Margin in kilometers to add around the bounds to account for
        topography when geocoding (default: 5).
    longest_side_max : int, optional
        Maximum number of pixels allowed for the longest side of the
        output geogrid. The resolution is determined by dividing the
        longest extent by this value (default: 2048).

    Returns
    -------
    geogrid : isce3.product.GeoGridParameters
        Computed geogrid.
    """

    if longest_side_max <= 0:
        raise ValueError(f"{longest_side_max=}, must be greater than 0.")
    if margin_in_km <= 0:
        raise ValueError(f"{margin_in_km=}, must be greater than 0.")

    # Parse WKT and apply margin
    poly = loads(bounding_polygon)
    poly_with_margin = apply_margin_to_geographic_box(
        poly, margin_in_km=margin_in_km
    )
    bounds = poly_with_margin.bounds  # (minx, miny, maxx, maxy)

    # Calculate extents in degrees
    width_deg = bounds[2] - bounds[0]
    length_deg = bounds[3] - bounds[1]

    # Determine resolution based on longest_side_max
    # (The longest side should have at most longest_side_max pixels)
    longest_extent = max(width_deg, length_deg)
    resolution = longest_extent / longest_side_max

    if resolution <= 0:
        raise ValueError(
            f"Resolution computed from bounding polygon is {resolution}, must"
            " be greater than 0."
        )

    start_x = bounds[0]
    start_y = bounds[3]  # Y starts at top

    # Calculate dimensions ensuring we don't exceed longest_side_max
    width = int(np.ceil(width_deg / resolution))
    length = int(np.ceil(length_deg / resolution))

    # Ensure neither dimension exceeds the max
    width = min(width, longest_side_max)
    length = min(length, longest_side_max)

    geogrid = isce3.product.GeoGridParameters(
        start_x=start_x,
        start_y=start_y,
        spacing_x=resolution,
        spacing_y=-resolution,
        width=width,
        length=length,
        epsg=epsg,
    )

    return geogrid


def get_zero_height_dem(
    width: int,
    length: int,
    epsg: int,
    start_x: float,
    spacing_x: float,
    start_y: float,
    spacing_y: float,
) -> pathlib.Path:
    """
    Generate a zero-height DEM TIF file in the scratch directory.

    Creates a GeoTIFF file containing an array of zeros with the specified
    dimensions and geotransform. The file is uniquely named with a timestamp
    to avoid collisions. This is useful for geocoding operations where
    a DEM is required but high accuracy is not critical (e.g., for browse
    image generation).

    Parameters
    ----------
    width : int
        Width of the DEM array in pixels.
    length : int
        Length (height) of the DEM array in pixels.
    epsg : int
        EPSG code defining the coordinate system of the DEM.
    start_x : float
        X-coordinate of the upper-left corner of the upper-left pixel.
    spacing_x : float
        Pixel spacing in the X direction (typically positive for west-to-east).
    start_y : float
        Y-coordinate of the upper-left corner of the upper-left pixel.
    spacing_y : float
        Pixel spacing in the Y direction (typically negative for north-to-south).

    Returns
    -------
    pathlib.Path
        Path to the uniquely-named zero-height DEM file, located in the
        nisarqa global scratch directory.

    Notes
    -----
    The caller is responsible for deleting the returned file when it is
    no longer needed. Use `file_path.unlink(missing_ok=True)` to safely
    remove the file.
    """

    # Create a uniquely-named file in the scratch directory
    scratch = nisarqa.get_global_scratch_dir()
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

    dem_file = scratch / f"zero_height_dem_{utc_now}.tif"

    dem_ds = gdal.GetDriverByName("GTiff").Create(
        dem_file, width, length, 1, gdal.GDT_Float64
    )

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    dem_ds.SetProjection(srs.ExportToWkt())

    output_geotransform = [
        start_x,
        spacing_x,
        0,
        start_y,
        0,
        spacing_y,
    ]
    dem_ds.SetGeoTransform(output_geotransform)
    dem_ds.GetRasterBand(1).WriteArray(
        np.zeros((length, width), dtype=np.float64)
    )
    dem_ds.FlushCache()
    dem_ds = None

    return dem_file


def geocode_radar_raster(
    radar_raster: nisarqa.RadarRaster,
    orbit: isce3.core.Orbit,
    wavelength: float,
    look_side: isce3.core.LookSide | str,
    geogrid: isce3.product.GeoGridParameters,
    dem_file: str | os.PathLike | None = None,
    resample: str = "biquintic",
) -> np.ndarray:
    """
    Geocode a radar-grid raster array onto the given geogrid.

    This function is used for Level-1 products (RSLC, RIFG, RUNW, ROFF).
    It uses ISCE3's geocoding functionality to transform data from
    radar coordinates (range-Doppler) to geographic coordinates (lon/lat).

    Parameters
    ----------
    radar_raster : nisarqa.RadarRaster
        RadarRaster whose `data` image will be be geocoded.
        Should be real-valued (float).
    orbit : isce3.core.Orbit
        The trajectory of the radar antenna phase center.
    wavelength : float
        The radar central wavelength, in meters.
    look_side : isce3.core.LookSide or {'left', 'right'}
        The look direction of the radar (left-looking or right-looking).
    geogrid : isce3.product.GeoGridParameters
        ISCE3 geogrid parameters specifying the output geocoded array.
    dem_file : str or path-like or None, optional
        Path to a DEM file; required for accurate geolocation of the pixels.
        If None, a zero-height DEM will be used.
        Defaults to None.
    resample : str, optional
        Resampling method for ISCE3 geocoding. Options: 'sinc', 'bilinear',
        'bicubic', 'nearest', 'biquintic'. Default: 'biquintic'.

    Returns
    -------
    geocoded_array : numpy.ndarray
        2D array of geocoded data.

    Warning
    -------
    This function is not optimized for large, full-size NISAR rasters.
    Recommend only using it to geocode relatively small rasters, such as
    browse image arrays.
    To geocode full-size NISAR rasters, suggest using ISCE3 directly.

    See Also
    --------
    get_zero_height_dem :
        Creates a temporary zero-height DEM file. Useful for e.g. testing this
        function, but will not lead to highly-accurate geolocation
        of the image pixels.
    """

    if radar_raster.is_complex:
        raise ValueError(
            f"RadarRaster is complex-valued. Only real-valued data"
            " currently supported."
        )

    # TODO - test to empirically determine a threshold for the warning
    if max(radar_raster.data.shape) > 50000:
        msg = (
            f"Raster has shape {radar_raster.data.shape} which may be slow"
            " to geocode. Consider using ISCE3 directly for large rasters."
        )
        nisarqa.get_logger().warning(msg)
        warnings.warn(msg)

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
    dem_filepath = None

    try:
        # Setup temporary input raster file, so that it can be converted to
        # the correct format for ISCE3 geocoding
        input_ds = gdal.GetDriverByName("GTiff").Create(
            input_file,
            radar_raster.data.shape[1],
            radar_raster.data.shape[0],
            1,
            gdal_dtype,
        )

        # Ensure float type
        raster_array = radar_raster.data.astype(np.float64)

        input_ds.GetRasterBand(1).WriteArray(raster_array)
        input_ds.FlushCache()
        input_ds = None  # Close
        input_raster = isce3.io.Raster(input_file)

        # Create temporary output raster file
        output_ds = gdal.GetDriverByName("GTiff").Create(
            output_file, geogrid.width, geogrid.length, 1, gdal_dtype
        )

        # Set output file's projection and geotransform
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(geogrid.epsg)
        output_ds.SetProjection(srs.ExportToWkt())
        output_geotransform = [
            geogrid.start_x,
            geogrid.spacing_x,
            0,
            geogrid.start_y,
            0,
            geogrid.spacing_y,
        ]
        output_ds.SetGeoTransform(output_geotransform)
        output_ds.FlushCache()
        output_ds = None  # Close
        output_raster = isce3.io.Raster(output_file, update=True)

        # Set up geocoding object
        geocode_obj = isce3.geocode.GeocodeFloat64()

        geocode_obj.orbit = orbit
        geocode_obj.ellipsoid = isce3.core.WGS84_ELLIPSOID
        geocode_obj.doppler = isce3.core.LUT2d()  # Zero-Doppler for NISAR
        geocode_obj.threshold_geo2rdr = 1.0e-8
        geocode_obj.numiter_geo2rdr = 25

        # Set data interpolator based on resample parameter
        # TODO - can we skip this mapping and instead simply pass the str
        # to `geocode_obj.data_interpolator`?
        resample_map = {
            "sinc": isce3.core.DataInterpMethod.SINC,
            "bilinear": isce3.core.DataInterpMethod.BILINEAR,
            "bicubic": isce3.core.DataInterpMethod.BICUBIC,
            "nearest": isce3.core.DataInterpMethod.NEAREST,
            "biquintic": isce3.core.DataInterpMethod.BIQUINTIC,
        }
        if resample not in resample_map:
            raise ValueError(
                f"Invalid resample method: {resample}. "
                f"Must be one of {list(resample_map.keys())}"
            )
        geocode_obj.data_interpolator = resample_map[resample]

        # Set the output geocoding object's geogrid
        geocode_obj.geogrid(
            geogrid.start_x,
            geogrid.start_y,
            geogrid.spacing_x,
            geogrid.spacing_y,
            geogrid.width,
            geogrid.length,
            geogrid.epsg,
        )

        # Set up DEM
        if dem_file is None:
            # Note: We will be responsible for deleting this temp file
            dem_filepath = get_zero_height_dem(
                width=geogrid.width,
                length=geogrid.length,
                epsg=geogrid.epsg,
                start_x=geogrid.start_x,
                spacing_x=geogrid.spacing_x,
                start_y=geogrid.start_y,
                spacing_y=geogrid.spacing_y,
            )
        else:
            dem_filepath = dem_file

        dem = isce3.io.Raster(dem_filepath)

        # Setup radar grid parameters
        radargrid = radar_raster.get_isce3_radar_grid_parameters(
            wavelength=wavelength,
            look_side=look_side,
        )

        # Perform geocoding
        geocode_obj.geocode(
            radar_grid=radargrid,
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
        output_ds = None

        return geocoded_array

    finally:
        # Delete temp files (always executes, even if exception occurred)
        input_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)
        if dem_file is None and dem_filepath is not None:
            dem_filepath.unlink(missing_ok=True)


def reproject_geo_raster(
    geo_raster: nisarqa.GeoRaster,
    *,
    output_epsg: int,
    longest_side_max: int = 2048,
    resample: str = "cubic",
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    Reproject a geocoded raster array.

    Parameters
    ----------
    geo_raster : nisarqa.GeoRaster
        GeoRaster whose `data` image will be reprojected.
        Should be real-valued (float).
    output_epsg : int
        EPSG to reproject the raster to. Example: 4326 for lat/lon.
    longest_side_max : int, optional
        Maximum number of pixels for the longest side of the output image.
        The image will be resized if necessary. Default: 2048.
    resample : str, optional
        Resampling algorithm for GDAL reprojection. Options: 'near', 'bilinear',
        'cubic', 'cubicspline', 'lanczos', 'average', 'mode'. Default: 'cubic'.

    Returns
    -------
    reprojected_array : numpy.ndarray
        2D array of reprojected data in the output EPSG.
    corners : tuple of float
        Corner coordinates for `reprojected_array`: (west, south, east, north).
        These coordinates define the full extent of image grid, e.g.
        the left side of the left-most pixels, lower side of the bottom pixels,
        the right side of the right-most pixels, upper edge of the top pixels.

    Warning
    -------
    This function is not optimized for large, full-size NISAR rasters.
    Recommend only using it to geocode relatively small rasters, such as
    browse image arrays.
    To geocode full-size NISAR rasters, suggest using ISCE3 directly.
    """
    if geo_raster.is_complex:
        raise ValueError(
            f"GeoRaster is complex-valued. Only real-valued data"
            " currently supported."
        )

    # Validate resample parameter
    valid_resample = [
        "near",
        "bilinear",
        "cubic",
        "cubicspline",
        "lanczos",
        "average",
        "mode",
    ]
    if resample not in valid_resample:
        raise ValueError(
            f"Invalid resample method: {resample}. "
            f"Must be one of {valid_resample}"
        )

    # Use scratch directory with temporary files for ISCE3 raster I/O.
    # GDAL's in-memory option has potential security issues.
    scratch = nisarqa.get_global_scratch_dir()
    # Create a uniquely-named string for filenaming
    utc_now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")

    # Ensure float type
    raster_array = geo_raster.data.astype(np.float64)

    source_epsg = geo_raster.epsg

    # Get geogrid parameters from the GeoRaster
    geogrid = geo_raster.get_isce3_geo_grid_parameters()

    # Calculate geotransform for the source raster
    # GDAL geotransform: [x_origin, x_pixel_size, 0, y_origin, 0, y_pixel_size]
    geotransform = [
        geogrid.start_x,
        geogrid.spacing_x,
        0,
        geogrid.start_y,
        0,
        geogrid.spacing_y,
    ]

    # Setup temporary file paths
    source_file = scratch / f"input_{utc_now}.tif"
    reprojected_file = scratch / f"reprojected_{utc_now}.tif"
    resized_file = None

    try:
        # Create temporary input source GeoTIFF
        driver = gdal.GetDriverByName("GTiff")
        height, width = raster_array.shape
        source_ds = driver.Create(
            source_file, width, height, 1, gdal.GDT_Float64
        )

        # Set projection and geotransform
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(source_epsg)
        source_ds.SetProjection(srs.ExportToWkt())
        source_ds.SetGeoTransform(geotransform)

        # Write data
        source_ds.GetRasterBand(1).WriteArray(raster_array)
        source_ds.FlushCache()
        source_ds = None  # Close

        # Reproject to desired EPSG
        warp_options = gdal.WarpOptions(
            srcSRS=f"EPSG:{source_epsg}",
            dstSRS=f"EPSG:{output_epsg}",
            resampleAlg=resample,
            format="GTiff",
        )
        gdal.Warp(reprojected_file, source_file, options=warp_options)

        # Open reprojected image
        reprojected_ds = gdal.Open(reprojected_file)
        reprojected_data = reprojected_ds.GetRasterBand(1).ReadAsArray()
        reprojected_gt = reprojected_ds.GetGeoTransform()

        # Get dimensions
        orig_height, orig_width = reprojected_data.shape
        longest_side = max(orig_height, orig_width)

        # Resize if necessary
        if longest_side > longest_side_max:
            scale_factor = longest_side_max / longest_side
            new_width = int(orig_width * scale_factor)
            new_height = int(orig_height * scale_factor)

            # Resample using GDAL
            resized_file = scratch / f"resized_{utc_now}.tif"
            gdal.Translate(
                resized_file,
                reprojected_file,
                width=new_width,
                height=new_height,
                resampleAlg=resample,
            )

            # Read resized data
            resized_ds = gdal.Open(resized_file)
            final_data = resized_ds.GetRasterBand(1).ReadAsArray()
            final_gt = resized_ds.GetGeoTransform()
            final_width = new_width
            final_height = new_height
            resized_ds = None
        else:
            final_data = reprojected_data
            final_gt = reprojected_gt
            final_width = orig_width
            final_height = orig_height

        reprojected_ds = None  # Close

        # Calculate corner coordinates for the output
        # GT(0) : x-coordinate of the upper-left corner of the upper-left pixel.
        west = final_gt[0]
        # GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
        north = final_gt[3]
        # GT(1) w-e pixel resolution / pixel width.
        east = final_gt[0] + final_width * final_gt[1]
        # GT(5) n-s pixel resolution / pixel height (negative value
        # for a north-up image).
        south = final_gt[3] + final_height * final_gt[5]

        return final_data, (west, south, east, north)

    finally:
        # Delete temp files (always executes, even if exception occurred)
        source_file.unlink(missing_ok=True)
        reprojected_file.unlink(missing_ok=True)
        if resized_file is not None:
            resized_file.unlink(missing_ok=True)


__all__ = nisarqa.get_all(__name__, objects_to_skip)
