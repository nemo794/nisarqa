from dataclasses import dataclass
import isce3
import nisar
import nisarqa
import numpy as np
import os
import textwrap

@dataclass
class LonLat:
    """
    A point in Lon/Lat space.

    Attributes
    ----------
    lon, lat : float
        The geodetic longitude and latitude, in radians.
    """
    lon: float
    lat: float

@dataclass
class LatLonQuad:
    """
    A quadrilateral defined by four Lon/Lat corner points.

    This class represents a KML gx:LatLonQuad, as described in
    https://developers.google.com/kml/documentation/kmlreference#gx:latlonquad

    The corners are provided as follows:
        * ul - upper-left
        * ur - upper-right
        * ll - lower-left
        * lr - lower-right

    Note that "upper", "lower", "left" and "right" are given from the image's
    native perspective prior to transformation to lon/lat coordinates, so e.g.
    the "upper-left" coordinate of a radar image is not necessarily the
    upper-left in lon/lat space, but in the un-geocoded radar image. This is
    done to provide the proper orientation of the overlay image.

    Attributes
    ----------
    ul, ur, ll, lr : LonLat
        The upper-left, upper-right, lower-left, and lower-right corners.
    """
    ul: LonLat
    ur: LonLat
    ll: LonLat
    lr: LonLat


def get_latlonquad(input_file: str | os.PathLike[str]) -> LatLonQuad:
    """
    Create a LatLonQuad for the corners of the input product,
    by geocoding the corners of the radar grid.

    Currently only implemented for RSLC files, will need to support
    other types of radar products and geocoded products.

    Parameters
    ----------
    input_file : path-like
        The path to the input RSLC product

    Returns
    -------
    llq : LatLonQuad
        A LatLonQuad object containing the four corner coordinates for the
        Frequency A images in `input_file`. (If Frequency A is not available,
        then Frequency B images will be used.)
    """
    input_file = os.fspath(input_file)
    product = nisar.products.readers.open_product(input_file)

    identification_path = product.IdentificationPath
    with nisarqa.open_h5_file(input_file, mode="r") as in_file:
        is_geocoded = in_file[identification_path]["isGeocoded"][()] == b"True"

    freq = "A" if ("A" in product.frequencies) else "B"

    if is_geocoded:
        raise NotImplementedError("Can only get LatLonQuad for radar products")
    else:
        orbit = product.getOrbit()
        radar_grid = product.getRadarGrid(freq)

        image_grid_doppler = 0.0  # assume zero-doppler for NISAR
        ellipsoid = isce3.core.Ellipsoid()  # assume WGS84 for NISAR

        # zero-height DEM
        dem = isce3.geometry.DEMInterpolator()

        geo_corners = ()
        for az in (radar_grid.sensing_start, radar_grid.sensing_stop):
            for rg in (radar_grid.starting_range, radar_grid.end_range):
                lon, lat, _ = isce3.geometry.rdr2geo(
                    aztime=az,
                    range=rg,
                    orbit=orbit,
                    side=radar_grid.lookside,
                    doppler=image_grid_doppler,
                    wavelength=radar_grid.wavelength,
                    dem=dem,
                    ellipsoid=ellipsoid,
                )
                geo_corners += (LonLat(lon, lat),)

    return LatLonQuad(*geo_corners)


def write_latlonquad_to_kml(
    llq: LatLonQuad,
    output_dir: str | os.PathLike[str],
    *,
    kml_filename: str = "BROWSE.kml",
    png_filename: str = "BROWSE.png",
) -> None:
    """
    Generate a KML file containing geolocation info of the corresponding
    browse image.

    Parameters
    ----------
    llq : LatLonQuad
        The LatLonQuad object containing the corner coordinates that will be
        serialized to KML.
    output_dir : path-like
        The directory to write the output KML file to. This directory
        must already exist. The PNG file that the KML corresponds to is
        expected to be placed in the same directory.
    kml_filename : str, optional
        The output filename of the KML file, specified relative to
        `output_dir`. Defaults to 'BROWSE.kml'.
    png_filename : str, optional
        The filename of the corresponding PNG file, specified relative
        to `output_dir`. Defaults to 'BROWSE.png'.
    """

    # extract upper/lower left/right corners
    ul, ur, ll, lr = llq.ul, llq.ur, llq.ll, llq.lr

    # convert lon/lat radians to string in degrees suitable for LatLonQuad
    ll_str = lambda c: str(np.rad2deg(c.lon)) + "," + str(np.rad2deg(c.lat))

    # The coordinates are specified in counter-clockwise order with the first
    # point corresponding to the lower-left corner of the overlayed image.
    # (https://developers.google.com/kml/documentation/kmlreference#gx:latlonquad)
    kml_file = textwrap.dedent(
        f"""
        <?xml version="1.0" encoding="UTF-8"?>
        <kml xmlns:gx="http://www.google.com/kml/ext/2.2">
          <Document>
            <name>overlay image</name>
            <GroundOverlay>
              <name>overlay image</name>
              <Icon>
                <href>{png_filename}</href>
              </Icon>
              <gx:LatLonQuad>
                <coordinates>{' '.join(ll_str(llh) for llh in (ll, lr, ur, ul))}</coordinates>
              </gx:LatLonQuad>
            </GroundOverlay>
          </Document>
        </kml>
        """
    ).strip()
    with open(os.path.join(output_dir, kml_filename), "w") as f:
        f.write(kml_file)