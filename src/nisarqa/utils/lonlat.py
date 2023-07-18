from dataclasses import dataclass

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
