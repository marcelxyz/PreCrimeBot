from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from .file_handler import load_m25_coordinates

__m25 = Polygon(load_m25_coordinates())


def is_within_m25(lat, lon):
    """Checks if a given geographical coordinate falls within the M25."""
    return __m25.contains(Point(lat, lon))
