"""Geometry: per-trip UTM projection (metres) and vectorised point-to-segment maths."""
import numpy as np
from pyproj import Transformer


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_projector(track):
    """UTM zone chosen from the trip's median GPS position; the same zone is used for nodes."""
    epsg = utm_epsg(float(track.lon.median()), float(track.lat.median()))
    tf = Transformer.from_crs(4326, epsg, always_xy=True)
    return lambda lon, lat: np.column_stack(tf.transform(np.asarray(lon), np.asarray(lat)))


def point_to_segment(p, a, b):
    """p, a, b: (..., 2). Returns perpendicular distance, fraction u in [0,1] along a->b, segment length."""
    ab = b - a
    seg_len2 = (ab ** 2).sum(-1)
    u = np.where(seg_len2 > 0, ((p - a) * ab).sum(-1) / np.where(seg_len2 > 0, seg_len2, 1), 0.0)
    u = np.clip(u, 0.0, 1.0)
    closest = a + u[..., None] * ab
    return np.linalg.norm(p - closest, axis=-1), u, np.sqrt(seg_len2)


def bearing(dx, dy):
    """Compass-style bearing in radians (0 = north/+y, clockwise)."""
    return np.arctan2(dx, dy)


def angle_diff(a, b):
    """Absolute smallest difference between two angles, in [0, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)
