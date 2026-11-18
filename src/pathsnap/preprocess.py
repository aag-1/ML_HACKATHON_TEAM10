"""Per-observation preprocessing: projection, time gaps, smoothed speed/heading, moving flag."""
import numpy as np
import pandas as pd

from . import config as C
from .geo import bearing


def preprocess_track(track, project):
    xy = project(track.lon.to_numpy(), track.lat.to_numpy())
    t = track.t.to_numpy()
    n = len(t)
    out = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "t": t})
    out["dt_prev"] = np.r_[0.0, np.diff(t)]
    out["gap_before"] = out.dt_prev > C.GAP_S
    step = np.r_[0.0, np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))]
    out["is_duplicate"] = step == 0

    # Smoothed velocity: displacement between points i-k and i+k (indices clipped at the ends)
    k = C.HEADING_WINDOW
    lo = np.clip(np.arange(n) - k, 0, n - 1)
    hi = np.clip(np.arange(n) + k, 0, n - 1)
    dx, dy = xy[hi, 0] - xy[lo, 0], xy[hi, 1] - xy[lo, 1]
    dt = t[hi] - t[lo]
    speed = np.where(dt > 0, np.hypot(dx, dy) / np.where(dt > 0, dt, 1), 0.0)
    out["speed"] = np.minimum(speed, C.SPEED_CLIP_MS)
    out["moving"] = (out.speed >= C.MOVING_SPEED_MS).astype(np.int8)

    # Heading is only meaningful while moving; a stopped car keeps its last moving heading.
    h = pd.Series(np.where(out.moving == 1, bearing(dx, dy), np.nan))
    out["heading"] = h.ffill().bfill().fillna(0.0).to_numpy()
    return out


def long_window_speed(pts, k=10):
    """Speed from the displacement between points i-k and i+k (~20 s at 1 Hz). GPS jitter of a parked car
    cancels out over this window, so this separates 'really travelling' from 'parked but jittering'
    (the +-2-point speed reads ~2 m/s for a parked car with 5-8 m jitter)."""
    xy, t = pts[["x", "y"]].to_numpy(), pts.t.to_numpy()
    n = len(t)
    i = np.arange(n)
    lo, hi = np.clip(i - k, 0, n - 1), np.clip(i + k, 0, n - 1)
    dt = t[hi] - t[lo]
    return np.where(dt > 0, np.hypot(*(xy[hi] - xy[lo]).T) / np.where(dt > 0, dt, 1), 0.0)
