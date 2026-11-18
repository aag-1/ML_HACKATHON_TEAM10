"""Candidate generation: for each GPS point, the arcs within CAND_RADIUS_M (at most CAND_MAX nearest).

Arcs are straight segments, possibly kilometres long, so we sample points every DENSIFY_SPACING_M along
each arc, put them in a KD-tree, find samples near each GPS point, then compute the exact distance.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import config as C
from .geo import point_to_segment


def build_index(nodes_xy, arcs, pts_xy, spacing=C.DENSIFY_SPACING_M, radius=C.CAND_RADIUS_M):
    a, b = nodes_xy[arcs[:, 0]], nodes_xy[arcs[:, 1]]
    # only arcs whose bounding box comes near the track
    lo, hi = pts_xy.min(0) - radius - spacing, pts_xy.max(0) + radius + spacing
    keep = np.where(np.all(np.minimum(a, b) <= hi, 1) & np.all(np.maximum(a, b) >= lo, 1))[0]
    a, b = a[keep], b[keep]
    length = np.linalg.norm(b - a, axis=1)
    n_s = np.ceil(length / spacing).astype(int) + 1
    owner = np.repeat(np.arange(len(keep)), n_s)
    start = np.r_[0, np.cumsum(n_s)[:-1]]
    frac = (np.arange(n_s.sum()) - start[owner]) / np.maximum(n_s[owner] - 1, 1)
    samples = a[owner] + frac[:, None] * (b[owner] - a[owner])
    return {"tree": cKDTree(samples), "owner": keep[owner], "spacing": spacing}


def generate(pts_xy, nodes_xy, arcs, index, radius=C.CAND_RADIUS_M, kmax=C.CAND_MAX):
    """Returns DataFrame [pt, arc, dist, u, arc_len] sorted by (pt, dist), <= kmax rows per point."""
    hits = index["tree"].query_ball_point(pts_xy, r=radius + index["spacing"] / 2 + 1.0)
    counts = np.fromiter((len(h) for h in hits), int, len(hits))
    pt = np.repeat(np.arange(len(pts_xy)), counts)
    arc = index["owner"][np.concatenate([np.asarray(h, int) for h in hits]) if counts.sum() else np.empty(0, int)]
    key = np.unique(pt.astype(np.int64) * len(arcs) + arc)
    pt, arc = key // len(arcs), key % len(arcs)

    dist, u, seg_len = point_to_segment(pts_xy[pt], nodes_xy[arcs[arc, 0]], nodes_xy[arcs[arc, 1]])
    df = pd.DataFrame({"pt": pt, "arc": arc, "dist": dist, "u": u, "arc_len": seg_len})
    df = df[df.dist <= radius].sort_values(["pt", "dist", "arc"], kind="stable")
    df = df[df.groupby("pt").cumcount() < kmax].reset_index(drop=True)
    return df
