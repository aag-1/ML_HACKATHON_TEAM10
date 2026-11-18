"""Feature engineering for (GPS point, candidate arc) pairs.

Three groups (decision D9):
  geometry  - dist, u, d_end, arc_len
  direction - heading_diff, cos_heading_diff, speed, moving
  relative  - dist_rank, dist_gap, n_cand, has_twin, heading_gap
All features are computed from the GPS trace and the road graph only; no label information is used.
"""
import numpy as np

from .geo import angle_diff, bearing

FEATURES = [
    "dist", "u", "d_end", "arc_len",
    "heading_diff", "cos_heading_diff", "speed", "moving",
    "dist_rank", "dist_gap", "n_cand", "has_twin", "heading_gap",
]


def add_features(cands, pts, nodes_xy, arcs):
    df = cands.copy()
    a, b = nodes_xy[arcs[df.arc, 0]], nodes_xy[arcs[df.arc, 1]]
    p = pts.iloc[df.pt.to_numpy()]

    # geometry
    df["d_end"] = np.minimum(df.u, 1 - df.u) * df.arc_len

    # direction: GPS heading vs the arc's travel direction (separates A->B from its B->A twin)
    arc_bearing = bearing(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
    hd = angle_diff(p.heading.to_numpy(), arc_bearing)
    df["heading_diff"] = np.degrees(hd)
    df["cos_heading_diff"] = np.cos(hd)
    df["speed"] = p.speed.to_numpy()
    df["moving"] = p.moving.to_numpy()

    # relative to the other candidates of the same GPS point
    g = df.groupby("pt")
    df["dist_rank"] = g.dist.rank(method="min")
    df["dist_gap"] = df.dist - g.dist.transform("min")
    df["n_cand"] = g.dist.transform("size")
    df["heading_gap"] = df.heading_diff - g.heading_diff.transform("min")
    n = len(nodes_xy)
    fwd = df.pt.to_numpy() * n * n + arcs[df.arc, 0] * n + arcs[df.arc, 1]
    rev = df.pt.to_numpy() * n * n + arcs[df.arc, 1] * n + arcs[df.arc, 0]
    df["has_twin"] = np.isin(rev, fwd).astype(np.int8)
    df["arc_u"], df["arc_v"] = arcs[df.arc, 0], arcs[df.arc, 1]
    return df
