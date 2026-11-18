"""Per-observation labels from the reference route (TRAINING/SCORING ONLY - never used for prediction).

.route is an ordered list of arcs, not one label per GPS point. We find the route position j(i) for each
GPS point i that minimises total alignment cost subject to j never moving backwards (monotonic alignment,
solved exactly by dynamic programming). This handles routes that revisit the same arc/area.

Alignment cost of point i on route arc j (decision D23):
    distance(i, arc j)  +  LABEL_HEADING_PENALTY_M * max(0, -cos(heading_i - bearing_j)) * moving_i
The direction term only matters when the route drives a road both ways: the two directions are exactly
equally close, so distance alone is a tie and could label the outbound leg with the return arc.
It is zero for arcs within 90 degrees of the travel direction (so turns are not penalised) and zero when
the vehicle is stationary (heading is unreliable). label_dist stays the pure distance.
"""
import numpy as np

from . import config as C
from .geo import angle_diff, bearing, point_to_segment


def align_to_route(pts_xy, dt_prev, route_a, route_b, heading=None, moving=None,
                   max_step=C.ALIGN_MAX_STEP, gap_s=C.GAP_S, heading_penalty=C.LABEL_HEADING_PENALTY_M):
    """Returns (route_position per point, distance to that route arc in metres)."""
    n, m = len(pts_xy), len(route_a)
    D = np.empty((n, m), np.float32)
    for s in range(0, n, 2000):  # chunk to bound memory
        D[s:s + 2000] = point_to_segment(pts_xy[s:s + 2000, None, :], route_a[None], route_b[None])[0]
    cost_m = D
    if heading is not None:
        arc_bearing = bearing(route_b[:, 0] - route_a[:, 0], route_b[:, 1] - route_a[:, 1])
        opposing = np.maximum(0.0, -np.cos(angle_diff(heading[:, None], arc_bearing[None, :])))
        cost_m = D + (heading_penalty * opposing * np.asarray(moving)[:, None]).astype(np.float32)

    idx = np.arange(m)
    back = np.empty((n - 1, m), np.int32)
    cost = cost_m[0].astype(np.float64)
    shifted = np.full((max_step + 1, m), np.inf)
    for i in range(1, n):
        if dt_prev[i] > gap_s:
            # after a GPS gap the vehicle may have advanced any distance along the route
            best = np.minimum.accumulate(cost)
            src = np.maximum.accumulate(np.where(cost == best, idx, 0))
        else:
            shifted.fill(np.inf)
            for s in range(max_step + 1):
                shifted[s, s:] = cost[: m - s]
            k = shifted.argmin(0)
            best, src = shifted[k, idx], idx - k
        cost = best + cost_m[i]
        back[i - 1] = src

    pos = np.empty(n, np.int64)
    pos[-1] = int(cost.argmin())
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i - 1, pos[i]]
    return pos, D[np.arange(n), pos].astype(float)


def label_points(pts, trip, nodes_xy):
    """Adds label_arc, label_dist, label_reliable to the points frame."""
    ra = nodes_xy[trip.arcs[trip.route, 0]]
    rb = nodes_xy[trip.arcs[trip.route, 1]]
    pos, dist = align_to_route(pts[["x", "y"]].to_numpy(), pts.dt_prev.to_numpy(), ra, rb,
                               pts.heading.to_numpy(), pts.moving.to_numpy())
    pts = pts.copy()
    pts["label_arc"] = trip.route[pos]
    pts["label_dist"] = dist
    pts["label_reliable"] = dist <= C.LABEL_MAX_DIST_M
    return pts


if __name__ == "__main__":
    # 1) a square driven twice; the second lap must map to route pos 4
    sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100.0]])
    a, b = np.r_[sq, sq[:1]], np.r_[sq[1:], sq[:2]]          # route = arc0, arc1, arc2, arc3, arc0 again
    p = np.array([[30, 2], [102, 40], [60, 98], [-1, 50], [40, -3], [80, 1]], float)
    pos, d = align_to_route(p, np.ones(len(p)), a, b, max_step=2)
    assert pos.tolist() == [0, 1, 2, 3, 4, 4], pos  # nearest-route-arc would wrongly give 0 for the lap

    # 2) out-and-back on one road: route = east arc, then the west twin. Both are equally close to every point,
    #    so distance alone can label the outbound leg 'west'. The direction term must fix that.
    a2, b2 = np.array([[0, 0], [200, 0.0]]), np.array([[200, 0], [0, 0.0]])
    xs = np.r_[np.arange(10, 200, 20.0), np.arange(190, 0, -20.0)]
    p2 = np.c_[xs, np.ones(len(xs))]
    head = np.where(np.arange(len(xs)) < 10, np.pi / 2, -np.pi / 2)  # east then west (compass bearing)
    pos2, _ = align_to_route(p2, np.ones(len(xs)), a2, b2, head, np.ones(len(xs)), max_step=1)
    assert pos2.tolist() == [0] * 10 + [1] * 10, pos2
    print("labels self-check ok", pos, pos2)
