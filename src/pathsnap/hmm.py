"""HMM + Viterbi route reconstruction (phase 2, decisions D19-D22).

Hidden states : the candidate arcs of each GPS observation (same candidates the classifier scored).
Emission      : w * log P_classifier(candidate is the true arc)
Transition    : -|d_road - d_gps| / beta   (log of Newson & Krumm's exponential)
                d_gps  = straight-line distance between consecutive GPS points
                d_road = shortest DIRECTED road distance from candidate A's projected position to candidate B's:
                         same arc forward          -> (u_b - u_a) * len
                         same arc, small backwards -> (u_b - u_a) * len, i.e. NEGATIVE (GPS jitter, up to
                                                      BACKTRACK_TOL_M); cheap only when the car barely moved
                         otherwise                 -> rest of arc A + shortest path(A.to -> B.from) + start of B
                Transitions longer than d_gps + MAX_DETOUR_M are unreachable.
Segments      : the chain is split (and Viterbi restarted) at time gaps > gap_s, at points without candidates,
                and where no candidate of the next point is reachable from any candidate of the current point.

The expensive part (road distances) does not depend on beta / w / gap_s, so it is computed once per trip
(`transition_costs`) and the cheap Viterbi (`viterbi`) can be re-run for every parameter combination.
"""
import heapq

import numpy as np

BACKTRACK_TOL_M = 15.0
MAX_DETOUR_M = 1000.0
# Each node's bounded Dijkstra is run once with at least this range and cached (D33, speed only). Distances are exact
# shortest paths, and the 'd_road <= limit' check below is unchanged, so results are identical; it only avoids
# re-running the same search whenever a slightly larger range is requested.
SEARCH_RANGE_M = MAX_DETOUR_M + 150.0
PRECOMPUTE_MAX_GAP_S = 120.0  # steps with larger gaps always split, no road distances needed


class DirectedGraph:
    """CSR adjacency of the directed road graph with edge length in metres; bounded Dijkstra with a cache."""

    def __init__(self, nodes_xy, arcs):
        order = np.argsort(arcs[:, 0], kind="stable")
        self.arc_id = order  # CSR edge index -> original arc_id
        self.dst = arcs[order, 1]
        self.w = np.linalg.norm(nodes_xy[arcs[order, 1]] - nodes_xy[arcs[order, 0]], axis=1)
        self.ptr = np.searchsorted(arcs[order, 0], np.arange(len(nodes_xy) + 1))
        self._cache = {}

    def distances_from(self, src, limit):
        hit = self._cache.get(src)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist, prev = {src: 0.0}, {}
        heap = [(0.0, src)]
        ptr, dst, w = self.ptr, self.dst, self.w
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf) or d > limit:
                continue
            for i in range(ptr[n], ptr[n + 1]):
                nd, m = d + w[i], dst[i]
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    prev[m] = i
                    heapq.heappush(heap, (nd, m))
        self._cache[src] = (limit, dist, prev)
        return dist

    def path_arcs(self, src, dst_node, limit):
        """Arc ids of the shortest directed path src -> dst_node (empty if src == dst_node, None if unreachable)."""
        if self.distances_from(src, limit).get(dst_node) is None:
            return None
        prev, out, n = self._cache[src][2], [], dst_node
        while n != src:
            i = prev[n]
            out.append(int(self.arc_id[i]))
            n = int(np.searchsorted(self.ptr, i, side="right") - 1)  # source node of CSR edge i
        return out[::-1]


def transition_costs(pts_xy, dt_prev, cand_groups, graph, stationary=None):
    """cand_groups[i] = dict of arrays (arc, u, arc_len, arc_u, arc_v) for point i (possibly empty).
    Returns list costs[i] (i >= 1): matrix |d_road - d_gps| of shape (n_{i-1}, n_i), np.inf = unreachable,
    or None when the step always splits (no candidates on either side, or gap > PRECOMPUTE_MAX_GAP_S).
    stationary[i] (optional, bool): the vehicle is not really travelling at point i (long-window speed). Then a small
    backward drift along the same arc is GPS jitter, costed as |abs(delta) - d_gps| instead of as reversing."""
    costs = [None]
    for i in range(1, len(cand_groups)):
        A, B = cand_groups[i - 1], cand_groups[i]
        if len(A["arc"]) == 0 or len(B["arc"]) == 0 or dt_prev[i] > PRECOMPUTE_MAX_GAP_S:
            costs.append(None)
            continue
        d_gps = float(np.hypot(*(pts_xy[i] - pts_xy[i - 1])))
        limit = d_gps + MAX_DETOUR_M
        M = np.full((len(A["arc"]), len(B["arc"])), np.inf)
        for a in range(len(A["arc"])):
            rest_a = (1.0 - A["u"][a]) * A["arc_len"][a]
            if rest_a > limit:
                reach = {}
            else:
                reach = graph.distances_from(int(A["arc_v"][a]), max(limit - rest_a, SEARCH_RANGE_M))
            for b in range(len(B["arc"])):
                if A["arc"][a] == B["arc"][b]:
                    delta = (B["u"][b] - A["u"][a]) * A["arc_len"][a]
                    if delta >= -BACKTRACK_TOL_M:
                        drift = abs(delta) if (stationary is not None and stationary[i]) else delta
                        M[a, b] = abs(drift - d_gps)
                        continue
                sp = reach.get(int(B["arc_u"][b]))
                if sp is not None:
                    d_road = rest_a + sp + B["u"][b] * B["arc_len"][b]
                    if d_road <= limit:
                        M[a, b] = abs(d_road - d_gps)
        costs.append(M)
    return costs


def viterbi(log_emit, costs, dt_prev, beta, w, gap_s):
    """log_emit[i]: array of log P for the candidates of point i. Returns (choice index per point, -1 = none;
    segment id per point). Max-product Viterbi in log space, restarted at every split."""
    n = len(log_emit)
    choice = np.full(n, -1)
    seg = np.full(n, -1)
    seg_id, back, start, delta = -1, [], 0, None

    def close(end):  # backtrack the segment [start, end)
        if delta is None:
            return
        k = int(np.argmax(delta))
        for i in range(end - 1, start - 1, -1):
            choice[i] = k
            seg[i] = seg_id
            if i > start:
                k = back[i - start][k]

    for i in range(n):
        e = w * log_emit[i]
        if len(e) == 0:
            close(i)
            delta, back = None, []
            continue
        split = delta is None or costs[i] is None or dt_prev[i] > gap_s
        if not split:
            scores = delta[:, None] - costs[i] / beta  # (n_prev, n_cur)
            best_prev = scores.argmax(0)
            best = scores[best_prev, np.arange(len(e))]
            split = not np.isfinite(best).any()
        if split:
            close(i)
            seg_id += 1
            start, back, delta = i, [None], e.copy()
        else:
            back.append(best_prev)
            delta = best + e
    close(n)
    return choice, seg


if __name__ == "__main__":  # self-check: a two-way street where the classifier prefers the wrong direction
    nodes = np.array([[0, 0], [100, 0], [200, 0.0]])
    arcs = np.array([[0, 1], [1, 0], [1, 2], [2, 1]])        # 0: east, 1: west twin, 2: east, 3: west twin
    g = DirectedGraph(nodes, arcs)
    xs = [10, 25, 40, 55, 70, 85, 110, 125]                 # car drives east at 15 m/s
    pts = np.array([[x, 1.0] for x in xs])
    groups, emit = [], []
    for x in xs:
        a = [0, 1] if x < 100 else [2, 3]
        u = [x / 100 if aa in (0, 2) else 1 - x / 100 for aa in a] if x < 100 else [(x - 100) / 100, 1 - (x - 100) / 100]
        groups.append({"arc": np.array(a), "u": np.array(u), "arc_len": np.array([100.0, 100.0]),
                       "arc_u": arcs[a, 0], "arc_v": arcs[a, 1]})
        emit.append(np.log(np.array([0.45, 0.55])))          # classifier slightly prefers the WRONG (west) twin
    costs = transition_costs(pts, np.ones(len(xs)), groups, g)
    ch, seg = viterbi(emit, costs, np.ones(len(xs)), beta=5.0, w=1.0, gap_s=30)
    picked = [groups[i]["arc"][c] for i, c in enumerate(ch)]
    assert picked == [0, 0, 0, 0, 0, 0, 2, 2], picked        # HMM recovers the eastbound arcs
    assert (seg == 0).all(), seg
    # straight two-way road, no junction: a wrong-direction chain moves 'backwards' and must lose
    nodes2 = np.array([[0, 0], [300, 0.0]])
    arcs2 = np.array([[0, 1], [1, 0]])
    g2 = DirectedGraph(nodes2, arcs2)
    xs2 = np.arange(20, 280, 15.0)
    grp2 = [{"arc": np.array([0, 1]), "u": np.array([x / 300, 1 - x / 300]), "arc_len": np.array([300.0, 300.0]),
             "arc_u": arcs2[[0, 1], 0], "arc_v": arcs2[[0, 1], 1]} for x in xs2]
    em2 = [np.log(np.array([0.45, 0.55]))] * len(xs2)
    c2 = transition_costs(np.c_[xs2, np.ones(len(xs2))], np.ones(len(xs2)), grp2, g2)
    ch2, _ = viterbi(em2, c2, np.ones(len(xs2)), beta=5.0, w=1.0, gap_s=30)
    assert (ch2 == 0).all(), ch2
    # parked car with jitter: tiny backwards moves on the correct arc stay on it
    xs3 = np.array([150, 151, 149.5, 150.5, 149, 150.0])
    grp3 = [{"arc": np.array([0, 1]), "u": np.array([x / 300, 1 - x / 300]), "arc_len": np.array([300.0, 300.0]),
             "arc_u": arcs2[[0, 1], 0], "arc_v": arcs2[[0, 1], 1]} for x in xs3]
    em3 = [np.log(np.array([0.7, 0.3]))] * len(xs3)
    c3 = transition_costs(np.c_[xs3, np.ones(len(xs3))], np.ones(len(xs3)), grp3, g2)
    ch3, seg3 = viterbi(em3, c3, np.ones(len(xs3)), beta=5.0, w=1.0, gap_s=30)
    assert (ch3 == 0).all() and (seg3 == 0).all(), (ch3, seg3)
    print("hmm self-check ok", picked)
