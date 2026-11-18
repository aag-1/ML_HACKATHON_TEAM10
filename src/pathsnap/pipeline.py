"""End-to-end inference pipeline shared by the evaluation scripts and the FastAPI backend.

    trip (no route) -> project -> preprocess -> candidates -> features -> classifier P -> HMM/Viterbi
                    -> predicted arc per GPS point -> reconstructed route (gaps filled with shortest paths)

Nothing on the prediction path reads the reference route; scoring against it lives in `score_trip`.
"""
import numpy as np
import pandas as pd

from .candidates import build_index, generate
from .features import FEATURES, add_features
from .geo import make_projector
from .hmm import SEARCH_RANGE_M, DirectedGraph, transition_costs, viterbi
from .preprocess import long_window_speed, preprocess_track

EPS = 1e-6


def prepare_trip(trip):
    """GPS + road graph -> per-point frame, candidate frame with features, projected nodes (same code as training)."""
    project = make_projector(trip.track)
    nodes_xy = project(trip.nodes[:, 0], trip.nodes[:, 1])
    pts = preprocess_track(trip.track, project)
    xy = pts[["x", "y"]].to_numpy()
    cands = generate(xy, nodes_xy, trip.arcs, build_index(nodes_xy, trip.arcs, xy))
    cands = add_features(cands, pts, nodes_xy, trip.arcs)
    return pts, cands, nodes_xy


def group_candidates(n_points, cands, proba):
    """Per-point candidate arrays (cands must be sorted by pt) and log-probabilities."""
    counts = np.bincount(cands.pt.to_numpy(), minlength=n_points)
    cuts = np.cumsum(counts)[:-1]
    cols = {c: np.split(cands[c].to_numpy(), cuts) for c in ("arc", "u", "arc_len", "arc_u", "arc_v", "dist")}
    groups = [{c: cols[c][i] for c in cols} for i in range(n_points)]
    log_emit = np.split(np.log(np.clip(proba, EPS, 1.0)), cuts)
    return groups, log_emit


def classifier_only(groups, log_emit):
    return np.array([g["arc"][int(np.argmax(e))] if len(e) else -1 for g, e in zip(groups, log_emit)])


OFF_ROAD_MAX_SPEED_MS = 3.0  # off-road tempering applies only to slow points (car parks, petrol stations)
OFF_ROAD_WEIGHT = 0.1


def decoder_inputs(pts, groups, params):
    """Decoder options (D30), all tuned on train:
    v_stationary - long-window speed below which backward drift along an arc is jitter, not reversing (None = off)
    d_off        - if every candidate is farther than d_off metres and the car is slow, the point is off the mapped
                   roads (car park): its classifier evidence is down-weighted to OFF_ROAD_WEIGHT (None = off)"""
    vl = long_window_speed(pts)
    stationary = None if params.get("v_stationary") is None else vl < params["v_stationary"]
    scale = np.ones(len(pts))
    if params.get("d_off") is not None:
        dmin = np.array([g["dist"].min() if len(g["dist"]) else np.inf for g in groups])
        scale = np.where((dmin > params["d_off"]) & (vl < OFF_ROAD_MAX_SPEED_MS), OFF_ROAD_WEIGHT, 1.0)
    return stationary, scale


def decode(pts, groups, log_emit, graph, params, costs=None):
    """Returns predicted arc per point (-1 = no candidates), segment id per point, and the transition costs."""
    xy, dt = pts[["x", "y"]].to_numpy(), pts.dt_prev.to_numpy()
    stationary, scale = decoder_inputs(pts, groups, params)
    if costs is None:
        costs = transition_costs(xy, dt, groups, graph, stationary)
    choice, seg = viterbi([e * k for e, k in zip(log_emit, scale)], costs, dt, params["beta"], params["w"], params["gap_s"])
    pred = np.array([g["arc"][c] if c >= 0 else -1 for g, c in zip(groups, choice)])
    return pred, seg, costs


def reconstruct_route(pred, seg, arcs, graph):
    """Ordered arc list: consecutive distinct predicted arcs, joined by the shortest directed path inside a segment."""
    route = []
    for i in np.where(pred >= 0)[0]:
        a = int(pred[i])
        if route and route[-1] == a:
            last = i
            continue
        if route and seg[i] == seg[last]:
            src, dst = int(arcs[route[-1], 1]), int(arcs[a, 0])
            # small search first (reuses the decoder's cached searches); widen only if needed
            path = graph.path_arcs(src, dst, SEARCH_RANGE_M) or graph.path_arcs(src, dst, 3000.0)
            if path:
                route.extend(path)
        route.append(a)
        last = i
    return route


def predict_trip(trip, model, params):
    """Full prediction for one trip (the trip must be loaded WITHOUT its route).
    Returns per-point DataFrame, reconstructed route (arc ids), preprocessed points, projected nodes, stage timings."""
    import time
    assert trip.route is None, "prediction must not see the reference route"
    t = [time.perf_counter()]
    pts, cands, nodes_xy = prepare_trip(trip)
    t.append(time.perf_counter())
    proba = model.predict_proba(cands[FEATURES].to_numpy())[:, 1] if len(cands) else np.empty(0)
    t.append(time.perf_counter())
    groups, log_emit = group_candidates(len(pts), cands, proba)
    graph = DirectedGraph(nodes_xy, trip.arcs)
    pred, seg, _ = decode(pts, groups, log_emit, graph, params)
    route = reconstruct_route(pred, seg, trip.arcs, graph)
    t.append(time.perf_counter())
    out = pd.DataFrame({"pt": np.arange(len(pts)), "t": pts.t, "lon": trip.track.lon, "lat": trip.track.lat,
                        "pred_arc": pred, "classifier_arc": classifier_only(groups, log_emit), "segment": seg,
                        "n_candidates": [len(g["arc"]) for g in groups]})
    timings = {"features_ms": 1000 * (t[1] - t[0]), "classifier_ms": 1000 * (t[2] - t[1]), "hmm_ms": 1000 * (t[3] - t[2])}
    return out, route, pts, nodes_xy, timings


def score_trip(out, pts, nodes_xy, trip_with_route):
    """SERVER-SIDE scoring against the reference route (never called by the prediction path).
    Labels use the same alignment as training; only points with reliable labels are scored."""
    from .labels import label_points
    lab = label_points(pts, trip_with_route, nodes_xy)
    rel = lab.label_reliable.to_numpy()
    y = lab.label_arc.to_numpy()[rel]
    return {"scored_points": int(rel.sum()),
            "hmm_top1": float(np.mean(out.pred_arc.to_numpy()[rel] == y)),
            "classifier_top1": float(np.mean(out.classifier_arc.to_numpy()[rel] == y))}


def route_continuity(pred_arcs, arcs):
    """Share of consecutive predicted arcs (points with a prediction) that are the same arc or directly connected."""
    p = pred_arcs[pred_arcs >= 0]
    if len(p) < 2:
        return float("nan")
    ok = (p[1:] == p[:-1]) | (arcs[p[:-1], 1] == arcs[p[1:], 0])
    return float(ok.mean())


def route_overlap(pred_route, ref_route, arcs, nodes_xy):
    """Length-weighted precision / recall / F1 of the reconstructed route's arc set vs the reference arc set."""
    length = lambda s: float(np.linalg.norm(nodes_xy[arcs[list(s), 1]] - nodes_xy[arcs[list(s), 0]], axis=1).sum()) if s else 0.0
    P, R = set(pred_route), set(int(a) for a in ref_route)
    inter = length(P & R)
    prec = inter / length(P) if P else 0.0
    rec = inter / length(R) if R else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)
