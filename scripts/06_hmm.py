"""Step 6 - HMM + Viterbi route reconstruction: decoder tuning (train) and evaluation (student-test).

Tuning (D22): out-of-fold probabilities of the tuned HGB on the 78 train trips (same 5 GroupKFold folds:
the classifier never scores a trip it was trained on) -> grid search over beta, w, gap_s -> best pooled Top-1.
Transition road distances do not depend on the grid parameters, so they are computed once per trip.

Evaluation: tuned HGB fitted on all train trips (models/HistGradientBoosting_tuned.joblib) -> student-test,
classifier-only vs classifier + HMM, reported once.

Outputs
  results/hmm_grid_train.csv       Top-1 on train (OOF) for every (beta, w, gap_s)
  results/hmm_params.json          chosen decoder parameters
  results/hmm_student_test.csv     per trip: classifier-only vs HMM (Top-1, continuity, route P/R/F1, segments)
  results/hmm_summary.json         pooled student-test results
  figures/hmm_*.png, maps/hmm_trip_<id>.html
"""
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap import config as C
from pathsnap.data import load_trip
from pathsnap.dataset import load
from pathsnap.features import FEATURES
from pathsnap.geo import make_projector
from pathsnap.hmm import DirectedGraph, transition_costs, viterbi
from pathsnap.pipeline import (classifier_only, decode, decoder_inputs, group_candidates, reconstruct_route,
                               route_continuity, route_overlap)
from pathsnap.viz import route_map

# Only the product beta*w matters to Viterbi (rescaling both leaves the argmax unchanged), so w is fixed at 1.
GRID = {"beta": [2, 3, 5, 8], "w": [1.0], "gap_s": [30, 60, 120], "v_stationary": [None, 4, 8, 12], "d_off": [None, 15, 20]}
KEYS = list(GRID)
COMBOS = [dict(zip(KEYS, v)) for v in itertools.product(*GRID.values())]
key = lambda c: tuple(c[k] for k in KEYS)


def trip_context(part, tid, pts, cands, proba):
    trip = load_trip(tid, part, with_route=True)
    project = make_projector(trip.track)
    nodes_xy = project(trip.nodes[:, 0], trip.nodes[:, 1])
    groups, log_emit = group_candidates(len(pts), cands, proba)
    graph = DirectedGraph(nodes_xy, trip.arcs)
    xy, dt = pts[["x", "y"]].to_numpy(), pts.dt_prev.to_numpy()
    return trip, nodes_xy, groups, log_emit, graph, xy, dt


def grid_worker(job):
    """One train trip: costs once, then Viterbi for every grid combination. Returns correct counts."""
    tid, pts, cands, proba = job
    trip, nodes_xy, groups, log_emit, graph, xy, dt = trip_context("train", tid, pts, cands, proba)
    rel, lab = pts.label_reliable.to_numpy(), pts.label_arc.to_numpy()
    clf = classifier_only(groups, log_emit)
    out = {"trip": tid, "n": int(rel.sum()), "clf_correct": int((clf[rel] == lab[rel]).sum())}
    cost_cache = {}
    for c in COMBOS:
        stationary, scale = decoder_inputs(pts, groups, c)
        if c["v_stationary"] not in cost_cache:  # road distances depend only on the stationary flag
            cost_cache[c["v_stationary"]] = transition_costs(xy, dt, groups, graph, stationary)
        choice, _ = viterbi([e * k for e, k in zip(log_emit, scale)], cost_cache[c["v_stationary"]], dt,
                            c["beta"], c["w"], c["gap_s"])
        pred = np.array([g["arc"][j] if j >= 0 else -1 for g, j in zip(groups, choice)])
        out[key(c)] = int((pred[rel] == lab[rel]).sum())
    return out


def test_worker(job):
    """One student-test trip with the chosen parameters: classifier-only vs HMM, full metrics, route, map."""
    tid, pts, cands, proba, params = job
    t0 = time.perf_counter()
    trip, nodes_xy, groups, log_emit, graph, xy, dt = trip_context("student_test", tid, pts, cands, proba)
    hmm, seg, _ = decode(pts, groups, log_emit, graph, params)  # exactly the API's decoder
    decode_s = time.perf_counter() - t0
    clf = classifier_only(groups, log_emit)
    rel, lab = pts.label_reliable.to_numpy(), pts.label_arc.to_numpy()
    hmm_route = reconstruct_route(hmm, seg, trip.arcs, graph)
    clf_route = [int(a) for i, a in enumerate(clf) if a >= 0 and (i == 0 or a != clf[i - 1])]
    hp, hr, hf = route_overlap(hmm_route, trip.route, trip.arcs, nodes_xy)
    cp, cr, cf = route_overlap(clf_route, trip.route, trip.arcs, nodes_xy)
    m = route_map(trip, f"Trip {tid} (student-test)", [
        ("Reference route (.route)", list(trip.route), "#1f9d55", 7),
        ("Classifier-only prediction", clf_route, "#e4572e", 3),
        ("Classifier + HMM/Viterbi prediction", hmm_route, "#7b2cbf", 4),
    ], note=f"Top-1: classifier {np.mean(clf[rel] == lab[rel]):.1%} &#8594; HMM {np.mean(hmm[rel] == lab[rel]):.1%}<br>")
    m.save(C.MAPS / f"hmm_trip_{tid}.html")
    return {
        "trip": tid, "n_points": int(rel.sum()),
        "clf_top1": float(np.mean(clf[rel] == lab[rel])), "hmm_top1": float(np.mean(hmm[rel] == lab[rel])),
        "clf_correct": int((clf[rel] == lab[rel]).sum()), "hmm_correct": int((hmm[rel] == lab[rel]).sum()),
        "clf_continuity": route_continuity(clf, trip.arcs), "hmm_continuity": route_continuity(hmm, trip.arcs),
        "clf_route_precision": cp, "clf_route_recall": cr, "clf_route_f1": cf,
        "hmm_route_precision": hp, "hmm_route_recall": hr, "hmm_route_f1": hf,
        "hmm_segments": int(seg.max() + 1), "decode_s": decode_s, "hmm_ms_per_point": 1000 * decode_s / len(pts),
    }


def full_trip_rows(part, trips):
    """All candidate rows / points of the given trips (unfiltered: the decoder needs every point of a trip)."""
    cand = pd.read_parquet(C.PROCESSED / f"candidates_{part}.parquet")
    pts = pd.read_parquet(C.PROCESSED / f"points_{part}.parquet")
    return cand[cand.trip.isin(trips)], pts[pts.trip.isin(trips)]


def main():
    best = json.loads((C.RESULTS / "bayes_opt_best.json").read_text())
    hgb_params = {**best["fixed_params"], **best["best_params"]}

    # ---- out-of-fold probabilities on the train trips (same folds as phase 1 / Bayesian optimisation)
    tr_c, _ = load("train")
    X, y, g = tr_c[FEATURES].to_numpy(), tr_c.y.to_numpy(), tr_c.trip.to_numpy()
    all_c, all_p = full_trip_rows("train", set(g))
    oof = np.full(len(all_c), np.nan)
    for a, b in GroupKFold(n_splits=C.N_FOLDS).split(X, y, g):
        m = HistGradientBoostingClassifier(**hgb_params).fit(X[a], y[a])
        rows = all_c.trip.isin(set(g[b])).to_numpy()
        oof[rows] = m.predict_proba(all_c.loc[rows, FEATURES].to_numpy())[:, 1]
    assert not np.isnan(oof).any()
    all_c = all_c.assign(proba=oof)

    jobs = [(tid, all_p[all_p.trip == tid].reset_index(drop=True), c.reset_index(drop=True), c.proba.to_numpy())
            for tid, c in all_c.groupby("trip", sort=True)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(grid_worker, jobs))
    grid_s = time.perf_counter() - t0
    n = sum(r["n"] for r in res)
    grid = pd.DataFrame([{**c, "train_top1": sum(r[key(c)] for r in res) / n} for c in COMBOS]
                        ).sort_values("train_top1", ascending=False)
    grid.to_csv(C.RESULTS / "hmm_grid_train.csv", index=False)
    top = grid.iloc[0]
    params = {k: (None if pd.isna(top[k]) else float(top[k])) for k in KEYS}
    train_clf = sum(r["clf_correct"] for r in res) / n
    (C.RESULTS / "hmm_params.json").write_text(json.dumps({
        **params, "selected_by": "max pooled Top-1 on 78 train trips with out-of-fold HGB probabilities",
        "train_top1_hmm": float(top.train_top1), "train_top1_classifier_only": train_clf,
        "grid": GRID, "grid_seconds": grid_s}, indent=2))
    print(f"train (OOF): classifier-only {train_clf:.4f} -> HMM {top.train_top1:.4f} with {params}", flush=True)

    # ---- student-test, once
    model = joblib.load(C.ROOT / "models" / "HistGradientBoosting_tuned.joblib")
    te_c, te_p = full_trip_rows("student_test", set(pd.read_parquet(C.PROCESSED / "points_student_test.parquet").trip))
    te_c = te_c.assign(proba=model.predict_proba(te_c[FEATURES].to_numpy())[:, 1])
    jobs = [(tid, te_p[te_p.trip == tid].reset_index(drop=True), c.reset_index(drop=True), c.proba.to_numpy(), params)
            for tid, c in te_c.groupby("trip", sort=True)]
    with ProcessPoolExecutor() as ex:
        per_trip = pd.DataFrame(list(ex.map(test_worker, jobs)))
    per_trip["delta_top1"] = per_trip.hmm_top1 - per_trip.clf_top1
    per_trip.to_csv(C.RESULTS / "hmm_student_test.csv", index=False)
    N = per_trip.n_points.sum()
    w_ = per_trip.n_points / N
    summary = {
        "params": params, "student_test_points": int(N),
        "classifier_only_top1": float(per_trip.clf_correct.sum() / N), "hmm_top1": float(per_trip.hmm_correct.sum() / N),
        "classifier_only_continuity": float((per_trip.clf_continuity * w_).sum()),
        "hmm_continuity": float((per_trip.hmm_continuity * w_).sum()),
        "classifier_only_route_f1_mean": float(per_trip.clf_route_f1.mean()), "hmm_route_f1_mean": float(per_trip.hmm_route_f1.mean()),
        "hmm_route_precision_mean": float(per_trip.hmm_route_precision.mean()),
        "hmm_route_recall_mean": float(per_trip.hmm_route_recall.mean()),
        "trips_where_hmm_did_not_help": per_trip[per_trip.delta_top1 <= 0].trip.tolist(),
        "hmm_decode_ms_per_point_mean": float(per_trip.hmm_ms_per_point.mean()),
    }
    (C.RESULTS / "hmm_summary.json").write_text(json.dumps(summary, indent=2))
    figures(grid, per_trip, params)
    print(per_trip.round(4).to_string())
    print(json.dumps(summary, indent=2))


def figures(grid, per_trip, params):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(per_trip))
    ax[0].bar(x - .2, per_trip.clf_top1, .4, label="classifier only", color="#e4572e")
    ax[0].bar(x + .2, per_trip.hmm_top1, .4, label="classifier + HMM/Viterbi", color="#7b2cbf")
    ax[0].set_xticks(x, per_trip.trip.str[-2:])
    ax[0].set(title="Student-test Top-1 per trip", xlabel="trip (last 2 digits)", ylim=(0.6, 1.0))
    ax[0].legend(loc="lower left")
    ax[0].grid(axis="y", alpha=.3)
    sub = grid[(grid.gap_s == params["gap_s"]) & (grid.beta == params["beta"])].copy()
    sub["v_stationary"] = sub.v_stationary.fillna(0).astype(int).astype(str).replace("0", "off")
    sub["d_off"] = sub.d_off.fillna(0).astype(int).astype(str).replace("0", "off")
    g = sub.pivot(index="d_off", columns="v_stationary", values="train_top1")
    im = ax[1].imshow(g.values, cmap="viridis", aspect="auto")
    ax[1].set_xticks(range(len(g.columns)), g.columns)
    ax[1].set_yticks(range(len(g.index)), g.index)
    for (i, j), v in np.ndenumerate(g.values):
        ax[1].text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="white" if v < g.values.mean() else "black")
    ax[1].set(title=f"Decoder grid on train (OOF), beta={params['beta']:.0f}, gap_s={params['gap_s']:.0f}: Top-1",
              xlabel="v_stationary (m/s)", ylabel="d_off (m)")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(C.FIGURES / "hmm_results.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
