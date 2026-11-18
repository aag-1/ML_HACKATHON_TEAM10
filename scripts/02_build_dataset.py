"""Step 2 - Preprocessing, label alignment, candidate generation and feature engineering.

For every trip in train + student_test:
  load -> UTM projection -> preprocess (gaps, speed, heading, moving) -> DP label alignment
  -> candidate arcs within 50 m (max 10) -> features -> y = (candidate arc == label arc)

Outputs
  data/processed/points_<partition>.parquet      one row per GPS observation (+ label columns)
  data/processed/candidates_<partition>.parquet  one row per (GPS observation, candidate arc) with features + y
  results/dataset_per_trip.csv                   label quality, candidate miss rate, class balance per trip
  results/dataset_summary.json
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from pathsnap import config as C
from pathsnap.data import list_trips, load_trip
from pathsnap.features import FEATURES
from pathsnap.labels import label_points
from pathsnap.pipeline import prepare_trip


def process(job):
    part, tid = job
    trip = load_trip(tid, part, with_route=True)
    pts, cands, nodes_xy = prepare_trip(trip)                   # identical code path to the API
    pts = label_points(pts, trip, nodes_xy)                     # labels: training/scoring only
    cands["y"] = (cands.arc.to_numpy() == pts.label_arc.to_numpy()[cands.pt]).astype(np.int8)

    pts["n_cand"] = np.bincount(cands.pt, minlength=len(pts))
    pts["true_in_cands"] = np.bincount(cands.pt, weights=cands.y, minlength=len(pts)) > 0
    pts["lon"], pts["lat"] = trip.track.lon.to_numpy(), trip.track.lat.to_numpy()
    for df in (pts, cands):
        df.insert(0, "trip", tid)
        df.insert(1, "partition", part)
    pts.insert(2, "pt", np.arange(len(pts)))
    cands["label_reliable"] = pts.label_reliable.to_numpy()[cands.pt]

    rel = pts.label_reliable
    stats = {
        "partition": part, "trip": tid, "n_obs": len(pts),
        "excluded_from_training": tid in C.EXCLUDED_TRAIN_TRIPS,
        "label_dist_median_m": round(float(pts.label_dist.median()), 2),
        "label_dist_p95_m": round(float(pts.label_dist.quantile(.95)), 2),
        "pct_unreliable_labels": round(100 * (1 - rel.mean()), 2),
        "mean_candidates": round(float(pts.n_cand.mean()), 2),
        "pct_no_candidates": round(100 * float((pts.n_cand == 0).mean()), 2),
        "candidate_miss_rate_pct": round(100 * float((~pts.true_in_cands[rel]).mean()), 3),
        "pct_positive_rows": round(100 * float(cands.y.mean()), 2),
        "n_candidate_rows": len(cands),
    }
    return pts, cands, stats


if __name__ == "__main__":
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    jobs = [(p, t) for p in C.PARTITIONS for t in list_trips(p)]
    out = {p: ([], []) for p in C.PARTITIONS}
    stats = []
    with ProcessPoolExecutor() as ex:
        for (part, tid), (pts, cands, st) in zip(jobs, ex.map(process, jobs)):
            out[part][0].append(pts)
            out[part][1].append(cands)
            stats.append(st)
            print(f"{part:12} {tid}  obs={st['n_obs']:6}  cand={st['n_candidate_rows']:7}  "
                  f"miss={st['candidate_miss_rate_pct']:.2f}%  unreliable={st['pct_unreliable_labels']:.2f}%", flush=True)

    summary = {}
    for part, (P, Cd) in out.items():
        P, Cd = pd.concat(P, ignore_index=True), pd.concat(Cd, ignore_index=True)
        P.to_parquet(C.PROCESSED / f"points_{part}.parquet", index=False)
        Cd.to_parquet(C.PROCESSED / f"candidates_{part}.parquet", index=False)
        use = P[P.label_reliable & ~P.trip.isin(C.EXCLUDED_TRAIN_TRIPS if part == "train" else set())]
        summary[part] = {
            "observations": len(P), "candidate_rows": len(Cd), "features": FEATURES,
            "observations_used_for_modelling": len(use),
            "unreliable_label_points": int((~P.label_reliable).sum()),
            "candidate_miss_rate_pct": round(100 * float((~use.true_in_cands).mean()), 3),
            "pct_points_no_candidates": round(100 * float((P.n_cand == 0).mean()), 3),
            "mean_candidates_per_point": round(float(P.n_cand.mean()), 2),
            "pct_positive_rows": round(100 * float(Cd.y.mean()), 2),
        }
    summary["excluded_train_trips"] = sorted(C.EXCLUDED_TRAIN_TRIPS)
    pd.DataFrame(stats).to_csv(C.RESULTS / "dataset_per_trip.csv", index=False)
    (C.RESULTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
