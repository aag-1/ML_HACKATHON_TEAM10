"""PathSnap FastAPI backend - runs on laptop 1.

Start (from the repository root):
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Evaluation trips are read from PATHSNAP_DATA_DIR (default: data/raw/evaluator/<trip_id>/<trip_id>.{track,nodes,arcs,route}).

Endpoints
  GET  /health            model + decoder loaded, number of trips available
  GET  /trips             trip IDs found in the evaluation folder (no hardcoded IDs)
  POST /predict           {"trip_id": "..."} -> full pipeline on the server, server-side accuracy, predicted route
  GET  /summary           average Top-1 over all evaluation trips. Trips not yet run in this API session are run
                          through the full live pipeline (each logged); results of /predict calls are reused.

Integrity: the prediction pipeline loads the trip WITHOUT its .route file. The reference route is loaded only
afterwards, inside score_trip(), and only the resulting accuracy numbers are returned - never the route itself.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.data import load_trip
from pathsnap.pipeline import predict_trip, route_continuity, score_trip

DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", ROOT / "data" / "raw" / "evaluator")).resolve()
ARTIFACTS = ROOT / "artifacts"
(ROOT / "logs").mkdir(exist_ok=True)

log = logging.getLogger("pathsnap")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(ROOT / "logs" / "api.log")):
    h.setFormatter(_fmt)
    log.addHandler(h)

MODEL = joblib.load(ARTIFACTS / "model.joblib")
PARAMS = json.loads((ARTIFACTS / "hmm_params.json").read_text())
# log the data folder relative to the project, never the absolute path (it contains the OS user name)
DATA_DIR_SHOWN = DATA_DIR.relative_to(ROOT) if DATA_DIR.is_relative_to(ROOT) else DATA_DIR.name
log.info("loaded model=%s hmm=%s data_dir=%s", type(MODEL).__name__, PARAMS, DATA_DIR_SHOWN)

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


SESSION_METRICS = {}  # trip_id -> metrics of the last live run in this API process (accuracy numbers only)


def available_trips():
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.iterdir() if (p / f"{p.name}.track").is_file())


def polylines(arc_ids, arcs, nodes):
    """Arc sequence -> list of [lat, lon] polylines, split wherever consecutive arcs are not connected."""
    lines = []
    for a in arc_ids:
        u, v = arcs[a]
        if lines and lines[-1][-1] == [round(nodes[u, 1], 6), round(nodes[u, 0], 6)]:
            lines[-1].append([round(nodes[v, 1], 6), round(nodes[v, 0], 6)])
        else:
            lines.append([[round(nodes[u, 1], 6), round(nodes[u, 0], 6)], [round(nodes[v, 1], 6), round(nodes[v, 0], 6)]])
    return lines


@app.get("/health")
def health():
    return {"status": "ok", "model": type(MODEL).__name__, "hmm_params": PARAMS, "trips_available": len(available_trips())}


@app.get("/trips")
def trips():
    return {"trips": available_trips()}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id.strip()
    if tid not in available_trips():  # also blocks path traversal: only folder names found on disk are accepted
        raise HTTPException(status_code=404, detail=f"unknown trip_id {tid!r}")
    client = request.client.host if request.client else "?"
    return run_trip(tid, client)


@app.get("/summary")
def summary(request: Request):
    client = request.client.host if request.client else "?"
    trips = available_trips()
    for tid in trips:
        if tid not in SESSION_METRICS:
            run_trip(tid, client, via="summary")
    rows = [{"trip_id": t, **{k: SESSION_METRICS[t][k] for k in ("hmm_top1", "classifier_top1", "scored_points")}}
            for t in trips]
    n = sum(r["scored_points"] for r in rows)
    out = {
        "trips": len(rows),
        "mean_hmm_top1": float(np.mean([r["hmm_top1"] for r in rows])) if rows else None,
        "mean_classifier_top1": float(np.mean([r["classifier_top1"] for r in rows])) if rows else None,
        "pooled_hmm_top1": sum(r["hmm_top1"] * r["scored_points"] for r in rows) / n if n else None,
        "pooled_classifier_top1": sum(r["classifier_top1"] * r["scored_points"] for r in rows) / n if n else None,
        "scored_points": n, "per_trip": rows,
    }
    log.info("SUMMARY client=%s trips=%d mean_top1_hmm=%.2f%% pooled_top1_hmm=%.2f%%", client, out["trips"],
             100 * (out["mean_hmm_top1"] or 0), 100 * (out["pooled_hmm_top1"] or 0))
    return out


def run_trip(tid, client, via="predict"):
    t0 = time.perf_counter()
    trip = load_trip(tid, DATA_DIR.name, with_route=False, root=DATA_DIR.parent)
    out, route, pts, nodes_xy, timings = predict_trip(trip, MODEL, PARAMS)

    t1 = time.perf_counter()
    reference = load_trip(tid, DATA_DIR.name, with_route=True, root=DATA_DIR.parent)  # server-side only
    score = score_trip(out, pts, nodes_xy, reference)
    timings["scoring_ms"] = 1000 * (time.perf_counter() - t1)
    timings["total_ms"] = 1000 * (time.perf_counter() - t0)

    clf = out.classifier_arc.to_numpy()
    clf_route = [int(a) for i, a in enumerate(clf) if a >= 0 and (i == 0 or a != clf[i - 1])]
    metrics = {
        **score,
        "hmm_route_continuity": route_continuity(out.pred_arc.to_numpy(), trip.arcs),
        "classifier_route_continuity": route_continuity(clf, trip.arcs),
        "points": len(out), "points_without_candidates": int((out.n_candidates == 0).sum()),
        "mean_candidates": float(out.n_candidates.mean()), "segments": int(out.segment.max() + 1),
        "route_arcs": len(route),
    }
    SESSION_METRICS[tid] = metrics
    log.info("%s client=%s trip=%s points=%d candidates=%.1f segments=%d top1_hmm=%.2f%% top1_classifier=%.2f%% "
             "features=%.0fms classifier=%.0fms hmm=%.0fms scoring=%.0fms total=%.0fms",
             via.upper(), client, tid, metrics["points"], metrics["mean_candidates"], metrics["segments"], 100 * score["hmm_top1"],
             100 * score["classifier_top1"], timings["features_ms"], timings["classifier_ms"], timings["hmm_ms"],
             timings["scoring_ms"], timings["total_ms"])
    return {
        "trip_id": tid, "metrics": metrics, "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "gps": np.round(out[["lat", "lon"]].to_numpy(), 6).tolist(),
        "predicted_route": polylines(route, trip.arcs, trip.nodes),
        "classifier_route": polylines(clf_route, trip.arcs, trip.nodes),
    }
