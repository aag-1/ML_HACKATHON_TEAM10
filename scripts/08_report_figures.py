"""Step 8a - Static route maps for the technical report (exported view of the Folium layers).

Runs the frozen pipeline (artifacts/) on two student-test trips and draws, in UTM metres:
road network, reference route, GPS observations, classifier-only prediction and HMM prediction.
Output: figures/report_route_maps.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from pathsnap import config as C
from pathsnap.data import load_trip
from pathsnap.geo import make_projector
from pathsnap.pipeline import predict_trip, score_trip

MODEL = joblib.load(C.ROOT / "artifacts" / "model.joblib")
PARAMS = json.loads((C.ROOT / "artifacts" / "hmm_params.json").read_text())
TRIPS = [("00000040", "typical trip"), ("00000077", "hardest trip (car-park crawling)")]

fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
for ax, (tid, label) in zip(axes, TRIPS):
    trip = load_trip(tid, "student_test")
    out, route, pts, nodes_xy, _ = predict_trip(trip, MODEL, PARAMS)
    ref = load_trip(tid, "student_test", with_route=True)
    score = score_trip(out, pts, nodes_xy, ref)
    xy = pts[["x", "y"]].to_numpy()
    seg = np.stack([nodes_xy[trip.arcs[:, 0]], nodes_xy[trip.arcs[:, 1]]], 1)
    lo, hi = xy.min(0) - 250, xy.max(0) + 250
    near = np.all((seg.min(1) >= lo) & (seg.max(1) <= hi), 1)
    clf = out.classifier_arc.to_numpy()
    clf_arcs = sorted(set(int(a) for a in clf if a >= 0))
    ax.add_collection(LineCollection(seg[near], colors="#cfcfcf", linewidths=.5, label="road network"))
    ax.add_collection(LineCollection(seg[ref.route], colors="#1f9d55", linewidths=6, alpha=.55, label="reference route"))
    ax.add_collection(LineCollection(seg[clf_arcs], colors="#e4572e", linewidths=1.6, linestyles="dashed",
                                     label="classifier-only arcs"))
    ax.add_collection(LineCollection(seg[route], colors="#7b2cbf", linewidths=2.2, label="classifier + HMM route", zorder=6))
    ax.scatter(*xy.T, s=1.5, color="#1d4ed8", alpha=.35, label="GPS observations", zorder=4)
    ax.locator_params(axis="x", nbins=4)
    ax.set(title=f"Trip {tid} ({label}): Top-1 classifier {score['classifier_top1']:.1%} -> HMM {score['hmm_top1']:.1%}",
           xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal", xlabel="UTM easting (m)", ylabel="UTM northing (m)")
    ax.title.set_fontsize(10)
    ax.legend(loc="best", fontsize=8)
fig.suptitle("Student-test partition: predicted vs reference routes (exported view of the Folium map layers)")
fig.tight_layout()
fig.savefig(C.FIGURES / "report_route_maps.png", dpi=140)
print("wrote figures/report_route_maps.png")
