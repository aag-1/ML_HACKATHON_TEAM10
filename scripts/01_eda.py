"""Step 1 - Dataset exploration (train + student_test partitions; evaluator partition is not available/used).

Outputs
  results/eda_inventory.csv      one row per trip: files, counts, missing values, sampling, network stats
  results/eda_summary.json       partition-level summary numbers
  figures/eda_*.png              plots
  maps/eda_*.html                interactive Folium maps
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap import config as C
from pathsnap.data import _read, list_trips, load_trip
from pathsnap.geo import make_projector
from pathsnap.preprocess import preprocess_track

for d in (C.RESULTS, C.FIGURES, C.MAPS):
    d.mkdir(exist_ok=True)


def inventory_row(part, tid):
    d = C.RAW / part / tid
    raw_track = _read(d / f"{tid}.track", ["lon", "lat", "t"], float)
    tr = load_trip(tid, part, with_route=True)
    proj = make_projector(tr.track)
    pts = preprocess_track(tr.track, proj)
    A, N = tr.arcs, len(tr.nodes)
    g = coo_matrix((np.ones(len(A)), (A[:, 0], A[:, 1])), shape=(N, N)).tocsr()
    n_wcc, wl = connected_components(g, directed=True, connection="weak")
    n_scc, sl = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(A.ravel(), minlength=N)
    arcset = set(map(tuple, A))
    one_way = np.mean([(v, u) not in arcset for u, v in A])
    r = tr.route
    route_breaks = int(np.sum(A[r[:-1], 1] != A[r[1:], 0]))
    dt = pts.dt_prev.to_numpy()[1:]
    xy = pts[["x", "y"]].to_numpy()
    return {
        "partition": part, "trip": tid,
        "track_kb": round((d / f"{tid}.track").stat().st_size / 1024, 1),
        "nodes_kb": round((d / f"{tid}.nodes").stat().st_size / 1024, 1),
        "arcs_kb": round((d / f"{tid}.arcs").stat().st_size / 1024, 1),
        "n_obs": len(tr.track), "missing_values": int(raw_track.isna().sum().sum()),
        "timestamps_sorted": bool(np.all(np.diff(raw_track.t) > 0)),
        "duration_min": round((tr.track.t.iloc[-1] - tr.track.t.iloc[0]) / 60, 1),
        "pct_dt_1s": round(100 * np.mean(dt == 1), 1), "pct_dt_2s": round(100 * np.mean(dt == 2), 1),
        "max_gap_s": float(dt.max()), "n_gaps_gt5s": int(np.sum(dt > C.GAP_S)),
        "n_duplicate_pts": int(pts.is_duplicate.sum()), "pct_stationary": round(100 * (1 - pts.moving.mean()), 1),
        "max_speed_ms_raw": round(float(np.max(np.hypot(*np.diff(xy, axis=0).T) / np.maximum(dt, 1e-9))), 1),
        "median_speed_kmh": round(3.6 * float(pts.speed[pts.moving == 1].median()), 1),
        "path_km": round(float(np.hypot(*np.diff(xy, axis=0).T).sum()) / 1000, 2),
        "extent_km": round(float(np.ptp(xy, axis=0).max()) / 1000, 2),
        "lon": round(float(tr.track.lon.median()), 3), "lat": round(float(tr.track.lat.median()), 3),
        "n_nodes": N, "n_arcs": len(A), "isolated_nodes": int(np.sum(deg == 0)),
        "self_loops": int(np.sum(A[:, 0] == A[:, 1])), "duplicate_arcs": len(A) - len(arcset),
        "pct_one_way": round(100 * one_way, 1),
        "n_weak_components": int(n_wcc), "largest_wcc_pct": round(100 * np.bincount(wl).max() / N, 1),
        "n_strong_components": int(n_scc), "largest_scc_pct": round(100 * np.bincount(sl).max() / N, 1),
        "n_route_arcs": len(r), "route_breaks": route_breaks, "route_repeated_arcs": len(r) - len(set(r)),
    }


def plots(inv):
    colors = {"train": "#3b6fb6", "student_test": "#d9822b"}
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    for part, g in inv.groupby("partition"):
        ax[0, 0].hist(g.n_obs, bins=30, alpha=.7, label=part, color=colors[part])
        ax[0, 1].scatter(g.n_nodes, g.n_arcs, label=part, color=colors[part], s=18)
        ax[1, 0].scatter(g.lon, g.lat, label=part, color=colors[part], s=18)
        ax[1, 1].hist(g.pct_dt_1s, bins=20, alpha=.7, label=part, color=colors[part])
        ax[1, 2].hist(g.pct_one_way, bins=20, alpha=.7, label=part, color=colors[part])
    ax[0, 0].set(title="GPS observations per trip", xlabel="observations", ylabel="trips")
    ax[0, 1].set(title="Road-network size per trip", xlabel="nodes", ylabel="arcs", xscale="log", yscale="log")
    ax[1, 0].set(title="Trip locations (worldwide)", xlabel="longitude", ylabel="latitude", xlim=(-180, 180), ylim=(-60, 75))
    ax[1, 1].set(title="% of time steps exactly 1 s", xlabel="% of steps", ylabel="trips")
    ax[1, 2].set(title="% one-way arcs in network", xlabel="% arcs without reverse twin", ylabel="trips")
    s = inv.sort_values("max_gap_s", ascending=False).head(12)
    ax[0, 2].barh(s.trip, s.max_gap_s / 60, color=[colors[p] for p in s.partition])
    ax[0, 2].set(title="Largest time gap (top 12 trips)", xlabel="minutes", xscale="log")
    for a in ax.ravel():
        a.grid(alpha=.3)
    ax[0, 0].legend()
    fig.suptitle("PathSnap EDA - train (80 trips) and student_test (10 trips)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_overview.png", dpi=130)
    plt.close(fig)


def trajectory_plot(trips):
    fig, ax = plt.subplots(1, len(trips), figsize=(5.5 * len(trips), 5.5))
    for a, (part, tid) in zip(ax, trips):
        tr = load_trip(tid, part, with_route=True)
        proj = make_projector(tr.track)
        xy, nxy = proj(tr.track.lon, tr.track.lat), proj(tr.nodes[:, 0], tr.nodes[:, 1])
        lo, hi = xy.min(0) - 300, xy.max(0) + 300
        seg = np.stack([nxy[tr.arcs[:, 0]], nxy[tr.arcs[:, 1]]], 1)
        inside = np.all((seg.min(1) >= lo) & (seg.max(1) <= hi), 1)
        from matplotlib.collections import LineCollection
        a.add_collection(LineCollection(seg[inside], colors="#c8c8c8", linewidths=.5, label="road network"))
        rs = seg[tr.route]
        a.add_collection(LineCollection(rs, colors="#2a9d4b", linewidths=2.2, label="reference route", zorder=4))
        a.scatter(*xy.T, s=3, color="#2b5fd9", alpha=.45, label="GPS observations", zorder=3)
        a.set(title=f"trip {tid} ({part})", xlabel="UTM easting (m)", ylabel="UTM northing (m)",
              xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect="equal")
        a.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "eda_trajectories.png", dpi=130)
    plt.close(fig)


def folium_trip_map(part, tid, note=""):
    tr = load_trip(tid, part, with_route=True)
    lat, lon = tr.track.lat.to_numpy(), tr.track.lon.to_numpy()
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    pad = 0.003
    nd = tr.nodes
    a, b = nd[tr.arcs[:, 0]], nd[tr.arcs[:, 1]]
    inside = ((np.minimum(a[:, 0], b[:, 0]) >= lon.min() - pad) & (np.maximum(a[:, 0], b[:, 0]) <= lon.max() + pad)
              & (np.minimum(a[:, 1], b[:, 1]) >= lat.min() - pad) & (np.maximum(a[:, 1], b[:, 1]) <= lat.max() + pad))
    net = folium.FeatureGroup(name="Road network (arcs near trip)", show=True)
    lines = [[[a[i, 1], a[i, 0]], [b[i, 1], b[i, 0]]] for i in np.where(inside)[0]]
    folium.PolyLine(lines, color="#9a9a9a", weight=1, opacity=.6).add_to(net)
    net.add_to(m)
    route = folium.FeatureGroup(name="Reference route (.route)", show=True)
    rl = [[[a[i, 1], a[i, 0]], [b[i, 1], b[i, 0]]] for i in tr.route]
    folium.PolyLine(rl, color="#1f9d55", weight=5, opacity=.85).add_to(route)
    route.add_to(m)
    gps = folium.FeatureGroup(name="GPS observations (.track)", show=True)
    step = max(1, len(lat) // 3000)
    for y, x, t in zip(lat[::step], lon[::step], tr.track.t.to_numpy()[::step]):
        folium.CircleMarker([y, x], radius=2, color="#1d4ed8", fill=True, fill_opacity=.8, weight=0,
                            tooltip=f"t={t:.0f}s").add_to(gps)
    gps.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    legend = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
              f'border:1px solid #999;border-radius:6px;font:12px sans-serif">'
              f'<b>Trip {tid} ({part})</b><br>{note}'
              f'<span style="color:#1d4ed8">&#9679;</span> GPS observation<br>'
              f'<span style="color:#1f9d55">&#9473;&#9473;</span> reference route<br>'
              f'<span style="color:#9a9a9a">&#9473;&#9473;</span> road network</div>')
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(C.MAPS / f"eda_trip_{tid}.html")


def folium_world(inv):
    m = folium.Map(location=[30, 10], zoom_start=2, tiles="OpenStreetMap")
    for _, r in inv.iterrows():
        folium.CircleMarker([r.lat, r.lon], radius=5, color="#1d4ed8" if r.partition == "train" else "#d9822b",
                            fill=True, tooltip=f"{r.trip} ({r.partition}) {r.n_obs} obs").add_to(m)
    m.save(C.MAPS / "eda_world_trips.html")


if __name__ == "__main__":
    rows = []
    for part in C.PARTITIONS:
        for tid in list_trips(part):
            rows.append(inventory_row(part, tid))
            print("eda", part, tid, flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(C.RESULTS / "eda_inventory.csv", index=False)
    summary = {}
    for part, g in inv.groupby("partition"):
        summary[part] = {
            "trips": len(g), "observations": int(g.n_obs.sum()),
            "obs_per_trip_min_median_max": [int(g.n_obs.min()), int(g.n_obs.median()), int(g.n_obs.max())],
            "missing_values_total": int(g.missing_values.sum()),
            "trips_with_unsorted_timestamps": int((~g.timestamps_sorted).sum()),
            "pct_steps_1s_median": float(g.pct_dt_1s.median()),
            "trips_with_gap_gt_5min": int((g.max_gap_s > 300).sum()),
            "nodes_min_median_max": [int(g.n_nodes.min()), int(g.n_nodes.median()), int(g.n_nodes.max())],
            "isolated_nodes_total": int(g.isolated_nodes.sum()),
            "one_way_pct_median": float(g.pct_one_way.median()),
            "trips_with_route_breaks": int((g.route_breaks > 0).sum()),
            "route_arcs_total": int(g.n_route_arcs.sum()),
        }
    summary["missing_trip_ids_evaluator_partition"] = sorted(set(range(100)) - set(inv.trip.astype(int)))
    (C.RESULTS / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    plots(inv)
    trajectory_plot([("train", "00000029"), ("train", "00000033"), ("student_test", "00000056")])
    folium_world(inv)
    folium_trip_map("train", "00000001", "Typical trip: GPS sits on the route.<br>")
    folium_trip_map("train", "00000033", "Excluded from training: route 60-140 m from GPS.<br>")
    folium_trip_map("student_test", "00000056", "Route revisits arcs (255 repeats).<br>")
    print(json.dumps(summary, indent=2))
