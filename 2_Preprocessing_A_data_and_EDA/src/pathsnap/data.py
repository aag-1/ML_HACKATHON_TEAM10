"""Raw-file loading. One trip = .track (GPS), .nodes, .arcs (road graph), .route (reference, optional).

The reference route is loaded only when explicitly asked for, so the prediction path never sees it.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import RAW


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame       # columns lon, lat, t  (one row per GPS observation)
    nodes: np.ndarray         # (N, 2) lon, lat; row index = node_id
    arcs: np.ndarray          # (A, 2) from_node, to_node; row index = arc_id (directed)
    route: np.ndarray | None  # ordered arc_ids driven (reference), None unless requested


def _read(path, names, dtype):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype, engine="c")


def list_trips(partition, root=RAW):
    return sorted(p.name for p in (root / partition).iterdir() if p.is_dir())


def load_trip(trip_id, partition, with_route=False, root=RAW):
    d = root / partition / trip_id
    track = _read(d / f"{trip_id}.track", ["lon", "lat", "t"], float)
    track = track.dropna().sort_values("t", kind="stable").reset_index(drop=True)
    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"], float).to_numpy()
    arcs = _read(d / f"{trip_id}.arcs", ["u", "v"], np.int64).to_numpy()
    route = None
    if with_route:
        route = _read(d / f"{trip_id}.route", ["arc"], np.int64)["arc"].to_numpy()
    return Trip(trip_id, partition, track, nodes, arcs, route)
