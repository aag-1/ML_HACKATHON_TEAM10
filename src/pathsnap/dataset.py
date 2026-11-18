"""Load the processed modelling data (written by scripts/02_build_dataset.py) with the phase-1 filters applied:
train drops the excluded bad-label trips (D7); both partitions drop points with unreliable labels."""
import pandas as pd

from . import config as C


def load(part):
    cand = pd.read_parquet(C.PROCESSED / f"candidates_{part}.parquet")
    pts = pd.read_parquet(C.PROCESSED / f"points_{part}.parquet")
    keep = pts.label_reliable
    if part == "train":
        keep &= ~pts.trip.isin(C.EXCLUDED_TRAIN_TRIPS)
    pts = pts[keep]
    cand = cand.merge(pts[["trip", "pt"]], on=["trip", "pt"])  # same filter on candidate rows
    return cand.reset_index(drop=True), pts.reset_index(drop=True)


def n_points(pts, trips):
    return int(pts.trip.isin(trips).sum())
