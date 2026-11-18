"""Evaluation metrics, candidate-level and observation-level.

Candidate level (binary true/false per candidate row): accuracy, precision, recall, Macro F1, log loss.
Observation level:
  Top-1 accuracy  - for each evaluable GPS point, the highest-scoring candidate is the label arc.
                    Points whose label arc is not among the candidates (or have no candidates) count as wrong.
  Route continuity - share of consecutive predicted arcs (per trip, in time order) that are the same arc or
                    directly connected in the directed graph (prev.to_node == next.from_node).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score


def top1_choice(cands, score):
    c = cands[["trip", "pt", "arc", "arc_u", "arc_v", "y"]].copy()
    c["score"] = score
    return c.loc[c.groupby(["trip", "pt"], sort=False).score.idxmax()].sort_values(["trip", "pt"])


def route_continuity(chosen):
    same_trip = chosen.trip.to_numpy()[1:] == chosen.trip.to_numpy()[:-1]
    arc, u, v = chosen.arc.to_numpy(), chosen.arc_u.to_numpy(), chosen.arc_v.to_numpy()
    ok = (arc[1:] == arc[:-1]) | (v[:-1] == u[1:])
    return float(ok[same_trip].mean())


def evaluate(cands, proba, n_eval_points):
    y = cands.y.to_numpy()
    pred = (proba >= 0.5).astype(int)
    chosen = top1_choice(cands, proba)
    return {
        "top1_accuracy": float(chosen.y.sum()) / n_eval_points,
        "macro_f1": f1_score(y, pred, average="macro"),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "candidate_accuracy": accuracy_score(y, pred),
        "log_loss": log_loss(y, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "route_continuity": route_continuity(chosen),
    }


if __name__ == "__main__":  # tiny self-check
    c = pd.DataFrame({"trip": ["a"] * 4, "pt": [0, 0, 1, 1], "arc": [5, 6, 7, 8], "arc_u": [1, 2, 3, 9],
                      "arc_v": [3, 4, 5, 9], "y": [1, 0, 0, 1]})
    r = evaluate(c, np.array([.9, .2, .6, .4]), n_eval_points=2)
    assert r["top1_accuracy"] == 0.5 and r["route_continuity"] == 1.0, r  # arc5 (1->3) then arc7 (3->5)
    print("metrics self-check ok", r)
