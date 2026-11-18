"""Step 3 - Train and compare 5 candidate-scoring models (+ a no-training nearest-arc baseline).

Protocol
  * Fit/CV data: 80 train trips minus the 2 excluded bad-label trips; points with unreliable labels dropped.
  * 5-fold GroupKFold by trip (no trip is split across folds) -> CV metrics (mean +- std).
  * Refit on all CV data -> evaluate once on the 10 student_test trips.
  * Scalers live inside Pipelines, so they are fitted on training folds only.
  * Natural class ratio (no resampling) so probabilities stay usable as HMM emissions.

Outputs
  results/cv_fold_scores.csv, results/model_comparison.csv, results/experiment_log.json
  figures/model_*.png, models/<name>.joblib
"""
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap import config as C
from pathsnap.dataset import load, n_points
from pathsnap.features import FEATURES
from pathsnap.metrics import evaluate

MODELS_DIR = C.ROOT / "models"

MODELS = {
    "LogisticRegression": ("Linear / probabilistic", Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000))])),
    "KNN": ("Instance-based", Pipeline([
        ("scale", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))])),
    "RandomForest": ("Ensemble (bagging)", RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=C.SEED)),
    "HistGradientBoosting": ("Boosting", HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=300, max_leaf_nodes=31, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=10, random_state=C.SEED)),
    "MLP": ("Neural network", Pipeline([
        ("scale", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3,
                              batch_size=512, max_iter=100, early_stopping=True, n_iter_no_change=8,
                              random_state=C.SEED))])),
}


def log(msg):
    print(msg, flush=True)
    with open(C.RESULTS / "train_log.txt", "a") as f:
        f.write(msg + "\n")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    (C.RESULTS / "train_log.txt").write_text("")
    tr_c, tr_p = load("train")
    te_c, te_p = load("student_test")
    Xtr, ytr, gtr = tr_c[FEATURES].to_numpy(), tr_c.y.to_numpy(), tr_c.trip.to_numpy()
    Xte = te_c[FEATURES].to_numpy()
    log(f"train: {tr_c.trip.nunique()} trips, {len(tr_p)} points, {len(tr_c)} candidate rows, "
        f"{100 * ytr.mean():.2f}% positive | student_test: {len(te_p)} points, {len(te_c)} rows")

    folds = list(GroupKFold(n_splits=C.N_FOLDS).split(Xtr, ytr, gtr))
    fold_rows, comp_rows, configs = [], [], {}

    # Baseline (no learning): score = -distance, i.e. pick the nearest arc.
    for k, (_, va) in enumerate(folds):
        r = evaluate(tr_c.iloc[va], 1 / (1 + tr_c.dist.to_numpy()[va]), n_points(tr_p, set(gtr[va])))
        fold_rows.append({"model": "NearestArc(baseline)", "fold": k, **r})
    rb = evaluate(te_c, 1 / (1 + te_c.dist.to_numpy()), len(te_p))
    comp_rows.append({"model": "NearestArc(baseline)", "family": "Rule (no training)",
                      "config": "pick candidate with smallest distance", **{f"test_{k}": v for k, v in rb.items()}})

    test_proba = {}
    for name, (family, model) in MODELS.items():
        configs[name] = {k: repr(v) for k, v in model.get_params(deep=True).items() if "__" in k or not hasattr(model, "steps")}
        fit_times = []
        for k, (tr, va) in enumerate(folds):
            t0 = time.perf_counter()
            m = sklearn.base.clone(model).fit(Xtr[tr], ytr[tr])
            fit_times.append(time.perf_counter() - t0)
            p = m.predict_proba(Xtr[va])[:, 1]
            r = evaluate(tr_c.iloc[va], p, n_points(tr_p, set(gtr[va])))
            fold_rows.append({"model": name, "fold": k, "fit_s": fit_times[-1], **r})
            log(f"{name:22} fold {k}: top1={r['top1_accuracy']:.4f} macroF1={r['macro_f1']:.4f} "
                f"logloss={r['log_loss']:.4f} fit={fit_times[-1]:.1f}s")

        t0 = time.perf_counter()
        final = sklearn.base.clone(model).fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = final.predict_proba(Xte)[:, 1]
        infer_s = time.perf_counter() - t0
        test_proba[name] = p
        r = evaluate(te_c, p, len(te_p))
        cv = pd.DataFrame([f for f in fold_rows if f["model"] == name])
        comp_rows.append({
            "model": name, "family": family, "config": short_config(name, final),
            **{f"cv_{k}_mean": cv[k].mean() for k in ("top1_accuracy", "macro_f1", "log_loss", "route_continuity")},
            **{f"cv_{k}_std": cv[k].std() for k in ("top1_accuracy", "macro_f1")},
            **{f"test_{k}": v for k, v in r.items()},
            "train_fit_s": fit_s, "test_inference_s": infer_s,
            "inference_ms_per_gps_point": 1000 * infer_s / len(te_p),
        })
        log(f"{name:22} STUDENT_TEST top1={r['top1_accuracy']:.4f} macroF1={r['macro_f1']:.4f} "
            f"logloss={r['log_loss']:.4f} continuity={r['route_continuity']:.4f} fit={fit_s:.1f}s infer={infer_s:.2f}s")
        if name != "KNN":  # KNN stores the full training set; refit is cheap, no need to keep it
            joblib.dump(final, MODELS_DIR / f"{name}.joblib", compress=3)
        if name == "HistGradientBoosting":
            hgb_curve = (-final.train_score_, -final.validation_score_)
        if name == "RandomForest":
            rf_importance = pd.Series(final.feature_importances_, FEATURES).sort_values()

    folds_df, comp = pd.DataFrame(fold_rows), pd.DataFrame(comp_rows)
    folds_df.to_csv(C.RESULTS / "cv_fold_scores.csv", index=False)
    comp.to_csv(C.RESULTS / "model_comparison.csv", index=False)
    mlp_curve = mlp_epoch_curves(tr_c, Xtr, ytr, gtr)
    figures(comp, folds_df, te_c, test_proba, hgb_curve, mlp_curve, rf_importance)
    (C.RESULTS / "experiment_log.json").write_text(json.dumps({
        "seed": C.SEED, "cv": f"GroupKFold(n_splits={C.N_FOLDS}) grouped by trip",
        "train_trips_used": int(tr_c.trip.nunique()), "excluded_train_trips": sorted(C.EXCLUDED_TRAIN_TRIPS),
        "features": FEATURES, "model_params": configs,
        "candidate_radius_m": C.CAND_RADIUS_M, "candidate_max": C.CAND_MAX,
        "python": platform.python_version(), "sklearn": sklearn.__version__, "machine": f"{platform.system()} {platform.machine()}",
    }, indent=2))
    log("\n" + comp.round(4).to_string())


def short_config(name, m):
    return {
        "LogisticRegression": "StandardScaler + L2, C=1.0, lbfgs",
        "KNN": "StandardScaler + k=25, distance-weighted, Euclidean",
        "RandomForest": "200 trees, min_samples_leaf=5, max_features=sqrt",
        "HistGradientBoosting": f"lr=0.1, max_leaf_nodes=31, early-stopped at {getattr(m, 'n_iter_', '?')} iters",
        "MLP": f"StandardScaler + (64,32) ReLU, Adam lr=1e-3, alpha=1e-4, early-stopped at "
               f"{getattr(m[-1] if hasattr(m, 'steps') else m, 'n_iter_', '?')} epochs",
    }[name]


def mlp_epoch_curves(tr_c, X, y, g, epochs=40):
    """Real epoch-wise train/validation log loss for the MLP on a trip-grouped 80/20 split of the train data."""
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=C.SEED).split(X, y, g))
    sc = StandardScaler().fit(X[tr])
    Xa, Xv = sc.transform(X[tr]), sc.transform(X[va])
    m = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, learning_rate_init=1e-3, batch_size=512,
                      random_state=C.SEED)
    rng = np.random.default_rng(C.SEED)
    hist = []
    for e in range(epochs):
        idx = rng.permutation(len(Xa))
        m.partial_fit(Xa[idx], y[tr][idx], classes=[0, 1])
        hist.append((log_loss(y[tr], m.predict_proba(Xa)[:, 1]), log_loss(y[va], m.predict_proba(Xv)[:, 1])))
    return np.array(hist)


def figures(comp, folds_df, te_c, test_proba, hgb_curve, mlp_curve, rf_imp):
    models = [m for m in comp.model if m != "NearestArc(baseline)"]
    cmp = comp.set_index("model")
    col = "#3b6fb6"

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].bar(models, cmp.loc[models, "cv_macro_f1_mean"], yerr=cmp.loc[models, "cv_macro_f1_std"], color=col, capsize=4)
    ax[0].set(title="Cross-validation Macro F1 (5-fold GroupKFold, mean ± std)", ylim=(0.5, 1))
    w = 0.38
    xs = np.arange(len(models) + 1)
    names = ["NearestArc(baseline)"] + models
    ax[1].bar(xs - w / 2, cmp.loc[names, "test_top1_accuracy"], w, label="Top-1 accuracy", color=col)
    ax[1].bar(xs + w / 2, cmp.loc[names, "test_macro_f1"], w, label="Macro F1", color="#d9822b")
    ax[1].set_xticks(xs, ["Nearest\n(baseline)"] + models)
    ax[1].set(title="Student-test: Top-1 accuracy and Macro F1", ylim=(0.4, 1))
    ax[1].legend()
    ax[2].bar(models, cmp.loc[models, "test_log_loss"], color=col)
    ax[2].set(title="Student-test log loss (lower is better)")
    for a in ax:
        a.tick_params(axis="x", rotation=25)
        a.grid(axis="y", alpha=.3)
        for c in a.containers:
            if hasattr(c, "datavalues"):
                a.bar_label(c, fmt="%.3f", fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "model_comparison.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(models, cmp.loc[models, "inference_ms_per_gps_point"], color=col)
    ax[0].set(title="Inference time per GPS point (ms, student-test)", yscale="log")
    ax[1].bar(models, cmp.loc[models, "train_fit_s"], color="#6c8f3b")
    ax[1].set(title="Training time on full train data (s)", yscale="log")
    for a in ax:
        a.tick_params(axis="x", rotation=25)
        a.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "model_cost.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8))
    for a, name in zip(ax, models):
        cm = confusion_matrix(te_c.y, (test_proba[name] >= 0.5).astype(int), normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=["wrong arc", "true arc"]).plot(ax=a, colorbar=False, values_format=".3f")
        a.set_title(name, fontsize=10)
    fig.suptitle("Student-test confusion matrices (candidate level, threshold 0.5, row-normalised)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "model_confusion_matrices.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].plot(hgb_curve[0], label="train (90%)")
    ax[0].plot(hgb_curve[1], label="early-stopping validation (10%)")
    ax[0].set(title="HistGradientBoosting: log loss per boosting iteration", xlabel="iteration", ylabel="log loss")
    ax[1].plot(range(1, len(mlp_curve) + 1), mlp_curve[:, 0], label="train trips (80%)")
    ax[1].plot(range(1, len(mlp_curve) + 1), mlp_curve[:, 1], label="held-out trips (20%)")
    ax[1].set(title="MLP: log loss per epoch (trip-grouped split)", xlabel="epoch", ylabel="log loss")
    ax[2].barh(rf_imp.index, rf_imp.values, color=col)
    ax[2].set(title="RandomForest feature importance (impurity)")
    for a in ax[:2]:
        a.legend()
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "model_training_curves.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
