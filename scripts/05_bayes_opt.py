"""Step 5 - Bayesian hyperparameter optimisation of the selected model (HistGradientBoosting).

Method    : Optuna TPE sampler (Bayesian, sequential model-based), seed 42, 10 random warm-up trials
Objective : maximise mean CV Top-1 accuracy, 5-fold GroupKFold by trip on the 78 train trips
            (same data and folds as phase 1, so tuned vs untuned is a like-for-like comparison)
Pruning   : MedianPruner - a trial whose running mean Top-1 after a fold is below the median of earlier
            trials at that fold is stopped early (it still counts as one of the 40 trials)
Space     : 6 hyperparameters (see space() below); max_iter=400 with early stopping (patience 10).
            learning_rate starts at 0.05: below that, 400 rounds are not enough and trials took 20+ min (D24)
Student-test is used only after the search, once, to compare the tuned and untuned models.

Outputs
  results/bayes_opt_study.db        Optuna study (full search record, resumable)
  results/bayes_opt_trials.csv      every trial: params, CV Top-1 / log loss / Macro F1, state, time
  results/bayes_opt_best.json       best parameters and scores
  results/tuned_vs_untuned.csv      untuned vs tuned on identical metrics and partitions
  figures/bayes_opt_*.png           optimisation history, parameter importance, parameter slices
  models/HistGradientBoosting_tuned.joblib
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from pathsnap import config as C
from pathsnap.dataset import load, n_points
from pathsnap.features import FEATURES
from pathsnap.metrics import evaluate

# Planned budget was 40 trials; the team concluded the search after 23 completed trials (D31).
# Re-running with PATHSNAP_BO_TRIALS=0 only finalises an existing study (refit, evaluation, plots).
N_TRIALS = int(os.environ.get("PATHSNAP_BO_TRIALS", 40))
STUDY = "hgb_top1"
FIXED = dict(max_iter=400, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=C.SEED)


def space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 255, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 2000, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "max_bins": trial.suggest_int("max_bins", 63, 255),
    }


def main():
    tr_c, tr_p = load("train")
    te_c, te_p = load("student_test")
    X, y, g = tr_c[FEATURES].to_numpy(), tr_c.y.to_numpy(), tr_c.trip.to_numpy()
    folds = list(GroupKFold(n_splits=C.N_FOLDS).split(X, y, g))

    def objective(trial):
        params = space(trial)
        scores, t0 = [], time.perf_counter()
        for k, (a, b) in enumerate(folds):
            m = HistGradientBoostingClassifier(**FIXED, **params).fit(X[a], y[a])
            r = evaluate(tr_c.iloc[b], m.predict_proba(X[b])[:, 1], n_points(tr_p, set(g[b])))
            r["n_iter"] = m.n_iter_
            scores.append(r)
            trial.report(float(np.mean([s["top1_accuracy"] for s in scores])), step=k)
            if trial.should_prune():
                trial.set_user_attr("folds_done", k + 1)
                raise optuna.TrialPruned()
        df = pd.DataFrame(scores)
        for col in ("log_loss", "macro_f1", "route_continuity", "n_iter"):
            trial.set_user_attr(f"cv_{col}", float(df[col].mean()))
        trial.set_user_attr("cv_top1_std", float(df.top1_accuracy.std()))
        trial.set_user_attr("folds_done", len(folds))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        print(f"trial {trial.number:2d}: top1={df.top1_accuracy.mean():.4f} logloss={df.log_loss.mean():.4f} "
              f"{params}", flush=True)
        return float(df.top1_accuracy.mean())

    study = optuna.create_study(
        study_name=STUDY, direction="maximize", storage=f"sqlite:///{C.RESULTS / 'bayes_opt_study.db'}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
    remaining = N_TRIALS - len(study.trials)
    t_search = time.perf_counter()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    search_s = time.perf_counter() - t_search

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    trials.to_csv(C.RESULTS / "bayes_opt_trials.csv", index=False)
    best = study.best_trial

    # refit tuned model on all 78 train trips, evaluate once on student-test
    t0 = time.perf_counter()
    tuned = HistGradientBoostingClassifier(**FIXED, **best.params).fit(X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    p = tuned.predict_proba(te_c[FEATURES].to_numpy())[:, 1]
    infer_s = time.perf_counter() - t0
    test = evaluate(te_c, p, len(te_p))
    joblib.dump(tuned, C.ROOT / "models" / "HistGradientBoosting_tuned.joblib", compress=3)

    un = pd.read_csv(C.RESULTS / "model_comparison.csv").set_index("model").loc["HistGradientBoosting"]
    comp = pd.DataFrame([
        {"model": "HGB untuned (phase 1)", "cv_top1": un.cv_top1_accuracy_mean, "cv_top1_std": un.cv_top1_accuracy_std,
         "cv_macro_f1": un.cv_macro_f1_mean, "cv_log_loss": un.cv_log_loss_mean,
         "test_top1": un.test_top1_accuracy, "test_macro_f1": un.test_macro_f1, "test_log_loss": un.test_log_loss,
         "test_route_continuity": un.test_route_continuity, "fit_s": un.train_fit_s,
         "inference_ms_per_gps_point": un.inference_ms_per_gps_point},
        {"model": "HGB tuned (Bayesian)", "cv_top1": best.value, "cv_top1_std": best.user_attrs["cv_top1_std"],
         "cv_macro_f1": best.user_attrs["cv_macro_f1"], "cv_log_loss": best.user_attrs["cv_log_loss"],
         "test_top1": test["top1_accuracy"], "test_macro_f1": test["macro_f1"], "test_log_loss": test["log_loss"],
         "test_route_continuity": test["route_continuity"], "fit_s": fit_s,
         "inference_ms_per_gps_point": 1000 * infer_s / len(te_p)},
    ])
    comp.to_csv(C.RESULTS / "tuned_vs_untuned.csv", index=False)
    states = trials.state.value_counts().to_dict()
    (C.RESULTS / "bayes_opt_best.json").write_text(json.dumps({
        "method": "Optuna TPESampler(seed=42, n_startup_trials=10) + MedianPruner(n_startup_trials=5)",
        "objective": "maximise mean 5-fold GroupKFold CV Top-1 accuracy (78 train trips)",
        "n_trials": len(study.trials), "trial_states": states, "search_seconds_this_run": search_s,
        "fixed_params": FIXED, "best_trial": best.number, "best_params": best.params,
        "best_cv_top1": best.value, "best_cv_user_attrs": best.user_attrs, "tuned_student_test": test,
        "tuned_refit_iterations": int(tuned.n_iter_),
    }, indent=2, default=str))
    figures(study, trials, un)
    print(comp.round(4).to_string())


def figures(study, trials, un):
    done = trials[trials.state == "COMPLETE"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    ax[0].scatter(done.number, done.value, s=18, label="completed trial", color="#3b6fb6")
    pr = trials[trials.state == "PRUNED"]
    ax[0].scatter(pr.number, [np.nan] * len(pr), s=0)
    for n in pr.number:
        ax[0].axvline(n, color="#ccc", lw=.6, zorder=0)
    ax[0].plot(done.number, done.value.cummax(), color="#d9822b", label="best so far")
    ax[0].axhline(un.cv_top1_accuracy_mean, ls="--", color="grey", label="untuned (phase 1)")
    ax[0].set(title="Bayesian optimisation history (grey lines = pruned trials)", xlabel="trial", ylabel="CV Top-1")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    try:
        imp = optuna.importance.get_param_importances(study)
        ax[1].barh(list(imp)[::-1], list(imp.values())[::-1], color="#3b6fb6")
        ax[1].set(title="Hyperparameter importance (fANOVA)")
    except Exception as e:  # importance needs enough completed trials
        ax[1].text(.1, .5, f"importance unavailable: {e}")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "bayes_opt_history.png", dpi=130)
    plt.close(fig)

    params = [c for c in done.columns if c.startswith("params_")]
    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    for a, c in zip(ax.ravel(), params):
        a.scatter(done[c], done.value, c=done.number, cmap="viridis", s=20)
        a.set(title=c.replace("params_", ""), ylabel="CV Top-1")
        if c.split("_", 1)[1] in ("learning_rate", "l2_regularization", "min_samples_leaf", "max_leaf_nodes"):
            a.set_xscale("log")
        a.grid(alpha=.3)
    fig.suptitle("Parameter slices: CV Top-1 per completed trial (colour = trial order)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "bayes_opt_slices.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
