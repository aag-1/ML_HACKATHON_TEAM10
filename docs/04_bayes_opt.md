# Step 4: Bayesian hyperparameter optimisation (HistGradientBoosting)

**Code:** `scripts/05_bayes_opt.py`
**Outputs:**
- tables: `results/bayes_opt_study.db` (full Optuna study), `results/bayes_opt_trials.csv`, `results/bayes_opt_best.json`, `results/tuned_vs_untuned.csv`
- figures: `figures/bayes_opt_history.png`, `figures/bayes_opt_slices.png`
- model: `models/HistGradientBoosting_tuned.joblib`

## Set-up (decisions D17, D24, D31)

| Item | Choice |
|---|---|
| Method | Optuna **TPE** sampler (Bayesian, sequential model-based), seed 42, 10 random warm-up trials |
| Pruning | MedianPruner: a trial whose running mean Top-1 after a fold is below the median of earlier trials is stopped |
| Objective | **Maximise mean CV Top-1**: 5-fold GroupKFold by trip on the 78 train trips (same folds as phase 1) |
| Budget | Planned 40 trials; **concluded after 23 completed** (9 pruned, 1 interrupted when stopped; 33 in total) |
| Compute | ≈71 min in total; a median completed trial took 2.1 min |
| Fixed | max_iter 400, early stopping (10% validation, patience 10), random_state 42 |

## Search space and result

| Hyperparameter | Range (distribution) | Best (trial 19) |
|---|---|---|
| learning_rate | 0.05–0.3 (log-uniform) | **0.0630** |
| max_leaf_nodes | 15–255 (log-uniform, integer) | **19** |
| min_samples_leaf | 20–2000 (log-uniform, integer) | **608** |
| l2_regularization | 1e-6–10 (log-uniform) | **0.00163** |
| max_features | 0.4–1.0 (uniform) | **0.883** |
| max_bins | 63–255 (integer) | **178** |

The best configuration is a **smaller, more regularised** model than the default: fewer leaves (19 vs 31), large leaves (608 samples each), a slower learning rate and feature subsampling. That is consistent with noisy labels (GPS noise, junction ambiguity): heavier regularisation generalises better across trips.

Completed trials ranged only from 0.8874 to 0.8900 CV Top-1 (median 0.8895). The objective is flat around the optimum, and the last 13 trials did not beat trial 19, which supports stopping at 23.

## Tuned vs untuned (same metrics, same partitions)

| Model | CV Top-1 | CV Macro F1 | CV log loss | Test Top-1 | Test Macro F1 | Test log loss | Test continuity | Fit (s) | ms / point |
|---|---|---|---|---|---|---|---|---|---|
| HGB untuned (phase 1) | 0.8886 ± 0.029 | 0.9317 | 0.0939 | 0.8790 | 0.9262 | 0.0885 | 0.9245 | 6.7 | 0.008 |
| **HGB tuned (Bayesian)** | **0.8900 ± 0.028** | **0.9323** | **0.0918** | **0.8835** | **0.9291** | **0.0858** | **0.9265** | 21.7 | 0.034 |

- **The tuned model is better on every quality metric:** +0.14 points CV Top-1, +0.45 points student-test Top-1, and lower log loss (0.0885 → 0.0858). Better calibrated probabilities matter most here, because they are the HMM's emission scores.
- **The gain is modest.** That is expected: the CV std across trips (≈0.028) is much larger than any hyperparameter effect. Classifier-only accuracy is limited by per-point ambiguity (reverse twins, junctions), which the HMM addresses.
- **Cost:** training is about 3× slower (more, smaller trees). Inference is still 0.03 ms per GPS point.
- **Limitation:** the refit used all 400 boosting rounds (early stopping did not trigger), so the max_iter cap was binding. A larger cap could add a little, at higher training cost.
