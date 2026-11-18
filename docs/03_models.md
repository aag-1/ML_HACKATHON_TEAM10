# Step 3: Model training, comparison and selection

**Code:** `scripts/03_train_models.py`, `src/pathsnap/metrics.py`
**Outputs:**
- tables: `results/model_comparison.csv`, `results/cv_fold_scores.csv`, `results/experiment_log.json`, `results/train_log.txt`
- figures: `figures/model_comparison.png`, `figures/model_confusion_matrices.png`, `figures/model_training_curves.png`, `figures/model_cost.png`
- models: `models/*.joblib`

## Protocol
- **Training/CV data:** 78 train trips (trips 33 and 54 excluded, D7), 180,715 GPS points, 1,162,714 candidate rows, 15.5% positive.
- **Cross-validation:** 5-fold **GroupKFold by trip** (D10). Each model is then refit on all 78 trips and evaluated **once** on the 10 student-test trips (19,133 points, 131,484 candidate rows).
- **Preprocessing:** scalers sit inside Pipelines, so they are fitted on training folds only.
- **Class balance:** no resampling (D12).
- **Seed:** 42.
- **Primary metric:** observation-level Top-1 accuracy (D11). Log loss is the tie-breaker.

## Results

| Model | Family | Key configuration | CV Top-1 (mean ± std) | CV Macro F1 | Test Top-1 | Test Macro F1 | Test log loss | Route continuity | Fit (s) | ms / point |
|---|---|---|---|---|---|---|---|---|---|---|
| Nearest arc (baseline) | Rule | Smallest distance | N/A | N/A | 0.540 | 0.542 | N/A (no probabilities) | 0.820 | – | – |
| Logistic Regression | Linear | Scaler + L2, C=1.0 | 0.871 ± 0.025 | 0.909 | 0.856 | 0.905 | 0.136 | 0.911 | 1 | 0.001 |
| KNN | Instance-based | Scaler + k=25, distance-weighted | 0.875 ± 0.026 | 0.922 | 0.871 | 0.921 | 0.219 | 0.920 | 1 | 1.108 |
| Random Forest | Bagging | 200 trees, min_samples_leaf=5 | 0.886 ± 0.027 | 0.931 | 0.878 | 0.927 | 0.094 | 0.924 | 128 | 0.038 |
| **HistGradientBoosting** | Boosting | lr=0.1, 31 leaves, early-stopped at 100 iterations | 0.886 ± 0.027 | 0.930 | 0.875 | 0.924 | 0.094 | 0.923 | 8 | 0.008 |
| MLP | Neural network | Scaler + (64, 32) ReLU, Adam | 0.884 ± 0.027 | 0.928 | 0.877 | 0.924 | 0.093 | 0.924 | 34 | 0.002 |

## Per-model observations
- **Logistic Regression:** the weakest learned model. Its boundary is linear, so it can't express interactions such as "heading matters only when moving". This is mild underfitting. It is still 32 points above the baseline.
- **KNN:** competitive Top-1, but its log loss is the worst (0.219). Distance-weighted votes from 25 neighbours give coarse, over-confident probabilities. It is also about 140× slower per point than HGB, because it searches 1.16M stored rows.
- **Random Forest:** best raw scores by a margin far smaller than the fold std. Fit takes 128 s and the model file is 127 MB, which exceeds GitHub's 100 MB per-file limit.
- **HistGradientBoosting:** training and validation loss track each other (no overfitting). Early stopping ended at 100 iterations after a small loss spike near iteration 92.
- **MLP:** best log loss (0.093) and fastest inference. On the trip-grouped curve, the held-out trips' loss sits below the training loss: those trips are simply easier, not a sign of an error.
- **Over/underfitting check:** CV Top-1 is 0.6–1.5 points above student-test Top-1 for every model, consistent with trip-to-trip variation (fold std ≈0.027). No model shows a train/validation divergence.

## Error analysis (HGB, student-test, 2,402 errors)

| Error type | Share |
|---|---|
| Chose the reverse twin (same road, opposite direction) | 45% |
| Chose an adjacent arc sharing a node (junction) | 35% |
| Point was stationary | 12% |
| True arc not among candidates | 1% |

These are sequence errors that a per-point classifier cannot resolve. They motivate the HMM/Viterbi decoder.

## Selection: HistGradientBoosting (D16)
- **Generalisation:** CV Top-1 0.8857 vs RF 0.8861, a difference of 0.0004, against a fold std of 0.027. The top three models are statistically tied.
- **Probability quality:** test log loss 0.094, the same as RF and on par with MLP. The probabilities become HMM emissions.
- **Cost:** fits in 8 s (RF needs 128 s), which makes Bayesian optimisation with dozens of 5-fold trials practical. Inference takes 0.008 ms per point.
- **Deployability:** a 180 KB model file (RF: 127 MB) loads instantly in the API and fits in the GitHub repository.
- **Explainability:** a boosted tree ensemble, explainable as sequential error correction (each tree fits the errors of the previous ones).
