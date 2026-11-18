# Step 5: HMM + Viterbi route reconstruction, and final evaluation

**Code:**
- decoder: `src/pathsnap/hmm.py` (self-check: `PYTHONPATH=src python -m pathsnap.hmm`)
- shared pipeline: `src/pathsnap/pipeline.py`
- tuning and evaluation: `scripts/06_hmm.py`
- export: `scripts/07_export_artifacts.py`

**Outputs:**
- decoder tuning: `results/hmm_grid_train.csv`, `results/hmm_params.json`
- student-test: `results/hmm_student_test.csv`, `results/hmm_summary.json`, `figures/hmm_results.png`, `maps/hmm_trip_<id>.html`
- frozen evaluator run: `results/frozen_pipeline_sha256.txt`, `results/evaluator_frozen_run.csv`

## Implementation

| Question (brief §5) | Answer |
|---|---|
| Hidden states | The candidate arcs of each GPS point (arcs within 50 m, max 10), i.e. the same candidates the classifier scored. Each state keeps its arc id, so the candidate-to-road-segment correspondence is preserved. |
| Emission probability | Tuned-HGB probability P(candidate is the true arc), used in log space with weight w (D19). |
| Transition probability | exp(−\|d_road − d_gps\| / β) (Newson & Krumm). d_road is the **shortest directed path** from candidate A's projected position to candidate B's, so one-way roads and travel direction are respected. Moving onto the reverse twin requires driving to a node and turning. Transitions longer than d_gps + 1 km are impossible (D20). |
| Parked periods | If the ≈20 s speed is below v_stationary, backward drift along the same arc is treated as GPS jitter, not reversing (D30). |
| Missing candidates / large gaps | The chain is split and Viterbi restarted at points with no candidates and at time gaps > gap_s (D21). |
| Disconnected consecutive segments | If no candidate of point t+1 is reachable from any candidate of point t, the chain is split and restarted, so one bad stretch cannot corrupt the rest of the trip. |
| Viterbi recurrence | δ_t(b) = max_a [δ_{t−1}(a) − cost(a, b)/β] + w·log P_t(b); backtracking from argmax δ_T. |
| Output | One predicted arc per GPS point. The route is reconstructed by joining consecutive predicted arcs with the shortest directed path. |

**Tuning (D22):** the grid covered β ∈ {2, 3, 5, 8}, gap_s ∈ {30, 60, 120}, v_stationary ∈ {off, 4, 8, 12} and d_off ∈ {off, 15, 20}. w is fixed at 1, because only β·w matters. Selection used out-of-fold tuned-HGB probabilities on the 78 train trips.
- **Chosen:** β = 3, gap_s = 60 s, v_stationary = 12 m/s, d_off = off.
- **Train out-of-fold Top-1:** classifier-only 88.88% → HMM 93.32%.

## Student-test (10 trips, 19,133 scored points, reported once)

| Metric | Classifier only | **Classifier + HMM** |
|---|---|---|
| Strict observation-level Top-1 | 88.35% | **93.09%** |
| Route continuity | 92.66% | **98.33%** |
| Route F1 (length-weighted, mean over trips) | 0.930 | **0.984** |
| Route precision / recall (HMM) | – | 0.972 / 0.997 |

| Trip | Classifier | **HMM** | Trip | Classifier | **HMM** |
|---|---|---|---|---|---|
| 00000052 | 97.33% | **99.62%** | 00000056 | 90.92% | 95.60% |
| 00000000 | 95.13% | **97.88%** | 00000065 | 90.49% | 94.79% |
| 00000007 | 95.72% | **97.34%** | 00000091 | 87.76% | 94.62% |
| 00000040 | 93.19% | **97.06%** | 00000009 | 88.37% | 92.00% |
| 00000076 | 90.78% | **96.84%** | 00000077 | 73.36% | 80.75% |

- **The HMM improved every trip,** by 1.6 to 7.4 points. There was no trip where it hurt.
- **Where it helped least:** trip 7 (+1.6) and trip 77, which stays at 80.75%. Trip 77 spends about half its time crawling around a car park. There, the reference labels switch between arcs with the GPS jitter, so no consistent answer exists (see the error analysis below).

### Why strict Top-1 is not higher: error analysis (train, out-of-fold)
- **While driving** (≈20 s speed ≥ 3 m/s, 76% of points): strict Top-1 is **98.3%**.
- **While parked or crawling** (24% of points): **76.9%**.
- **The cause is the labels:** in 70% of stops, the reference label itself jumps between two or more arcs while the car is stationary, because GPS jitter crosses arc boundaries and the label alignment follows it. The label matches the arc the car arrived on only 39% of the time.
- **Remaining errors:**
  - same road in the opposite direction while parked: 2.6% of points;
  - neighbouring arc at a junction: 2.6%;
  - other: 1.2%.
- **Correct physical road** (direction ignored) on student-test: ≈96%.

This is the reference-route limitation the brief asks teams to identify (Dataset Exploration §6).

## Frozen evaluator-test run (deployment check)
- **Freeze recorded first:** SHA-256 fingerprints of the final artifacts and all pipeline/API code were recorded **before** the evaluator data was first used (`results/frozen_pipeline_sha256.txt`, 14:47 IST).
- **Run once:** the 10 evaluator trips were then run once through the API (`POST /predict`), exactly as on demo day.
- **No changes afterwards:** the fingerprints were re-verified after the run, and nothing was tuned or changed based on these results.

| Trip | Classifier | **HMM** | Trip | Classifier | **HMM** |
|---|---|---|---|---|---|
| 00000087 | 94.57% | **99.79%** | 00000094 | 92.32% | 95.48% |
| 00000049 | 98.94% | **99.62%** | 00000017 | 93.02% | 95.02% |
| 00000067 | 96.94% | **98.69%** | 00000023 | 94.10% | 94.54% |
| 00000073 | 97.73% | **98.69%** | 00000015 | 71.50% | 90.54% |
| 00000059 | 95.83% | **98.06%** | 00000030 | 81.19% | 86.54% |

- **Pooled over 19,652 scored points:** classifier-only 86.57% → **HMM 94.75%**.
- **5 of 10 trips** score ≥ 96%.
- **Every trip completed,** with a maximum of 8.6 s server time per trip.
