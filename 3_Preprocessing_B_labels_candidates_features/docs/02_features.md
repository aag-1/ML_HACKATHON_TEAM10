# Step 2: Feature engineering

**Code:** `src/pathsnap/features.py`, called from `scripts/02_build_dataset.py`
**Unit of prediction:** one row per (GPS observation, candidate arc). Target `y = 1` if the candidate is the observation's label arc.

## Why these features
Our probe on student_test (before modelling) showed where a naive matcher goes wrong:

| Observation | Value | Feature response |
|---|---|---|
| Picking the nearest arc is right only | 53% of the time | Distance alone is not enough |
| Points where the true arc's reverse twin (same geometry, opposite direction) is also a candidate | 76% | `heading_diff`, `cos_heading_diff`, `has_twin`, `heading_gap` |
| Points within 10 m of an arc end (junction ambiguity) | 32% | `u`, `d_end`, `arc_len` |
| Nearest arc + GPS heading tie-break | 81% | Direction must be combined with distance |
| Road networks range from 299 to 155,003 nodes | | Relative features: `dist_rank`, `dist_gap`, `n_cand` |
| Stationary points have meaningless heading | 2.4% of train points | `speed`, `moving`; heading carried forward |

## The 13 features

| # | Feature | Group | Definition |
|---|---|---|---|
| 1 | `dist` | Geometry | Perpendicular distance from the GPS point (UTM) to the arc segment, in m |
| 2 | `u` | Geometry | Projection position along the arc, clipped to [0, 1] |
| 3 | `d_end` | Geometry | min(u, 1−u) × arc_len, i.e. metres to the nearest arc end |
| 4 | `arc_len` | Geometry | Arc length, in m |
| 5 | `heading_diff` | Direction | Smallest angle between the GPS heading and the arc bearing (from-node → to-node), 0–180° |
| 6 | `cos_heading_diff` | Direction | cos(heading_diff) |
| 7 | `speed` | Direction | Smoothed speed (±2 points), clipped at 70 m/s |
| 8 | `moving` | Direction | 1 if speed ≥ 1 m/s |
| 9 | `dist_rank` | Relative | Rank of `dist` among the point's candidates (1 = closest; ties share a rank) |
| 10 | `dist_gap` | Relative | dist − min(dist) over the point's candidates |
| 11 | `n_cand` | Relative | Number of candidates of the point |
| 12 | `has_twin` | Relative | 1 if the reverse arc (to → from) is also a candidate |
| 13 | `heading_gap` | Relative | heading_diff − min(heading_diff) over the point's candidates |

## Leakage safeguards
- Features use only the GPS trace and the road graph. `.route` and the labels are never inputs.
- Smoothing windows use neighbouring GPS points only. This is valid at inference time, because the API receives the whole trip.
- Scaling (for LR, KNN and MLP) is inside sklearn Pipelines, fitted on training folds only.

## Considered but not used (D9)
Graph/context features (node degree, one-way flag, connection to the previous point's best candidate). Sequence and connectivity information is handled in phase 2 by the HMM, which uses the graph directly. Keeping it out of the classifier keeps the classifier-only vs classifier+HMM comparison clean.
