"""Step 8b - Technical report (PDF) in the structure required by 'Deliverables.pdf' (sections 2-8).

Every number is read from results/ (produced by scripts 01-07); figures come from figures/.
Output: docs/PathSnap_Technical_Report.pdf

Run scripts/08_report_figures.py first (static route maps).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from pathsnap import config as C

R, F = C.RESULTS, C.FIGURES
eda = json.loads((R / "eda_summary.json").read_text())
ds = json.loads((R / "dataset_summary.json").read_text())
cmp = pd.read_csv(R / "model_comparison.csv").set_index("model")
tvu = pd.read_csv(R / "tuned_vs_untuned.csv").set_index("model")
bo = json.loads((R / "bayes_opt_best.json").read_text())
trials = pd.read_csv(R / "bayes_opt_trials.csv")
hmmp = json.loads((R / "hmm_params.json").read_text())
hs = json.loads((R / "hmm_summary.json").read_text())
ht = pd.read_csv(R / "hmm_student_test.csv", dtype={"trip": str})
ev = pd.read_csv(R / "evaluator_frozen_run.csv", dtype={"trip": str})
explog = json.loads((R / "experiment_log.json").read_text())

W = A4[0] - 30 * mm
TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=4)
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=5,
                    textColor=colors.HexColor("#1f3b70"))
H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=7, spaceAfter=3,
                    textColor=colors.HexColor("#1f3b70"))
B = ParagraphStyle("b", fontName="Helvetica", fontSize=9, leading=12.2, spaceAfter=3)
S = ParagraphStyle("s", parent=B, fontSize=7.8, leading=9.6)
CAP = ParagraphStyle("cap", parent=B, fontSize=7.8, leading=9.6, textColor=colors.HexColor("#555555"), spaceAfter=6)
CODE = ParagraphStyle("code", fontName="Courier", fontSize=6.6, leading=8.2, backColor=colors.HexColor("#f4f4f4"),
                      borderPadding=4, spaceAfter=6)
story = []
fig_no = [0]


def p(t, st=B):
    story.append(Paragraph(t, st))


def bullets(items, st=B):
    for t in items:
        story.append(Paragraph(f"&bull;&nbsp;{t}", st))


def qa(q, a):
    story.append(Paragraph(f"<b>{q}</b>", B))
    story.append(Paragraph(a, ParagraphStyle("ans", parent=B, leftIndent=10)))


def fig(path, caption, width=W):
    w, h = PILImage.open(path).size
    fig_no[0] += 1
    story.append(KeepTogether([Image(str(path), width=width, height=width * h / w),
                               Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", CAP)]))


def table(rows, widths, font=7.4, header=True, highlight=None):
    data = [[Paragraph(str(c), ParagraphStyle("tc", parent=S, fontSize=font, leading=font + 1.6,
                                              fontName="Helvetica-Bold" if (header and i == 0) else "Helvetica"))
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#b0b8c8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")))
    for r in highlight or []:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff3d6")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 5))


def placeholder(text, height=38 * mm):
    t = Table([[Paragraph(f"<i>{text}</i>", ParagraphStyle("ph", parent=B, alignment=TA_CENTER,
                                                         textColor=colors.HexColor("#888888")))]],
              colWidths=[W], rowHeights=[height])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), .8, colors.HexColor("#999999")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(t)
    story.append(Spacer(1, 6))


pct = lambda x: f"{100 * x:.2f}%"
f3 = lambda x: f"{x:.3f}"
m = lambda name, col: cmp.loc[name, col]
tr, st_ = eda["train"], eda["student_test"]
dtr, dst = ds["train"], ds["student_test"]

# ------------------------------------------------------------------ title
story += [Paragraph("PathSnap: Map-Matching and Route Reconstruction", TITLE),
          Paragraph("Technical report &middot; UE24CS352A Machine Learning &middot; Team: ________________", ParagraphStyle(
              "sub", parent=B, alignment=TA_CENTER)),
          Spacer(1, 8)]
p("<b>Summary.</b> We match noisy GPS observations to a road network with a two-stage system: a "
  "HistGradientBoosting classifier scores candidate road segments for each GPS point, and an HMM decoded with "
  "Viterbi turns those scores into a connected route. On the 10 student-testing trips the final system reaches "
  f"<b>{pct(hs['hmm_top1'])}</b> observation-level Top-1 accuracy (classifier alone: {pct(hs['classifier_only_top1'])}). "
  f"On the 10 evaluator trips, run once after the pipeline was frozen, it reaches <b>{pct((ev.hmm_top1 * ev.scored_points).sum() / ev.scored_points.sum())}</b> "
  f"pooled ({pct(ev.hmm_top1.mean())} average over trips). Every number below can be regenerated from the repository "
  "(scripts 01-08); the partition used is stated for each analysis.")

# ------------------------------------------------------------------ 2.2 EDA
story.append(Paragraph("2.2 Dataset understanding and exploratory data analysis", H1))
p(f"<b>Partitions.</b> train: {tr['trips']} trips, {tr['observations']:,} GPS points; student_test: {st_['trips']} trips, "
  f"{st_['observations']:,} points; evaluator test: 10 trips (IDs {', '.join(map(str, eda['missing_trip_ids_evaluator_partition']))}), "
  "kept out of all development. EDA used train and student_test only.")
p("<b>Structure.</b> Each trip has four headerless text files: <i>.track</i> (longitude, latitude, time in s), <i>.nodes</i> "
  "(longitude, latitude; the row number is the node id), <i>.arcs</i> (from-node, to-node; the row number is the arc id) and "
  "<i>.route</i> (the ordered arc ids the vehicle drove). The most important finding was that <b>every trip ships its own "
  f"road network</b>: trips lie on six continents and networks range from {tr['nodes_min_median_max'][0]:,} to "
  f"{tr['nodes_min_median_max'][2]:,} nodes. Arc ids therefore mean nothing outside their trip, which ruled out treating "
  "the arc id as a class label and shaped the whole design.")

story.append(Paragraph("Q1. How did we preprocess the data?", H2))
bullets([
    "Loaded each file with explicit types and checked counts, ordering and value ranges per trip (results/eda_inventory.csv).",
    "Projected longitude/latitude into each trip's own UTM zone (pyproj), so distances and angles are in metres. "
    "Euclidean distance on degrees is invalid: at 56&deg;N one degree of longitude is about 62 km but one of latitude about 111 km.",
    "Computed the time step per point and flagged gaps longer than 5 s; smoothed speed and heading from the "
    "displacement between points i-2 and i+2; capped speed at 70 m/s (raw jumps reached 180 m/s); marked a point as "
    "moving when its speed is at least 1 m/s, and let a stopped car keep its last moving heading.",
    "Built per-point labels. The route file lists arcs, not one arc per GPS point, so we aligned points to the route with "
    "dynamic programming that only moves forward along the route and minimises total distance. It handles routes that "
    "revisit the same road. A small direction term breaks the tie between the two directions of a two-way road when the "
    "route drives it both ways. Labels more than 50 m from their point were marked unreliable.",
    "Removed trips 00000033 and 00000054 from training: their reference routes sit a median 122 m and 140 m away from the "
    f"GPS trace. After filtering, {dtr['observations_used_for_modelling']:,} train points (78 trips) and "
    f"{dst['observations_used_for_modelling']:,} student-test points were used.",
    f"Generated candidates: every arc within 50 m of a point, at most the 10 nearest (KD-tree over arcs sampled every 20 m). "
    f"That gives {dtr['mean_candidates_per_point']} candidates per point, {dtr['pct_positive_rows']}% of them correct. The "
    f"true arc is missing from the candidate set for {dtr['candidate_miss_rate_pct']}% of train points and "
    f"{dst['candidate_miss_rate_pct']}% of student-test points, which caps attainable accuracy.",
])

story.append(Paragraph("Missing-value and data-quality analysis (train + student_test)", H2))
table([["Check", "Result"],
       ["Missing values (NaN or short rows)", f"{tr['missing_values_total'] + st_['missing_values_total']} in all 90 trips"],
       ["Timestamps strictly increasing", "Yes, every trip"],
       ["Sampling", f"median {tr['pct_steps_1s_median']}% of steps are exactly 1 s; four trips have 20-31% two-second steps"],
       ["Time gaps over 5 minutes", f"{tr['trips_with_gap_gt_5min']} train trips; longest about 42 h (trip 00000090)"],
       ["Isolated nodes", f"{tr['isolated_nodes_total'] + st_['isolated_nodes_total']}"],
       ["Duplicate arcs / self-loops", "1,903 / 24 in total (kept as supplied)"],
       ["Reference routes with connectivity breaks", f"{tr['trips_with_route_breaks']} train, {st_['trips_with_route_breaks']} student_test"],
       ["One-way share of arcs (median)", f"{tr['one_way_pct_median']}% train, {st_['one_way_pct_median']}% student_test"]],
      [70 * mm, W - 70 * mm])

story.append(Paragraph("Q2. What patterns did we observe, which features did we extract and why?", H2))
p("Three patterns drove the features. (1) Picking the nearest arc is right only "
  f"{pct(m('NearestArc(baseline)', 'test_top1_accuracy'))} of the time on student_test. (2) Two-way roads are stored as two "
  "arcs with identical geometry, so for most points the reverse arc is exactly as close as the true one; only direction can "
  "separate them. (3) About a third of points lie within 10 m of an arc end, where neighbouring arcs at a junction are "
  "equally plausible. We therefore built 13 features per (point, candidate) pair, using only GPS and map data:")
table([["Group", "Features", "Reason"],
       ["Geometry", "dist, u (position along arc), d_end, arc_len", "closeness and junction ambiguity"],
       ["Direction", "heading_diff, cos_heading_diff, speed, moving", "separates A&rarr;B from B&rarr;A; says when heading is reliable"],
       ["Relative", "dist_rank, dist_gap, n_cand, has_twin, heading_gap",
        "compares a candidate with its rivals, so values mean the same in a dense city and on a highway"]],
      [22 * mm, 68 * mm, W - 90 * mm])
fig(F / "eda_overview.png", "EDA overview (train and student_test): observations per trip, network size, largest time "
    "gaps, trip locations, sampling regularity and one-way share.")
fig(F / "eda_trajectories.png", "GPS trajectories with the road network and reference route. Left: typical train trip "
    "00000029. Middle: train trip 00000033, excluded because its reference route lies far from the GPS. Right: "
    "student_test trip 00000056, whose route revisits arcs. Interactive Folium versions: maps/eda_trip_*.html.")

# ------------------------------------------------------------------ 3 models
story.append(PageBreak())
story.append(Paragraph("3. Model experimentation and comparison", H1))
p("Each model scores every candidate with P(candidate is the true arc). The prediction for a GPS point is its "
  "highest-scoring candidate (Top-1). Models were trained on the 78 usable train trips with 5-fold GroupKFold "
  "cross-validation by trip, so no trip is split across folds. Each model was then refit on all 78 trips and evaluated once "
  "on student_test. Scalers sit inside pipelines and are fitted on training folds only. We kept the natural class ratio "
  "(about 15% positives) so the probabilities stay usable as HMM emissions. Seed 42 throughout.")
story.append(Paragraph("3.1 Overall model comparison", H2))
rows = [["Model", "Family", "Key configuration", "CV Macro F1", "Top-1 (student-test)", "Route continuity",
         "Student-test Macro F1", "Log loss", "Inference time"]]
for name in ["LogisticRegression", "KNN", "RandomForest", "HistGradientBoosting", "MLP"]:
    rows.append([name, m(name, "family"), m(name, "config"),
                 f"{m(name, 'cv_macro_f1_mean'):.4f} &plusmn; {m(name, 'cv_macro_f1_std'):.3f}",
                 pct(m(name, "test_top1_accuracy")), pct(m(name, "test_route_continuity")),
                 f"{m(name, 'test_macro_f1'):.4f}", f"{m(name, 'test_log_loss'):.4f}",
                 f"{m(name, 'inference_ms_per_gps_point'):.4f} ms/pt"])
rows.append(["Nearest arc (baseline)", "Rule", "smallest distance", "N/A: no training",
             pct(m("NearestArc(baseline)", "test_top1_accuracy")), pct(m("NearestArc(baseline)", "test_route_continuity")),
             f"{m('NearestArc(baseline)', 'test_macro_f1'):.4f}", "N/A: score is not a probability", "-"])
table(rows, [31 * mm, 17 * mm, 31 * mm, 18 * mm, 15 * mm, 15 * mm, 15 * mm, 18 * mm, 20 * mm], font=6.6, highlight=[4])
p(f"CV Top-1 (mean &plusmn; std over folds): " + "; ".join(
    f"{n} {pct(m(n, 'cv_top1_accuracy_mean'))} &plusmn; {100 * m(n, 'cv_top1_accuracy_std'):.1f}"
    for n in ["LogisticRegression", "KNN", "RandomForest", "HistGradientBoosting", "MLP"]) + ".", S)
p("<b>Primary selection metric:</b> observation-level Top-1 accuracy, because it is exactly what the deployed system "
  "reports per GPS point. Log loss was the tie-breaker, because the probabilities become HMM emissions.")

per_model = {
    "LogisticRegression": dict(
        A1="A linear model on our relative features (dist_gap, heading_gap) gives a calibrated probability per candidate, "
           "the form the HMM needs. It is the natural baseline for the learned models.",
        A2="Fast (1 s to fit, about 0.001 ms per point) and interpretable. It cannot express interactions such as 'heading "
           "matters only when the car moves' unless we hand-craft them, and that limited its accuracy.",
        B5="L2 penalty with C = 1.0, lbfgs solver, max_iter 2000, StandardScaler on all 13 features.",
        B6="Binary cross-entropy (log loss) plus the L2 penalty.",
        C9="Mild underfitting: it is the weakest learned model on both CV and student-test, about 1.5-2 points below the "
           "tree models, with a small CV-to-test gap. That pattern points to limited capacity rather than overfitting."),
    "KNN": dict(
        A1="Candidates with similar geometry and direction should behave alike in any city, and the relative features make "
           "them comparable across networks, so a similarity vote is a reasonable non-parametric approach.",
        A2="No training step and flexible boundaries. But it stores all 1.16 M training rows, so it is by far the slowest at "
           "prediction, and its vote-based probabilities are coarse (the worst log loss here).",
        B5="k = 25, inverse-distance weighting, Euclidean distance on standardised features.",
        B6="None: KNN is a lazy learner and nothing is optimised at training time; a prediction is a weighted vote.",
        C9="No overfitting in accuracy: CV and student-test Top-1 are close. The high log loss comes from over-confident "
           "probabilities (votes near 0 or 1), which is a calibration problem."),
    "RandomForest": dict(
        A1="Captures non-linear interactions between distance and direction without scaling, and averaging many "
           "decorrelated trees keeps it stable across very different trips.",
        A2="Best CV Macro F1 and among the best Top-1. But the saved model is 127 MB (over GitHub's 100 MB file limit), "
           "training takes about 175 s, and inference is slower than boosting.",
        B5="200 trees, max_features = sqrt(13), min_samples_leaf = 5, bootstrap sampling.",
        B6="Each tree greedily minimises Gini impurity; the forest averages the tree probabilities.",
        C9="No clear overfitting: the CV-to-test gap (about 0.6 points) matches trip-to-trip variation (fold std about "
           "2.8 points), and min_samples_leaf = 5 limits memorisation. We did not record in-sample scores, so this judgement "
           "rests on the CV-test gap and fold stability."),
    "HistGradientBoosting": dict(
        A1="Gradient boosting is usually the strongest family on tabular data, outputs probabilities directly and trains "
           "quickly on 1.16 M rows thanks to histogram binning.",
        A2="Tied for best CV Top-1, the best log loss of the five, an 8 s fit and a model file under 200 KB. It has more "
           "hyperparameters to tune, and early stopping holds out random rows rather than whole trips.",
        B5="learning_rate 0.1, max_leaf_nodes 31, max_iter 300 with early stopping (10% validation, patience 10).",
        B6="Binary log loss; each new tree fits the negative gradient of the loss.",
        C9="No overfitting: training and validation log loss fall together and stay together (Figure 6, left), and early "
           "stopping halted training at 81 iterations."),
    "MLP": dict(
        A1="A small neural network can learn feature interactions automatically, and it gives a genuine epoch-by-epoch "
           "training history.",
        A2="Matches the tree models on Top-1, has a strong log loss and the fastest inference. It is sensitive to scaling and "
           "to random seed, and it is the hardest of the five to explain.",
        B5="Hidden layers (64, 32) with ReLU, Adam with learning rate 1e-3, L2 alpha 1e-4, batch 512, max_iter 100, early "
           "stopping (10% validation, patience 8).",
        B6="Binary log loss plus the L2 penalty.",
        C9="No overfitting: on a trip-grouped 80/20 split, held-out loss flattens after about 20 epochs and never rises "
           "(Figure 6, middle). It sits below the training loss because the held-out trips happen to be easier."),
}
story.append(Paragraph("3.2 Questions for every model", H2))
for name, a in per_model.items():
    sec = [Paragraph(f"{name} ({m(name, 'family')})", H2)]
    for q, key in [("A1. Why suitable for PathSnap?", "A1"), ("A2. Strengths and limitations", "A2"),
                   ("B5. Important hyperparameters and initial values", "B5"), ("B6. Loss / objective", "B6")]:
        sec += [Paragraph(f"<b>{q}</b> {a[key]}", B)]
    sec += [Paragraph(f"<b>C7-C8. Student-test results.</b> Top-1 {pct(m(name, 'test_top1_accuracy'))}, Macro F1 "
                      f"{m(name, 'test_macro_f1'):.4f} (precision {m(name, 'test_precision_macro'):.4f}, recall "
                      f"{m(name, 'test_recall_macro'):.4f}), log loss {m(name, 'test_log_loss'):.4f}, route continuity "
                      f"{pct(m(name, 'test_route_continuity'))}. CV Top-1 {pct(m(name, 'cv_top1_accuracy_mean'))}, "
                      f"fit {m(name, 'train_fit_s'):.0f} s.", B),
            Paragraph(f"<b>C9. Over/underfitting.</b> {a['C9']}", B)]
    story.append(KeepTogether(sec))

story.append(Paragraph("3.3 Model visualisations", H2))
fig(F / "model_comparison.png", "Left: CV Macro F1 (5-fold GroupKFold, mean &plusmn; std). Middle: student-test Top-1 and "
    "Macro F1, including the nearest-arc baseline. Right: student-test log loss. Partition: train (CV) and student_test.")
fig(F / "model_confusion_matrices.png", "Student-test confusion matrices at the candidate level (threshold 0.5, rows "
    "normalised). Diagnostic only: Top-1 is the decision rule actually used.")
fig(F / "model_training_curves.png", "Training histories, shown only for the models that produce them. Left: "
    "HistGradientBoosting log loss per iteration. Middle: MLP log loss per epoch on a trip-grouped split. Right: "
    "RandomForest feature importance (not a training curve).")
fig(F / "model_cost.png", "Inference time per GPS point and training time on the full train set (log scales).", width=W * .8)
p("<b>Selection.</b> RandomForest, HistGradientBoosting and MLP are statistically tied: their CV Top-1 differences are "
  "far smaller than the 2.8-point fold std. We chose <b>HistGradientBoosting</b>. It has the best log loss (the HMM "
  "consumes these probabilities), trains about 25&times; faster than RandomForest (which made Bayesian optimisation "
  "affordable), its model file is tiny, and its inference cost is negligible. We also ran an error analysis on student-test "
  "for the chosen model. Its remaining mistakes are mostly the reverse twin of the right arc and neighbouring arcs at "
  "junctions. Neither can be resolved one point at a time, which motivated the HMM.")

# ------------------------------------------------------------------ 4 BO
story.append(PageBreak())
story.append(Paragraph("4. Bayesian hyperparameter optimisation (HistGradientBoosting)", H1))
states = bo["trial_states"]
done = trials[trials.state == "COMPLETE"]
qa("1. Hyperparameters searched", "learning_rate, max_leaf_nodes, min_samples_leaf, l2_regularization, max_features, max_bins. "
   "max_iter was fixed at 400, with early stopping at 10% validation and patience 10.")
qa("2. Ranges / distributions", "learning_rate 0.05-0.3 (log-uniform); max_leaf_nodes 15-255 (log-uniform, integer); "
   "min_samples_leaf 20-2000 (log-uniform, integer); l2_regularization 1e-6-10 (log-uniform); max_features 0.4-1.0 "
   "(uniform); max_bins 63-255 (integer). We started learning_rate at 0.05 because with 0.01 trials needed up to 1,000 "
   "rounds and took 20+ minutes each.")
qa("3. Objective", "Maximise mean CV Top-1 over the same 5 trip-grouped folds used in Section 3. Optuna's TPE sampler (a "
   "Bayesian method, seed 42, 10 random warm-up trials) proposed configurations, and a median pruner stopped trials that "
   "were clearly behind after a fold.")
qa("4. Number of trials", f"{bo['n_trials']} trials: {states.get('COMPLETE', 0)} completed, {states.get('PRUNED', 0)} "
   f"pruned, {states.get('FAIL', 0)} interrupted when we stopped the search. We had planned 40, but stopped once gains "
   f"flattened: completed trials span {pct(done.value.min())} to {pct(done.value.max())}, and the last 13 never beat trial "
   f"{bo['best_trial']}. The full search record is in results/bayes_opt_trials.csv and results/bayes_opt_study.db.")
bp = bo["best_params"]
qa("5. Best values", f"Trial {bo['best_trial']}: learning_rate {bp['learning_rate']:.4f}, max_leaf_nodes {bp['max_leaf_nodes']}, "
   f"min_samples_leaf {bp['min_samples_leaf']}, l2_regularization {bp['l2_regularization']:.4f}, max_features "
   f"{bp['max_features']:.3f}, max_bins {bp['max_bins']}. This is a smaller, more regularised model than the default: "
   "fewer leaves, very large leaves and a slower learning rate, which suits labels that carry GPS noise.")
u, t = tvu.loc["HGB untuned (phase 1)"], tvu.loc["HGB tuned (Bayesian)"]
qa("6. Did tuning help?", f"Yes, on every quality metric, by a modest amount: CV Top-1 {pct(u.cv_top1)} &rarr; "
   f"{pct(t.cv_top1)}, student-test Top-1 {pct(u.test_top1)} &rarr; {pct(t.test_top1)}, log loss {u.test_log_loss:.4f} "
   f"&rarr; {t.test_log_loss:.4f}. Differences between trips (about 2.8 points of fold std) are much larger than any "
   "hyperparameter effect. The tuned model used all 400 boosting rounds, so that cap was binding.")
table([["Model", "CV Macro F1", "CV Top-1", "Student-test Top-1", "Student-test Macro F1", "Log loss (test)", "Inference time"],
       ["HGB untuned", f"{u.cv_macro_f1:.4f}", pct(u.cv_top1), pct(u.test_top1), f"{u.test_macro_f1:.4f}",
        f"{u.test_log_loss:.4f}", f"{u.inference_ms_per_gps_point:.3f} ms/pt"],
       ["HGB tuned", f"{t.cv_macro_f1:.4f}", pct(t.cv_top1), pct(t.test_top1), f"{t.test_macro_f1:.4f}",
        f"{t.test_log_loss:.4f}", f"{t.inference_ms_per_gps_point:.3f} ms/pt"]],
      [26 * mm, 22 * mm, 22 * mm, 26 * mm, 28 * mm, 22 * mm, W - 146 * mm], highlight=[2])
fig(F / "bayes_opt_history.png", "Left: CV Top-1 per completed trial and best-so-far (grey lines mark pruned trials; the "
    "dashed line is the untuned model). Right: fANOVA importance; l2_regularization and learning_rate mattered most.")
fig(F / "bayes_opt_slices.png", "CV Top-1 against each hyperparameter for the completed trials (colour shows trial order).")

# ------------------------------------------------------------------ 5 HMM
story.append(PageBreak())
story.append(Paragraph("5. HMM and Viterbi route reconstruction", H1))
qa("1. Hidden states", "The candidate arcs of each GPS point: the same arcs within 50 m (at most 10) that the classifier "
   "scored. Each state keeps its arc id, so every decoded state maps straight back to a road segment.")
qa("2. Emission probabilities", "The tuned classifier's probability for the candidate, used in log space with weight w = 1.")
qa("3. Transition probabilities", "Following Newson and Krumm: exp(-|d_road - d_gps| / &beta;), where d_gps is the "
   "straight-line distance between consecutive GPS points and d_road is the shortest <i>directed</i> road distance between "
   "the two projected positions. One-way streets are respected, and reaching the reverse twin requires driving to a node and "
   "turning. Road distances more than 1 km longer than d_gps are treated as impossible. While the car is effectively "
   f"stationary (speed over a 20-second window below {hmmp['v_stationary']:.0f} m/s), small backward drift along the same "
   "arc counts as GPS jitter rather than reversing. Without this, a parked car's jitter made Viterbi flip to the reverse "
   "twin.")
qa("4. Missing candidates and large gaps", f"The Viterbi chain is split and restarted at any point without candidates and "
   f"after any time gap longer than {hmmp['gap_s']:.0f} s, so a long dropout cannot distort decisions elsewhere.")
qa("5. Disconnected consecutive segments", "If no candidate of the next point is reachable from any candidate of the current "
   "point, the chain is split and decoding restarts. Every point that has candidates still receives a prediction.")
qa("6. Did HMM decoding help?", f"Yes, on every student-test trip. Pooled Top-1 rose from {pct(hs['classifier_only_top1'])} "
   f"to <b>{pct(hs['hmm_top1'])}</b>, route continuity from {pct(hs['classifier_only_continuity'])} to "
   f"{pct(hs['hmm_continuity'])}, and length-weighted route F1 from {hs['classifier_only_route_f1_mean']:.3f} to "
   f"{hs['hmm_route_f1_mean']:.3f} (recall {hs['hmm_route_recall_mean']:.3f}).")
p(f"<b>Decoder tuning</b> (train only, using out-of-fold classifier probabilities so the classifier never scored a trip it "
  f"had seen). We searched &beta;, the gap threshold, the stationary threshold and an off-road option. Only the product "
  f"&beta;&middot;w matters to Viterbi, so w was fixed at 1. Chosen: &beta; = {hmmp['beta']:.0f} m, gap = "
  f"{hmmp['gap_s']:.0f} s, stationary threshold = {hmmp['v_stationary']:.0f} m/s, off-road option disabled. Viterbi "
  "recurrence: &delta;<sub>t</sub>(b) = max<sub>a</sub>[&delta;<sub>t-1</sub>(a) - cost(a,b)/&beta;] + w&middot;log "
  "P<sub>t</sub>(b), followed by backtracking.")
story.append(Spacer(1, 6))
rows = [["Trip", "Points", "Classifier only", "Classifier + HMM", "Change", "Continuity (HMM)", "Route F1 (HMM)", "Segments"]]
for _, r in ht.sort_values("hmm_top1", ascending=False).iterrows():
    rows.append([r.trip, f"{r.n_points:,}", pct(r.clf_top1), pct(r.hmm_top1), f"+{100 * r.delta_top1:.2f}",
                 pct(r.hmm_continuity), f"{r.hmm_route_f1:.3f}", int(r.hmm_segments)])
rows.append(["All", f"{hs['student_test_points']:,}", pct(hs["classifier_only_top1"]), pct(hs["hmm_top1"]),
             f"+{100 * (hs['hmm_top1'] - hs['classifier_only_top1']):.2f}", pct(hs["hmm_continuity"]),
             f"{hs['hmm_route_f1_mean']:.3f}", "-"])
table(rows, [20 * mm, 16 * mm, 24 * mm, 26 * mm, 16 * mm, 26 * mm, 24 * mm, W - 152 * mm], highlight=[len(rows) - 1])
fig(F / "hmm_results.png", "Left: student-test Top-1 per trip, classifier-only vs classifier + HMM. Right: decoder grid "
    "on train (out-of-fold) for the chosen &beta; and gap.")
p("<b>Where the HMM helped least.</b> Trip 00000007 gained only 1.6 points and trip 00000077 stays at 80.75%. Both spend "
  "long stretches parked or crawling, in car parks and next to roads. We split train points by whether the car was really "
  "travelling (speed of at least 3 m/s over 20 s). Strict Top-1 is 98.3% while driving (76% of points) but 76.9% while "
  "parked or crawling. In 70% of those stops the reference label itself switches between arcs while the car stands still, "
  "because GPS jitter crosses arc boundaries and the label alignment follows it; the label matches the arc the car "
  "arrived on only 39% of the time. We treat this as a limitation of the reference data (the brief warns that reference "
  "routes are not error-free) rather than something more modelling can fix, and we report strict Top-1 as defined in the "
  "brief.")

# ------------------------------------------------------------------ 6 folium
story.append(Paragraph("6. Geographical visualisation with Folium", H1))
p("Folium maps are generated from the actual pipeline output: scripts/01_eda.py writes the EDA maps and scripts/06_hmm.py "
  "writes one map per student-test trip (maps/hmm_trip_&lt;id&gt;.html). Each map has separate layers for the GPS "
  "observations, the nearby road network, the reference route, the classifier-only prediction and the classifier + HMM "
  "route. Layers use distinct colours, there is a legend and a layer-control panel, start and end markers, and the view is "
  "fitted to the trip. Figure 10 is an exported static view of the same layers, drawn from the same pipeline output.")
fig(F / "report_route_maps.png", "Student-test trips 00000040 (typical) and 00000077 (hardest). Green: reference "
    "route. Purple: classifier + HMM route. Dashed orange: arcs chosen by the classifier alone. Blue: GPS observations. "
    "The dashed orange stubs off the route are classifier-only mistakes that the HMM removes.")

# ------------------------------------------------------------------ 7 API/UI
story.append(PageBreak())
story.append(Paragraph("7. API, UI and deployment", H1))
p("<b>Laptop 1 (model + API).</b> A FastAPI app (api/main.py) loads the exported artifacts (artifacts/model.joblib, "
  "hmm_params.json, manifest.json) and reads the evaluation trips from data/raw/evaluator. Endpoints: GET /health, "
  "GET /trips (lists the trips found on disk, with no hard-coded ids), POST /predict {trip_id} (runs the full pipeline: "
  "projection, candidates, features, classifier, HMM, then scores the result server-side against the reference route) and "
  "GET /summary (average Top-1 over all evaluation trips, each run through the live pipeline). The prediction step loads "
  "the trip without its route file; the route is read only afterwards for scoring, and only accuracy numbers are "
  "returned. Each request writes one log line with client, trip, points, accuracy and time per stage.")
p("<b>Laptop 2 (UI).</b> A single HTML page with a Leaflet map (ui/index.html), served with python -m http.server. The "
  "evaluator enters laptop 1's address and connects. The page shows a banner with the average Top-1 across all evaluation "
  "trips and one button per trip with its accuracy. Choosing a trip draws the GPS points and predicted route (with an "
  "optional classifier-only layer) and fills a metrics panel: Top-1 for HMM and classifier, continuity, points, "
  "candidates, segments and timings.")
placeholder("Insert screenshot: UI on laptop 2 showing the evaluation-set banner, trip list and a selected trip's route.")
placeholder("Insert screenshot: API log on the CMD of laptop 1 while the UI sends requests.")
p("Excerpt of the API log recorded on laptop 1 during our two-laptop test (client 10.0.5.229 is laptop 2):", S)
story.append(Preformatted(
    "INFO:     10.0.5.229:51796 - \"GET /trips HTTP/1.1\" 200 OK\n"
    "INFO SUMMARY client=10.0.5.229 trips=10 mean_top1_hmm=95.70% pooled_top1_hmm=94.75%\n"
    "INFO PREDICT client=10.0.5.229 trip=00000087 points=1399 candidates=7.0 segments=1 top1_hmm=99.79%\n"
    "     top1_classifier=94.57% features=116ms classifier=41ms hmm=777ms scoring=63ms total=1041ms\n"
    "INFO:     10.0.5.229:51796 - \"POST /predict HTTP/1.1\" 200 OK", CODE))

story.append(Paragraph("Evaluator-test results (frozen pipeline, run once)", H2))
p("Before the evaluator data was first used, we recorded SHA-256 fingerprints of the final artifacts and all pipeline and "
  "API code (results/frozen_pipeline_sha256.txt). The 10 trips were then run once through the API. Nothing was tuned on "
  "these results. Two later changes, a speed-up verified to give identical predictions and a logging fix, are listed in "
  "the same file with new fingerprints.")
rows = [["Trip", "Points scored", "Classifier only", "Classifier + HMM"]]
for _, r in ev.sort_values("hmm_top1", ascending=False).iterrows():
    rows.append([r.trip, f"{r.scored_points:,}", pct(r.classifier_top1), pct(r.hmm_top1)])
pooled_c = (ev.classifier_top1 * ev.scored_points).sum() / ev.scored_points.sum()
pooled_h = (ev.hmm_top1 * ev.scored_points).sum() / ev.scored_points.sum()
rows.append(["Pooled", f"{ev.scored_points.sum():,}", pct(pooled_c), pct(pooled_h)])
rows.append(["Mean of trips", "-", pct(ev.classifier_top1.mean()), pct(ev.hmm_top1.mean())])
table(rows, [30 * mm, 30 * mm, 40 * mm, 40 * mm], highlight=[len(rows) - 2, len(rows) - 1])

# ------------------------------------------------------------------ 8 integrity
story.append(Paragraph("8. Academic integrity and data handling", H1))
bullets([
    "Training, cross-validation, Bayesian optimisation and decoder tuning used only the 78 usable train trips. "
    "student_test was used for model comparison and a single final report. Evaluator data was used once, after freezing.",
    "Ground-truth routes never enter feature computation or the prediction path. They are used only to build training labels "
    "and, on the server, to score predictions.",
    "Every table and figure is produced by scripts 01-08 from results/. Decisions, alternatives and evidence are logged in "
    "docs/decisions.md.",
    "AI assistance disclosure: ______________________________________________ (to be completed by the team as required "
    "by the event's policy).",
])

out = C.ROOT / "docs" / "PathSnap_Technical_Report.pdf"


def footer(canvas, doc):
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawRightString(A4[0] - 15 * mm, 9 * mm, f"PathSnap technical report - page {doc.page}")


SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                  bottomMargin=15 * mm, title="PathSnap Technical Report").build(story, onFirstPage=footer,
                                                                                   onLaterPages=footer)
print("wrote", out)
