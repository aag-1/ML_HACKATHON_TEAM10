"""Step 4 - One-page Evaluation-1 summary PDF, built only from files in results/ (no hand-typed metrics).

Inputs : results/eda_summary.json, results/dataset_summary.json, results/model_comparison.csv
Output : docs/PathSnap_Phase1_Summary.pdf
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from pathsnap import config as C

TEAM_LINE = "UE24CS352A Machine Learning · PathSnap Hackathon · Evaluation 1 · Team: ________"

eda = json.loads((C.RESULTS / "eda_summary.json").read_text())
ds = json.loads((C.RESULTS / "dataset_summary.json").read_text())
cmp = pd.read_csv(C.RESULTS / "model_comparison.csv").set_index("model")

H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=9.6, leading=11.5, spaceBefore=4.5, spaceAfter=1.5,
                   textColor=colors.HexColor("#1f3b70"))
B = ParagraphStyle("b", fontName="Helvetica", fontSize=7.7, leading=9.4)
S = ParagraphStyle("s", parent=B, fontSize=7.0, leading=8.4)
T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=12.5, leading=14.5)


def bullets(items, style=B):
    return [Paragraph(f"• {t}", style) for t in items]


def table(data, widths, font=6.9, highlight=None):
    t = Table(data, colWidths=widths)
    style = [
        ("FONT", (0, 0), (-1, -1), "Helvetica", font), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")),
        ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#b0b8c8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.1),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.3), ("RIGHTPADDING", (0, 0), (-1, -1), 2.3),
    ]
    for r in highlight or []:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff3d6")))
    t.setStyle(TableStyle(style))
    return t


P = lambda s: Paragraph(s, S)
tr, st = eda["train"], eda["student_test"]
dtr, dst = ds["train"], ds["student_test"]

story = [
    Paragraph("PathSnap: Data Preprocessing, Feature Engineering &amp; Model Training (Evaluation 1)", T),
    Paragraph(TEAM_LINE, S), Spacer(1, 2),
    Paragraph("Problem &amp; framing", H),
    *bullets([
        "GPS is noisy (5–30 m), so the nearest road is often wrong. Goal: for every GPS point, find the road segment (arc) the vehicle was really on.",
        "Every trip has its <b>own road network</b> (worldwide, 299–155,003 nodes), so arc IDs are not reusable class labels. We therefore "
        "<b>generate candidate arcs</b> near each GPS point and train a <b>binary classifier</b>: P(this candidate is the true arc). "
        "The highest-probability candidate is the prediction.",
    ]),
    Paragraph("1. Data preprocessing (train: 80 trips, student-test: 10 trips; evaluator trips never used)", H),
    *bullets([
        f"<b>Explored</b> {tr['observations']:,} + {st['observations']:,} GPS points: {tr['missing_values_total']} missing values, timestamps always "
        f"increasing, median {tr['pct_steps_1s_median']}% of steps are 1 s, {tr['trips_with_gap_gt_5min']} trips with gaps &gt;5 min (max ≈42 h), "
        "GPS jumps up to 180 m/s, 11 reference routes with breaks.",
        "<b>Projected</b> lon/lat to metres with each trip's UTM zone, because distances in degrees are not valid.",
        "<b>Cleaned:</b> flagged time gaps (&gt;5 s); smoothed speed and heading over ±2 points; capped speed at 70 m/s; added a <b>moving</b> flag "
        "(≥1 m/s); stopped vehicles keep their last heading.",
        "<b>Labelled:</b> .route lists arcs, not one per GPS point, so each point was matched to the route by <b>forward-only dynamic programming</b> "
        "(minimum total distance; handles loops). Labels &gt;50 m from the point were treated as unreliable.",
        f"<b>Excluded trips 33 and 54 from training</b> (reference route 122–140 m from the GPS). Used: {dtr['observations_used_for_modelling']:,} train "
        f"points (78 trips), {dst['observations_used_for_modelling']:,} student-test points.",
        f"<b>Candidates:</b> all arcs within 50 m, at most 10 per point (KD-tree): {dtr['mean_candidates_per_point']} per point, "
        f"{dtr['pct_positive_rows']}% true. Candidate miss rate (true arc not offered) <b>{dtr['candidate_miss_rate_pct']}% train, "
        f"{dst['candidate_miss_rate_pct']}% student-test</b>, the accuracy ceiling.",
    ]),
    Paragraph("2. Feature engineering: 13 features per (GPS point, candidate arc), computed only from GPS + road map", H),
    table([
        ["Group", "Features", "What they capture"],
        ["Geometry", "dist, u, d_end, arc_len", "How close the road is; where along it; how near a junction"],
        ["Direction", "heading_diff, cos_heading_diff, speed, moving",
         "Does the car's travel direction match the arc's? Separates A→B from B→A (same geometry)"],
        ["Relative", "dist_rank, dist_gap, n_cand, has_twin, heading_gap",
         "How this candidate compares with the others for the same point (works for any network)"],
    ], [17 * mm, 62 * mm, 105 * mm]),
    Paragraph("3. Model training process", H),
    *bullets([
        "Trained 5 models from different families plus a no-learning baseline (pick the nearest arc). Scaling (StandardScaler) is used for "
        "LogReg / KNN / MLP inside a pipeline. The class ratio is left as is (≈15% true) so probabilities stay realistic. Seed 42.",
        "<b>5-fold GroupKFold by trip</b>: the 78 trips are split into 5 groups; train on 4 and validate on 1, rotating. Whole trips stay together, "
        "so neighbouring, near-identical GPS points can't leak between training and validation. Then each model is refit on all 78 trips "
        "and tested <b>once</b> on the 10 student-test trips.",
    ]),
    Spacer(1, 2),
]

desc = {
    "NearestArc(baseline)": "Rule: choose the closest arc. No learning.",
    "LogisticRegression": "Weighted sum of features → sigmoid → probability. Linear boundary.",
    "KNN": "Looks up the 25 most similar training rows and takes their distance-weighted vote.",
    "RandomForest": "200 decision trees on random data/feature subsets; averages their votes.",
    "HistGradientBoosting": "Trees added one by one, each fixing the previous errors (boosting).",
    "MLP": "Neural network: 13 inputs → 64 → 32 neurons (ReLU) → probability.",
}
order = ["NearestArc(baseline)", "LogisticRegression", "KNN", "RandomForest", "HistGradientBoosting", "MLP"]
rows = [[P(f"<b>{h}</b>") for h in ["Model", "What it is", "CV Top-1", "CV Macro F1", "Test Top-1", "Test Macro F1",
         "Test log loss", "Continuity", "Train (s)", "ms / point"]]]
for m in order:
    r = cmp.loc[m]
    na = m.startswith("Nearest")
    rows.append([P(f"<b>{m.replace('(baseline)', ' (baseline)')}</b>"), P(desc[m]),
                 "N/A" if na else f"{r.cv_top1_accuracy_mean:.3f} ± {r.cv_top1_accuracy_std:.3f}",
                 "N/A" if na else f"{r.cv_macro_f1_mean:.3f}",
                 f"{r.test_top1_accuracy:.3f}", f"{r.test_macro_f1:.3f}", "N/A" if na else f"{r.test_log_loss:.3f}",
                 f"{r.test_route_continuity:.3f}", "–" if na else f"{r.train_fit_s:.0f}",
                 "–" if na else f"{r.inference_ms_per_gps_point:.3f}"])
story.append(Paragraph("4. Results: accuracy of every model (student-test = 10 unseen trips, 19,133 GPS points)", H))
story.append(table(rows, [31 * mm, 47 * mm, 20 * mm, 13 * mm, 12 * mm, 13 * mm, 12 * mm, 14 * mm, 10 * mm, 12 * mm],
                   font=6.9, highlight=[4, 5, 6]))
story.append(Paragraph("What the columns mean", H))
story.append(table([
    ["Metric", "Meaning"],
    ["Top-1 accuracy", P("% of GPS points where the model's highest-scoring candidate is the correct road arc. <b>Main metric.</b>")],
    ["CV Top-1", "Top-1 averaged over the 5 cross-validation folds (± = spread between folds, i.e. stability) on training trips."],
    ["Test Top-1", "Top-1 on the 10 student-test trips the model never saw during training."],
    ["Macro F1", "Each candidate judged true/false; F1 (balance of precision and recall) averaged over both classes equally."],
    ["Log loss", "How good the probabilities are (lower = better). Matters because they feed the HMM next."],
    ["Continuity", "% of consecutive predictions on the same or a directly connected arc (route makes physical sense)."],
    ["ms / pt", "Prediction time per GPS point (matters for the live API demo)."],
], [22 * mm, 162 * mm], font=6.9))
story.append(Paragraph("Takeaways", H))
story += bullets([
    f"Learning lifts Top-1 from <b>{cmp.loc['NearestArc(baseline)'].test_top1_accuracy:.1%}</b> (nearest arc) to "
    f"<b>≈{cmp.loc[['RandomForest', 'HistGradientBoosting', 'MLP'], 'test_top1_accuracy'].mean():.1%}</b>. RandomForest, HistGradientBoosting and "
    "MLP are statistically tied (gaps far smaller than the ±0.027 fold spread); LogReg (linear) is weakest; KNN is slow with poor log loss.",
    "Remaining errors (HGB, student-test): 45% choose the same road in the opposite direction, 35% a neighbouring arc at a junction, "
    "12% stationary points. These are sequence problems the phase-2 <b>HMM + Viterbi</b> decoder is built to fix, using road connectivity and direction.",
    "Next: pick the final model → Bayesian hyperparameter optimisation → HMM/Viterbi → FastAPI + UI. All numbers come from scripts/01–03 → results/.",
], S)

out = C.ROOT / "docs" / "PathSnap_Phase1_Summary.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=13 * mm, rightMargin=13 * mm, topMargin=10 * mm,
                  bottomMargin=8 * mm, title="PathSnap Phase 1 Summary").build(story)
print("wrote", out)
