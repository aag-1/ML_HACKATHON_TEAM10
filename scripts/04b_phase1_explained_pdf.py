"""Annotated companion to the Phase-1 one-pager: every line of the summary followed by its plain-language meaning.

Inputs : results/eda_summary.json, results/dataset_summary.json, results/model_comparison.csv
Output : docs/PathSnap_Phase1_Summary_Explained.pdf
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
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

from pathsnap import config as C

eda = json.loads((C.RESULTS / "eda_summary.json").read_text())
ds = json.loads((C.RESULTS / "dataset_summary.json").read_text())
cmp = pd.read_csv(C.RESULTS / "model_comparison.csv").set_index("model")
tr, st, dtr, dst = eda["train"], eda["student_test"], ds["train"], ds["student_test"]
pct = lambda m: f"{100 * cmp.loc[m].test_top1_accuracy:.1f}%"

T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14, leading=17)
H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=8, spaceAfter=3,
                   textColor=colors.HexColor("#1f3b70"))
O = ParagraphStyle("o", fontName="Helvetica", fontSize=8.6, leading=10.8, spaceBefore=4)
L = ParagraphStyle("l", fontName="Helvetica-Oblique", fontSize=8.4, leading=10.6, leftIndent=12,
                   textColor=colors.HexColor("#2f6b3a"), backColor=colors.HexColor("#f1f8f2"), borderPadding=(2, 3, 2, 3),
                   spaceBefore=2, spaceAfter=2)

SECTIONS = [
    ("Title", [
        ("PathSnap: Data Preprocessing, Feature Engineering &amp; Model Training (Evaluation 1)",
         "This page covers the first half of the project: preparing the data, turning it into useful numbers for the computer, "
         "and teaching several models to pick the right road."),
    ]),
    ("Problem &amp; framing", [
        ("GPS is noisy (5–30 m), so the nearest road is often wrong. Goal: for every GPS point, find the road segment (arc) the vehicle was really on.",
         "Your phone's location dot wobbles around. The closest road to the dot is often a side street or the opposite lane. "
         "We want to work out which road the car was actually driving on, every second of the trip."),
        ("Every trip has its own road network (worldwide, 299–155,003 nodes), so arc IDs are not reusable class labels.",
         "Each trip comes with its own little map, from a different city or country. \"Road number 500\" in one trip has nothing to do with "
         "\"road number 500\" in another, so we cannot teach a model to memorise road numbers."),
        ("We therefore generate candidate arcs near each GPS point and train a binary classifier: P(this candidate is the true arc). "
         "The highest-probability candidate is the prediction.",
         "Instead, for every GPS dot we list the few roads nearby (the \"candidates\") and ask a yes/no question for each one: "
         "\"is this the road the car was on?\" The model gives each candidate a confidence score, and we pick the highest."),
    ]),
    ("1. Data preprocessing", [
        ("Partitions: train 80 trips, student-test 10 trips; evaluator trips never used.",
         "We learn from 80 trips, check ourselves on 10 other trips, and never touch the 10 trips the examiners keep for the final test. "
         "Like studying from the textbook, doing a mock exam, and not peeking at the real exam."),
        (f"Explored {tr['observations']:,} + {st['observations']:,} GPS points: {tr['missing_values_total']} missing values, timestamps always increasing, "
         f"median {tr['pct_steps_1s_median']}% of steps are 1 s, {tr['trips_with_gap_gt_5min']} trips with gaps &gt;5 min (max ≈42 h), "
         "GPS jumps up to 180 m/s, 11 reference routes with breaks.",
         "Before building anything we inspected the data. Nothing was missing and time always moves forward. The GPS records about once per second. "
         "But some trips have long pauses (one is almost two days), some dots jump impossibly fast (180 m/s is about 650 km/h), "
         "and some of the \"correct answer\" routes have small holes in them."),
        ("Projected lon/lat to metres with each trip's UTM zone, because distances in degrees are not valid.",
         "GPS gives positions as angles on a globe. We converted them to a flat local map measured in metres, so \"20 m away\" really means 20 metres. "
         "One degree east is not the same distance as one degree north, so measuring in degrees would be wrong."),
        ("Cleaned: flagged time gaps (&gt;5 s); smoothed speed and heading over ±2 points; capped speed at 70 m/s; added a moving flag (≥1 m/s); "
         "stopped vehicles keep their last heading.",
         "We marked moments when the GPS went silent. We worked out speed and travel direction using the 2 dots before and after each dot, "
         "which averages out the wobble. Impossible speeds were capped. We also noted whether the car was moving; when it is parked, the "
         "direction reading is junk, so we keep the direction it last had while driving."),
        ("Labelled: .route lists arcs, not one per GPS point, so each point was matched to the route by forward-only dynamic programming "
         "(minimum total distance; handles loops). Labels &gt;50 m from the point were treated as unreliable.",
         "The answer file only says \"the car drove road A, then B, then C\". It doesn't say which GPS dot belongs to which road. "
         "We matched them up with a step-by-step method that only moves forward along the route, the way a car does, and keeps the total distance "
         "as small as possible. This works even when the car drives the same street twice. If the matched road is more than 50 m away from the "
         "dot, we don't trust that answer."),
        (f"Excluded trips 33 and 54 from training (reference route 122–140 m from the GPS). Used: {dtr['observations_used_for_modelling']:,} train points "
         f"(78 trips), {dst['observations_used_for_modelling']:,} student-test points.",
         "In two trips the \"correct\" route is clearly wrong: it sits more than 100 m from where the GPS says the car was. Learning from those would "
         "teach the model mistakes, so we left them out of training (but still show them in our analysis)."),
        (f"Candidates: all arcs within 50 m, at most 10 per point (KD-tree): {dtr['mean_candidates_per_point']} per point, {dtr['pct_positive_rows']}% true. "
         f"Candidate miss rate {dtr['candidate_miss_rate_pct']}% train, {dst['candidate_miss_rate_pct']}% student-test, the accuracy ceiling.",
         "For each dot we shortlist every road within 50 m (at most 10), using a fast search structure (a KD-tree, like an index in a book). "
         "On average there are about 6 roads to choose from, and only one is right. The right road was missing from the shortlist only about "
         "3 times in 1,000, so the best score any model could get is about 99.7%."),
    ]),
    ("2. Feature engineering", [
        ("13 features per (GPS point, candidate arc), computed only from GPS + road map.",
         "For every dot-and-road pair we calculate 13 measurements that describe how well they fit. We only use the GPS and the map, never the "
         "answers, so the model can't cheat."),
        ("Geometry: dist, u, d_end, arc_len. How close the road is; where along it; how near a junction.",
         "Shape and distance: how far the dot is from the road, whether it falls at the start, middle or end of the road piece, how close it is to an "
         "intersection, and how long the road piece is."),
        ("Direction: heading_diff, cos_heading_diff, speed, moving. Separates A→B from B→A (same geometry).",
         "Direction of travel: does the car's direction match the road's direction? A two-way street is stored as two roads drawn on exactly the same "
         "line, one per direction, so direction is the only way to tell them apart. Speed and \"moving\" tell the model when the direction can be trusted."),
        ("Relative: dist_rank, dist_gap, n_cand, has_twin, heading_gap. How this candidate compares with the other candidates for the same point.",
         "Comparison with the other options: is this the closest road, or the 3rd closest? How much farther is it than the best one? How many options were "
         "there? Is the opposite-direction twin also on the list? Comparing works the same in a dense city and on an empty highway."),
    ]),
    ("3. Model training process", [
        ("Trained 5 models from different families plus a no-learning baseline (pick the nearest arc).",
         "We tried 5 different kinds of model, plus a dumb rule (\"always pick the closest road\"), to see how much real learning helps."),
        ("Scaling (StandardScaler) is used for LogReg / KNN / MLP inside a pipeline.",
         "Some models get confused when one measurement is in metres (0–50) and another is 0/1. Scaling puts all measurements on a similar range. "
         "It is learned only from the training data, so the test data stays unseen."),
        ("The class ratio is left as is (≈15% true) so probabilities stay realistic. Seed 42.",
         "Only about 1 in 7 candidates is the right road. We did not artificially balance this, so the model's confidence scores stay honest, which we need later. "
         "\"Seed 42\" fixes the randomness, so anyone re-running our code gets the same results."),
        ("5-fold GroupKFold by trip: the 78 trips are split into 5 groups; train on 4 and validate on 1, rotating.",
         "We split the training trips into 5 groups. We train on 4 groups and quiz the model on the 5th, then repeat so each group gets a turn as the quiz. "
         "Averaging the 5 quiz scores gives a reliable estimate."),
        ("Whole trips stay together, so neighbouring, near-identical GPS points can't leak between training and validation.",
         "Dots one second apart are almost identical. If one were in training and its neighbour in the quiz, the model would effectively have seen the answer. "
         "Keeping each trip entirely on one side prevents that."),
        ("Then each model is refit on all 78 trips and tested once on the 10 student-test trips.",
         "Finally each model learns from all training trips and takes one \"mock exam\" on 10 trips it has never seen."),
    ]),
    ("4. Results: accuracy of every model (student-test = 10 unseen trips, 19,133 GPS points)", [
        (f"NearestArc (baseline): Test Top-1 {pct('NearestArc(baseline)')}.",
         "Just choosing the closest road is right only about half the time. Mostly that is because of the two-way-street twins: both directions are "
         "equally close, so it is a coin flip."),
        (f"LogisticRegression: Test Top-1 {pct('LogisticRegression')}; log loss {cmp.loc['LogisticRegression'].test_log_loss:.3f}.",
         "Gives each measurement a weight and adds them up. Simple and fast, but it can only learn straight-line rules, e.g. it struggles with "
         "\"direction matters only when moving\". The weakest of the five."),
        (f"KNN: Test Top-1 {pct('KNN')}; log loss {cmp.loc['KNN'].test_log_loss:.3f}; {cmp.loc['KNN'].inference_ms_per_gps_point:.2f} ms per point.",
         "Answers by looking up the 25 most similar past examples and letting them vote. Decent accuracy, but slow, because it searches over a million "
         "stored examples every time. Its confidence scores are also rough (worst log loss)."),
        (f"RandomForest: Test Top-1 {pct('RandomForest')}; CV Top-1 {cmp.loc['RandomForest'].cv_top1_accuracy_mean:.3f}; trains in "
         f"{cmp.loc['RandomForest'].train_fit_s:.0f} s.",
         "200 decision trees (flowcharts of yes/no questions), each trained on a random slice of the data; their votes are averaged. Highest scores by a hair, "
         "but slow to train and the saved file is huge (127 MB)."),
        (f"HistGradientBoosting: Test Top-1 {pct('HistGradientBoosting')}; CV Top-1 {cmp.loc['HistGradientBoosting'].cv_top1_accuracy_mean:.3f}; trains in "
         f"{cmp.loc['HistGradientBoosting'].train_fit_s:.0f} s.",
         "Builds small trees one after another, each fixing the previous ones' mistakes. Same accuracy as the forest but 16 times faster to train and a tiny "
         "file, which makes it the easiest to fine-tune next."),
        (f"MLP: Test Top-1 {pct('MLP')}; log loss {cmp.loc['MLP'].test_log_loss:.3f}; {cmp.loc['MLP'].inference_ms_per_gps_point:.3f} ms per point.",
         "A small neural network: layers of simple maths units that learn patterns. Matches the top two and is the fastest at answering, but is a bit of a "
         "black box and sensitive to its settings."),
        ("Highlighted rows (RandomForest, HistGradientBoosting, MLP).",
         "These three are the best and are practically tied; one of them will be chosen as the final model."),
    ]),
    ("What the columns mean", [
        ("Top-1 accuracy: % of GPS points where the model's highest-scoring candidate is the correct road arc. Main metric.",
         "Out of every 100 GPS dots, how many times did the model's first choice turn out to be the right road? This is the score that matters most."),
        ("CV Top-1: Top-1 averaged over the 5 cross-validation folds (± = spread between folds, i.e. stability) on training trips.",
         "The average of the 5 quiz scores during training. The ± number shows how much the score jumped between quizzes; small means consistent."),
        ("Test Top-1: Top-1 on the 10 student-test trips the model never saw during training.",
         "The mock-exam score on completely new trips, the most honest estimate of real performance."),
        ("Macro F1: each candidate judged true/false; F1 (balance of precision and recall) averaged over both classes equally.",
         "Treats every candidate as a yes/no question and checks two things: when it says \"yes\", is it right (precision), and does it find the right "
         "roads (recall)? F1 combines both; \"macro\" gives the rare \"yes\" class equal importance."),
        ("Log loss: how good the probabilities are (lower = better). Matters because they feed the HMM next.",
         "Checks whether the model's confidence is honest. Being 99% sure and wrong is punished heavily. Lower is better. Phase 2 relies on these "
         "confidence scores, so they need to be trustworthy."),
        ("Continuity: % of consecutive predictions on the same or a directly connected arc.",
         "Does the predicted route make physical sense? If one second the car is on road A and the next on road B, B should connect to A. "
         "About 92% of steps do; the rest are \"teleports\" that phase 2 will fix."),
        ("ms / pt: prediction time per GPS point.",
         "How long the model takes to answer for one dot, in thousandths of a second. It matters because the examiner watches the live demo."),
    ]),
    ("Takeaways", [
        (f"Learning lifts Top-1 from {pct('NearestArc(baseline)')} (nearest arc) to ≈87.7%. RandomForest, HistGradientBoosting and MLP are statistically tied; "
         "LogReg (linear) is weakest; KNN is slow with poor log loss.",
         "Teaching a model takes us from guessing right half the time to right almost 9 times in 10. The top three models are equally good; the "
         "differences are smaller than normal run-to-run variation."),
        ("Remaining errors (HGB, student-test): 45% choose the same road in the opposite direction, 35% a neighbouring arc at a junction, 12% stationary points.",
         "When it is wrong, it is mostly picking the right street but the wrong direction, or the road just around the corner at an intersection, "
         "or it's confused while the car is standing still."),
        ("These are sequence problems the phase-2 HMM + Viterbi decoder is built to fix, using road connectivity and direction.",
         "Looking at one dot at a time can't fix those. Phase 2 looks at the whole trip at once and only allows routes a real car could drive "
         "(no U-turns through walls, no jumping between unconnected roads). That's where accuracy should rise towards the high 90s."),
        ("Next: pick the final model → Bayesian hyperparameter optimisation → HMM/Viterbi → FastAPI + UI. All numbers come from scripts/01–03 → results/.",
         "Next steps: choose the model, fine-tune its settings smartly, add the whole-trip route logic, then build the server and the demo website. "
         "Every number on this page was produced by our code and saved to files, so it can be checked."),
    ]),
]

story = [Paragraph("PathSnap Evaluation 1, explained line by line", T),
         Paragraph("Each line of the one-page summary (black) is followed by its plain-language meaning (green).", O), Spacer(1, 4)]
for head, pairs in SECTIONS:
    story.append(Paragraph(head, H))
    for orig, lay in pairs:
        story.append(KeepTogether([Paragraph(orig, O), Paragraph(f"In simple words: {lay}", L)]))

out = C.ROOT / "docs" / "PathSnap_Phase1_Summary_Explained.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=12 * mm,
                  title="PathSnap Phase 1 Summary, explained").build(story)
print("wrote", out)
