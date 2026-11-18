"""Two-page plain-language study guide of the whole project. Numbers are read from results/.
Output: docs/PathSnap_Study_Guide.pdf
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

R = C.RESULTS
cmp = pd.read_csv(R / "model_comparison.csv").set_index("model")
tvu = pd.read_csv(R / "tuned_vs_untuned.csv").set_index("model")
bo = json.loads((R / "bayes_opt_best.json").read_text())
hs = json.loads((R / "hmm_summary.json").read_text())
ds = json.loads((R / "dataset_summary.json").read_text())
ev = pd.read_csv(R / "evaluator_frozen_run.csv", dtype={"trip": str})
pct = lambda x: f"{100 * x:.1f}%"
ev_pooled = (ev.hmm_top1 * ev.scored_points).sum() / ev.scored_points.sum()
ev_pooled_c = (ev.classifier_top1 * ev.scored_points).sum() / ev.scored_points.sum()

W = A4[0] - 24 * mm
T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14, leading=17)
H = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=10, leading=12, spaceBefore=6, spaceAfter=2,
                   textColor=colors.HexColor("#1f3b70"))
B = ParagraphStyle("b", fontName="Helvetica", fontSize=8.4, leading=10.6, spaceAfter=1.5)
S = ParagraphStyle("s", parent=B, fontSize=7.8, leading=9.4)
story = []


def p(t, st=B):
    story.append(Paragraph(t, st))


def bl(items):
    for t in items:
        story.append(Paragraph(f"&bull;&nbsp;{t}", B))


def tbl(rows, widths, hl=None):
    data = [[Paragraph(str(c), ParagraphStyle("c", parent=S, fontName="Helvetica-Bold" if i == 0 else "Helvetica"))
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths)
    st = [("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#b0b8c8")), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef8")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
          ("LEFTPADDING", (0, 0), (-1, -1), 2.5), ("RIGHTPADDING", (0, 0), (-1, -1), 2.5)]
    for r in hl or []:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff3d6")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 2))


m = lambda n, c: cmp.loc[n, c]
u, tu = tvu.loc["HGB untuned (phase 1)"], tvu.loc["HGB tuned (Bayesian)"]

story.append(Paragraph("PathSnap explained: everything we built, in plain language", T))
p("Read top to bottom. Each step says what we did, why, and the result. Technical words are in the glossary at the end.", S)

p("1. The problem", H)
p("A car's GPS records its position every second, but GPS is noisy: the dot wobbles 5-30 m, so it often lands on the "
  "wrong street. <b>Map-matching</b> means working out which road the car was really on at every second. We had 100 trips "
  "from around the world. Each trip comes with four files: the GPS dots (<i>.track</i>), map points (<i>.nodes</i>), road "
  "pieces joining two points (<i>.arcs</i>, one-way: a two-way street is two arcs, A&rarr;B and B&rarr;A) and the true "
  "route as a list of arcs (<i>.route</i>). <b>Key surprise:</b> every trip has its own map, so 'road 500' means something "
  "different in every trip. We could not just teach a model road numbers.")
tbl([["Data split", "Trips", "Used for"],
     ["Train", "80 (78 used)", "Learning, cross-validation, all tuning"],
     ["Student-test", "10", "Comparing models; reported once at the end"],
     ["Evaluator-test", "10", "Hidden exam set. Never trained or tuned on; run once after everything was frozen"]],
    [26 * mm, 22 * mm, W - 48 * mm])

p("2. Cleaning and preparing the data", H)
bl(["<b>Converted angles to metres</b> (each trip in its own UTM map projection). One degree east is not the same distance "
    "as one degree north, so distances in degrees are wrong.",
    "<b>Checked quality:</b> no missing values and time always moves forward. But there were pauses of up to 42 hours, GPS "
    "jumps of up to 180 m/s (impossible speeds, so we capped speed at 70 m/s), and two trips (33 and 54) whose 'true' "
    "route is over 100 m from the GPS. We removed those two from training.",
    "<b>Speed and direction</b> were worked out from the dots 2 seconds before and after each point, which averages out "
    "the wobble. A <b>moving flag</b> marks when the car really moves, because direction is meaningless when it's parked.",
    "<b>Labels (the right answers):</b> the route file says 'road A, then B, then C' but not which dot belongs to which road. "
    "We matched them with <b>dynamic programming</b>, which only moves forward along the route (like a car) and keeps the "
    "total distance smallest. A later fix: on roads driven both ways, it also checks the car's direction."])

p("3. Candidates and features: turning it into a yes/no question", H)
bl([f"For each GPS dot we list the roads within 50 m (at most 10), the <b>candidates</b>. That gives about "
    f"{ds['train']['mean_candidates_per_point']} per dot, and only one is right. The right road was missing from the list for "
    f"just {ds['train']['candidate_miss_rate_pct']}% of dots, so the best possible score is about 99.7%.",
    "For each dot-and-road pair we compute <b>13 measurements (features)</b> in three groups. <b>Geometry:</b> how far away "
    "the road is and whether the dot is near a junction. <b>Direction:</b> does the car's heading match the road's direction? "
    "This is the only way to tell A&rarr;B from B&rarr;A. <b>Comparison:</b> is this the closest option, and how much worse "
    "than the best? This works the same in a dense city or on a highway.",
    "The model then answers a <b>yes/no question</b> for every candidate: 'is this the right road?' It gives a probability, "
    "and we pick the highest (<b>Top-1</b>)."])

p("4. Training and comparing 5 models", H)
p("We used <b>5-fold cross-validation grouped by trip</b>: split the 78 trips into 5 groups, train on 4, test on the 5th, "
  "rotate, and average. Whole trips stay together, because dots one second apart are near-identical and splitting a trip "
  "would let the model cheat. Each model was then retrained on all 78 trips and tested once on the student-test trips.")
rows = [["Model", "What it is, in one line", "CV Top-1", "Test Top-1", "Log loss"]]
desc = {"NearestArc(baseline)": "Just pick the closest road. No learning.",
        "LogisticRegression": "Weighted sum of features turned into a probability. Can only learn straight-line rules.",
        "KNN": "Finds the 25 most similar past examples and lets them vote. Slow.",
        "RandomForest": "200 decision trees on random data slices; averages their votes.",
        "HistGradientBoosting": "Small trees built one after another, each fixing earlier mistakes.",
        "MLP": "Small neural network (13 &rarr; 64 &rarr; 32 &rarr; 1)."}
for n in desc:
    rows.append([n.replace("(baseline)", " (baseline)"), desc[n],
                 "-" if n.startswith("Nearest") else pct(m(n, "cv_top1_accuracy_mean")), pct(m(n, "test_top1_accuracy")),
                 "-" if n.startswith("Nearest") else f"{m(n, 'test_log_loss'):.3f}"])
tbl(rows, [33 * mm, W - 33 * mm - 54 * mm, 18 * mm, 18 * mm, 18 * mm], hl=[5])
p("Learning lifts accuracy from 54% to about 88%. The top three models are tied (their differences are smaller than the "
  "natural variation between trips). We chose <b>HistGradientBoosting</b>: the best probability quality (log loss), it trains "
  "25&times; faster than Random Forest, and its model file is tiny (Random Forest's is 127 MB, too big for GitHub).")

p("5. Bayesian hyperparameter optimisation", H)
p("Every model has settings (hyperparameters), such as how fast it learns or how big each tree is. Instead of trying them "
  "randomly, <b>Bayesian optimisation</b> (Optuna's TPE) learns from earlier attempts which settings look promising and tries "
  "those next. Weak attempts were stopped early (<b>pruning</b>). We ran "
  f"{bo['n_trials']} trials ({bo['trial_states'].get('COMPLETE', 0)} completed) and stopped once gains flattened. The best "
  f"was trial {bo['best_trial']}: a smaller, more cautious model. Student-test Top-1 went from {pct(u.test_top1)} to "
  f"{pct(tu.test_top1)} and log loss from {u.test_log_loss:.4f} to {tu.test_log_loss:.4f}: a small gain, as expected, because "
  "differences between trips matter far more than settings.")

p("6. HMM + Viterbi: using the whole trip, not one dot at a time", H)
p("The classifier looks at each dot alone, so it makes physically impossible jumps: the opposite direction one second, a "
  "side road the next. An <b>HMM (Hidden Markov Model)</b> adds common sense: the car can only move to a road it can actually "
  "reach by driving.")
bl(["<b>Hidden states</b> are the candidate roads for each dot. <b>Emission</b> means how much the classifier likes a "
    "candidate. <b>Transition</b> means how believable the move from road A to road B is: compare the driving distance along "
    "the map (one-way streets respected) with the straight-line distance between the two dots. If they're close, the move is "
    "believable; a detour or U-turn is penalised.",
    "<b>Viterbi</b> is an efficient method that finds the single most believable sequence of roads for the whole trip, without "
    "trying every combination.",
    "<b>Gaps / dead ends:</b> after a GPS pause over 60 s, or if no road can be reached, the decoder starts afresh, so one bad "
    "patch can't ruin the trip. <b>Parked-car fix:</b> a parked car's GPS jitters back and forth. We stopped counting that "
    "jitter as 'reversing', which had been making the decoder flip to the opposite direction.",
    "Decoder settings were tuned on train trips only, using classifier scores from models that had never seen those trips."])
p(f"<b>Result (student-test):</b> Top-1 {pct(hs['classifier_only_top1'])} &rarr; <b>{pct(hs['hmm_top1'])}</b>. Route makes "
  f"physical sense {pct(hs['classifier_only_continuity'])} &rarr; {pct(hs['hmm_continuity'])} of the time. It improved every "
  "trip.")

p("7. Final exam (evaluator trips) and why not 96%", H)
p("Before touching the hidden trips we <b>froze</b> the system and saved digital fingerprints (SHA-256) of every file, then "
  f"ran the trips once. Result: <b>{pct(ev_pooled)}</b> over all points ({pct(ev.hmm_top1.mean())} average of the 10 trips; "
  f"classifier alone {pct(ev_pooled_c)}). 5 of the 10 trips scored over 96%, the best 99.8%. Nothing was changed afterwards.")
p("<b>Why not 96% overall?</b> While the car is driving we score <b>about 98%</b>. The losses come from parked or crawling "
  "periods (about a quarter of points). There, the 'true answer' itself jumps between roads, because GPS jitter crosses road "
  "boundaries, so no model can predict it reliably. The brief warns that reference routes have limitations; we report this "
  "honestly instead of bending the numbers.")

p("8. The working system (two laptops)", H)
bl(["<b>Laptop 1, API (FastAPI):</b> loads the model and settings and holds the 10 evaluation trips. When asked for a trip, "
    "it runs the whole pipeline live, then scores the result against the true route on the server. The true route is never "
    "sent out. Every request prints a log line (the brief requires showing this).",
    "<b>Laptop 2, UI (a single web page with a map):</b> enter laptop 1's address, click Connect, and see the average accuracy "
    "banner and the trip buttons. Click a trip to see the GPS dots, the predicted route and the accuracy numbers."])

p("Glossary", H)
tbl([["Term", "Plain meaning"],
     ["Top-1 accuracy", "% of GPS dots where the model's first choice is the right road. Our main score."],
     ["Precision / Recall / F1", "When the model says 'yes', how often is it right / how many right roads did it find / a "
                                 "balance of both. 'Macro' treats the rare 'yes' class as equally important."],
     ["Log loss", "Is the model's confidence honest? Being 99% sure and wrong is punished heavily. Lower is better."],
     ["Cross-validation (GroupKFold)", "Rotate which trips are held out to get a reliable score; keep each trip whole."],
     ["Overfitting / underfitting", "Memorising the training data (great on train, poor on new data) / too simple to learn "
                                    "the pattern (poor on both). Checked with training curves and train-vs-test gaps."],
     ["Data leakage", "Test information sneaking into training and making scores look better than reality. Avoided by "
                      "grouping trips, fitting scalers on training folds only, and freezing before the evaluator run."],
     ["Class imbalance", "Only 1 in about 7 candidates is right. We didn't rebalance, so the probabilities stay honest for the HMM."],
     ["Route continuity", "% of consecutive predictions on the same or a connected road: does the route make sense?"]],
    [36 * mm, W - 36 * mm])

p("Likely viva questions, with one-line answers", H)
tbl([["Question", "Answer"],
     ["Why not predict the road ID directly?", "Every trip has its own map, so road IDs aren't shared between trips. We score nearby candidates instead."],
     ["How did you get a label for each GPS dot?", "Dynamic programming that moves forward along the true route, minimising distance, with a direction check on roads driven both ways."],
     ["Why convert to UTM / metres?", "Distances in degrees are distorted (east-west shrinks away from the equator), and the brief forbids them."],
     ["Why GroupKFold by trip?", "Neighbouring dots are near-identical. A random split leaks them between train and test and inflates the score."],
     ["Why exclude trips 33 and 54?", "Their 'true' route is 120-140 m from the GPS, so the labels are wrong. Excluded from training only, and documented."],
     ["Why HistGradientBoosting?", "Tied for best accuracy, best log loss, 25x faster than Random Forest, tiny model file."],
     ["What does Bayesian optimisation do?", "Learns from past trials which settings look promising and tries those next, instead of searching blindly."],
     ["What are emission and transition?", "Emission: how much the classifier likes a road for this dot. Transition: how believable driving from one road to the next is."],
     ["What does Viterbi do?", "Finds the single most believable sequence of roads for the whole trip, efficiently."],
     ["Did you touch the evaluator data?", "Only once, after freezing (fingerprints saved first). Never trained or tuned on it."],
     ["Why isn't accuracy 96%?", "About 98% while driving. The losses are parked periods where the reference itself jumps between roads with GPS jitter."]],
    [52 * mm, W - 52 * mm])

out = C.ROOT / "docs" / "PathSnap_Study_Guide.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=12 * mm, rightMargin=12 * mm, topMargin=10 * mm,
                  bottomMargin=9 * mm, title="PathSnap Study Guide").build(story)
print("wrote", out)
