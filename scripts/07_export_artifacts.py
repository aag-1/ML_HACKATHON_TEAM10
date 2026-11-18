"""Step 7 - Export the inference artifacts loaded by the API (laptop 1).

  artifacts/model.joblib      tuned HistGradientBoosting (fitted on the 78 train trips)
  artifacts/hmm_params.json   decoder parameters chosen on train (beta, w, gap_s, v_stationary, d_off)
  artifacts/manifest.json     features, fixed preprocessing/candidate settings, metrics and library versions
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import sklearn

from pathsnap import config as C
from pathsnap.features import FEATURES

ART = C.ROOT / "artifacts"
ART.mkdir(exist_ok=True)
shutil.copyfile(C.ROOT / "models" / "HistGradientBoosting_tuned.joblib", ART / "model.joblib")
hmm = json.loads((C.RESULTS / "hmm_params.json").read_text())
(ART / "hmm_params.json").write_text(json.dumps({k: hmm[k] for k in ("beta", "w", "gap_s", "v_stationary", "d_off")}, indent=2))
bo = json.loads((C.RESULTS / "bayes_opt_best.json").read_text())
summary = json.loads((C.RESULTS / "hmm_summary.json").read_text())
(ART / "manifest.json").write_text(json.dumps({
    "model": "HistGradientBoostingClassifier (Bayesian-tuned)", "model_params": {**bo["fixed_params"], **bo["best_params"]},
    "features": FEATURES,
    "preprocessing": {"projection": "per-trip UTM (pyproj)", "heading_window": C.HEADING_WINDOW,
                      "moving_speed_ms": C.MOVING_SPEED_MS, "speed_clip_ms": C.SPEED_CLIP_MS, "gap_s": C.GAP_S},
    "candidates": {"radius_m": C.CAND_RADIUS_M, "max": C.CAND_MAX},
    "hmm": {k: hmm[k] for k in ("beta", "w", "gap_s", "v_stationary", "d_off")},
    "student_test": {"classifier_only_top1": summary["classifier_only_top1"], "hmm_top1": summary["hmm_top1"]},
    "sklearn": sklearn.__version__,
}, indent=2))
print("exported", sorted(p.name for p in ART.iterdir()))
