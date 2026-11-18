"""Project-wide paths and fixed settings. Every value here is a documented decision (see docs/decisions.md)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MAPS = ROOT / "maps"

PARTITIONS = ("train", "student_test")

# D5 candidate generation: arcs within 50 m of a GPS point, keep the 10 nearest
CAND_RADIUS_M = 50.0
CAND_MAX = 10
DENSIFY_SPACING_M = 20.0  # arc sampling step used only to find nearby arcs fast

# D4 labels: monotonic DP alignment of GPS points to the reference route
ALIGN_MAX_STEP = 15        # max route arcs a label may advance between consecutive points
LABEL_MAX_DIST_M = 50.0    # labels farther than this from the GPS point are unreliable
LABEL_HEADING_PENALTY_M = 30.0  # D23: extra alignment cost for a route arc opposing the travel direction

# D6 bad-label trips excluded from TRAINING ONLY (kept in EDA):
# their reference route lies 60-140 m from the GPS trace for most of the trip.
EXCLUDED_TRAIN_TRIPS = {"00000033", "00000054"}

# D7 preprocessing
HEADING_WINDOW = 2         # heading/speed from points i-2 .. i+2
MOVING_SPEED_MS = 1.0      # below this the vehicle is treated as stationary
SPEED_CLIP_MS = 70.0       # GPS jumps above 252 km/h are clipped
GAP_S = 5.0                # time step above this is a GPS gap

SEED = 42
N_FOLDS = 5
