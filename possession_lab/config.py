"""Constants, labels, and paths for the possession lab.

The package is intentionally self-contained and does not import the starter notebooks.  The match
constants mirror ``games.py`` for BAR-ATH because this lab is written for that one fixture; the
gated data directory can still be relocated with ``MTAH_DATA_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
H, A, D, X = 0, 1, 2, 3
LABELS = ("H", "A", "D", "X")
LABEL_TO_CODE = {label: code for code, label in enumerate(LABELS)}
CODE_TO_LABEL = {code: label for label, code in LABEL_TO_CODE.items()}

# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------
GAME = "BAR-ATH"
SOURCE = "tactical"
FPS = 25
N_FRAMES = 199_319
PITCH = (105.3, 68.0)  # metres, corrected tracking pitch used by games.py
PUBLIC_START, PUBLIC_STOP = 11_185, 18_685  # half-open
GRADED_HALVES = ((11_185, 90_652), (110_808, 190_198))  # recording windows; active halves start at the Kick Off tag
CONTEXT = 500

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
PUBLIC_GT_PATH = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
VIDEO_PATH = DATA_DIR / SOURCE / f"{GAME}_{SOURCE}.mp4"
# Written by the local viewer only.  Never read by the model, features, or training code.
USER_TRUTH_PATH = OUTPUT_DIR / "user_truth.json"
# Generated switch annotations (explain.py).  Written by predict/explain, read by the viewer.
EXPLANATIONS_PATH = OUTPUT_DIR / "explanations.json"

# ---------------------------------------------------------------------------
# Referee whistle (tactical MP4 audio)
# ---------------------------------------------------------------------------
# The whistle is a quiet, tonal peak in the 2.4-3.8 kHz band.  Detection uses its relative
# spectral shape against the low-frequency crowd bed, not absolute loudness.  The score is cached
# so a normal predict/evaluate run never has to decode the 5 GB video again.
WHISTLE_CACHE = OUTPUT_DIR / "whistle.npz"
AUDIO_SR = 16_000
FFT_SIZE = 2_048
FFT_HOP = 512
BAND_LO, BAND_HI = 2_400.0, 3_800.0
LOW_LO, LOW_HI = 400.0, 1_800.0
W_MIN = 4.0                 # absolute floor for a candidate; confirmation is contextual
W_LOCAL_RATIO = 2.0         # candidate must stand this far above the local crowd median
MIN_EVENT_S = 0.12          # shortest accepted blast; real toots can be one STFT block
MERGE_GAP_S = 1.0           # join score dropouts inside one whistle, not separate toots
ENVELOPE_TAU_S = 0.6        # decay of the per-frame dead-ball support around an event

# ---------------------------------------------------------------------------
# Ball trajectory reconstruction
# ---------------------------------------------------------------------------
BALL_TRACK_MAX_FILL_S = 1.2     # never invent a ball position across a longer hole
BALL_TRACK_TAU_S = 0.9          # confidence decay with time since the last observation
BALL_TRACK_VELOCITY_S = 0.6     # look-back used to estimate velocity before a hole
BALL_TRACK_SPEED_CAP = 38.0     # m/s; anything above this is a tracking outlier, not a ball
BALL_TRACK_DAMPING = 0.35       # rolling/friction velocity decay, 1/s

# ---------------------------------------------------------------------------
# Goal geometry
# ---------------------------------------------------------------------------
GOAL_HALF_WIDTH = 3.66          # 7.32 m goal
GOAL_Y_MARGIN = 1.60            # posts plus the ball/annotation tolerance
GOAL_LOOKAHEAD_S = 3.0          # a shot must reach the line this quickly
GOAL_MIN_SPEED = 3.0            # m/s along the goal line
GOAL_MAX_DEFICIT = 12.0         # last observation must be this close to the goal line
GOAL_FALLBACK_GAP_S = 8.0       # a tagged goal with no usable trajectory must hide the ball this long

# Confirmed stoppage: a whistle or an out-of-play ball plus genuinely low player motion.  The low
# motion run is what separates a real stoppage from a crowd whistle during live play.
STOP_TEAM_SPEED = 1.4           # m/s, mean tracked player speed
STOP_MIN_LOW_S = 2.0            # the low-motion run must last this long
STOP_WHISTLE_LOOKBACK_S = 2.5   # a whistle may precede the players coming to rest
STOP_IN_RUN_WHISTLE_S = 5.0     # a resting field this long plus a whistle inside it is a stoppage
STOP_IN_RUN_BALL_SPEED = 2.0    # m/s; a visible ball moving faster than this is still in play
STOP_IN_RUN_ACTIVE_FRAMES = 25  # opening second of the resting run that must show no live ball
STOP_CONTROL_R = 3.0            # if a player is this close to a visible ball, a whistle is not a stop
STOP_RESTART_SPEED = 7.0        # a kicked restart ends a dead-ball interval
STOP_RESTART_CONTROL_R = 3.0
STOP_RESTART_FRAMES = 2
STOP_MIN_TRACKED_PLAYERS = 10   # fewer tracked players means the view is a cut/replay, not live play
STOP_VISIBILITY_GAP_S = 5.0     # a stopped game can lose the wide view this soon after the restart
STOP_LIVE_TAG_CODES = (
    "PROGRESSION",
    "ATTACKING TRANSITION",
)

# The exact parameters of the recommended model.  Kept here so the CLI, tests, and report agree.
MODEL_PARAMS = {
    "sigma": 4.0,
    "d_dead": 8.0,
    "dead_scale": 1.5,
    "edge0": 5.0,
    "edge_scale": 1.5,
    "w_edge": 0.5,
    "w_gap": 0.0,
    "gap0": 100.0,
    "gap_scale": 25.0,
    "w_out": 1.0,
    "w_dead": 0.02,
    "w_motion": 1.0,
    "motion0": 1.6,
    "motion_scale": 0.35,
    "motion_control_R": 4.0,
    "w_whistle": 0.0,
    "whistle_control_R": 3.0,
    "w_est_out": 0.0,
    "est_blend": 0.0,
    "est_sigma": 4.0,
    "flight_R": 0.0,
    "flight_speed": 10.0,
    "flight_p": 0.55,
    # Aerial-duel launch attribution (aerial.launch_track).  Ships at zero weight: the public
    # window contains no z coverage and the validated run must not move.  A positive weight
    # supports the frozen launching team on causal flight frames instead of trusting proximity.
    "w_air": 0.0,
    "w_tag": 0.0,
    "w_sp": 0.0,
    # Last-touch support.  A visible ball within ``lt_R`` of the player who last touched it is
    # usually still that team's possession, even when an opponent is marginally nearer.  Keep the
    # weight small: a larger value improves spell-F1 on the public window but invents a long H
    # spell after a single defensive touch at 15656.
    "w_lt": 0.1,
    "lt_R": 0.8,
    "p_stay": 0.995,
    "p_ha": 0.0005,
    "p_hd": 0.001,
    # Confirmed stoppages are hard overrides.  A less sticky D prior keeps the HMM from entering
    # D tens of frames before the whistle, while the event mask still guarantees the final D run.
    "p_dh": 0.007,
    "gate": 0.0,
    "loose_R": 7.0,
    "loose_speed": 2.0,
    "touch_age": 125,
    "touch_p": 0.7,
    "miss_mode": "hybrid",
    "sigma_miss": 6.0,
    "gap_thresh": 120.0,
    "min_players": 6,
    # Parked ball at a boundary: the strongest single dead-ball signature measured on the public
    # window (spellF1 0.6074 -> 0.6857, D-F1 -0.014).
    "w_parked_edge": 1.5,
    "loose_stop_gate": False,
}


def asset_paths(data_dir: str | os.PathLike[str] | None = None) -> dict[str, Path]:
    """Paths for the tactical ATD, tags, and homography.

    Missing files are not checked here.  ``pipeline.load_lab`` produces a useful message if the
    gated bundle has not been unpacked.
    """
    root = Path(data_dir) if data_dir is not None else DATA_DIR
    stem = f"{GAME}_{SOURCE}"
    folder = root / SOURCE
    return {
        "atd_raw": folder / f"{stem}_atd.txt",
        "atd_meta": folder / f"{stem}_atd.xml",
        "smart_tagging": folder / f"{stem}_smart_tagging.xml",
        "homography": folder / f"{stem}_homography.csv",
    }


def ensure_output_dir(path: str | os.PathLike[str] | None = None) -> Path:
    out = Path(path) if path is not None else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out
