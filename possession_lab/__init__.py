"""Xplainable Soccer Predictor — Possession Lab.

Gap-aware hybrid HMM & multi-source fusion system for the BAR-ATH possession task.
"""

from __future__ import annotations

from .audio import WhistleData, detect_events, envelope, score_per_video_frame
from .balltrack import BallTrack, track_ball
from .config import (
    CODE_TO_LABEL,
    FPS,
    LABEL_TO_CODE,
    LABELS,
    MODEL_PARAMS,
    N_FRAMES,
    PITCH,
    A,
    D,
    H,
    X,
)
from .data import (
    Homography,
    TagTable,
    TrackingData,
    load_homography,
    load_public_gt,
    load_smart_tags,
    load_submission,
    load_tracking,
)
from .evaluate import evaluate_stress, full_match_stats, public_metrics
from .events import apply_event_overrides, build_events
from .explain import build_explanations, explain_prediction
from .features import Features, extract_features
from .fusion import FusionResult, build_fusion
from .model import (
    DEFAULT_PARAMS,
    DecodeResult,
    ModelParams,
    decode,
    forward_backward,
    merge_isolated_runs,
    viterbi,
)
from .perception import (
    VideoBallTracker,
    VideoPerception,
    load_perception,
    merge_perception,
    save_perception,
)
from .pipeline import (
    LabData,
    decode_public,
    load_lab,
    predict_full,
    validate_codes,
    validate_submission_file,
    write_submission,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Labels & constants
    "H",
    "A",
    "D",
    "X",
    "LABELS",
    "CODE_TO_LABEL",
    "LABEL_TO_CODE",
    "FPS",
    "N_FRAMES",
    "PITCH",
    "DEFAULT_PARAMS",
    "MODEL_PARAMS",
    # Data loaders
    "LabData",
    "load_lab",
    "TrackingData",
    "TagTable",
    "Homography",
    "load_tracking",
    "load_smart_tags",
    "load_homography",
    "load_public_gt",
    "load_submission",
    # Tracking
    "BallTrack",
    "track_ball",
    # Audio
    "WhistleData",
    "score_per_video_frame",
    "detect_events",
    "envelope",
    # Perception
    "VideoPerception",
    "VideoBallTracker",
    "load_perception",
    "merge_perception",
    "save_perception",
    # Features
    "Features",
    "extract_features",
    # Modeling & decoding
    "ModelParams",
    "DecodeResult",
    "decode",
    "viterbi",
    "forward_backward",
    "merge_isolated_runs",
    # Fusion
    "FusionResult",
    "build_fusion",
    # Events & Explainability
    "build_events",
    "apply_event_overrides",
    "explain_prediction",
    "build_explanations",
    # Evaluation & Pipeline
    "public_metrics",
    "evaluate_stress",
    "full_match_stats",
    "predict_full",
    "decode_public",
    "write_submission",
    "validate_codes",
    "validate_submission_file",
]
