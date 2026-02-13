"""
config.py — Central configuration for the Steel Defect Detection System.

All tunable parameters (model, thresholds, prompts, camera, severity)
are defined here so that every other module imports from a single source
of truth.
"""

import torch

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
MODEL_ID: str = "IDEA-Research/grounding-dino-base"

# Auto-select accelerator
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Use half-precision on CUDA for faster inference & lower VRAM
USE_HALF_PRECISION: bool = torch.cuda.is_available()

# ──────────────────────────────────────────────
# Detection thresholds
# ──────────────────────────────────────────────
# Tuned for zero-shot mode — raised thresholds to reduce false positives
# while keeping enough sensitivity for genuine defects.
BOX_THRESHOLD: float = 0.25
TEXT_THRESHOLD: float = 0.20
NMS_IOU_THRESHOLD: float = 0.35  # tighter NMS to remove duplicate detections

# ──────────────────────────────────────────────
# Text prompt fed into Grounding DINO
# ──────────────────────────────────────────────
# Use ONLY the 6 NEU categories — no generic terms that cause false positives.
# Each phrase is kept short and specific to match Grounding DINO's text encoder.
PROMPT_TEXT: str = (
    "crazing. "
    "inclusion. "
    "patches. "
    "pitted surface. "
    "rolled-in scale. "
    "scratches."
)

# ──────────────────────────────────────────────
# Defect severity classification
# ──────────────────────────────────────────────
# Number-of-detections → severity mapping
SEVERITY_MINOR_MAX: int = 2   # 1–2 detections  → MINOR
SEVERITY_CRITICAL_MIN: int = 3  # ≥ 3 detections → CRITICAL
# 0 detections → NO DEFECT

SEVERITY_LABELS: dict = {
    "none": "NO DEFECT",
    "minor": "MINOR DEFECT",
    "critical": "CRITICAL DEFECT",
}

# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
CAMERA_INDEX: int = 0
CAMERA_FRAME_SKIP: int = 3  # process every N-th frame

# ──────────────────────────────────────────────
# Roboflow dataset
# ──────────────────────────────────────────────
ROBOFLOW_API_KEY: str = "mYBoLp62ZUUfrZdCQn6q"
ROBOFLOW_WORKSPACE: str = "antino-h3lcd"
ROBOFLOW_PROJECT: str = "neu-surface-defect-dataset-ikvj9"
ROBOFLOW_VERSION: int = 1
ROBOFLOW_EXPORT_FORMAT: str = "coco"

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
EVALUATION_IOU_THRESHOLD: float = 0.30  # lowered for zero-shot (boxes won't perfectly align)
EVALUATION_RESULTS_CSV: str = "evaluation_results.csv"

# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
BBOX_COLOR: tuple = (0, 255, 0)       # BGR green
TEXT_COLOR: tuple = (255, 255, 255)    # BGR white
BANNER_COLORS: dict = {
    "none": (0, 180, 0),       # green
    "minor": (0, 180, 255),    # orange
    "critical": (0, 0, 220),   # red
}
FONT_SCALE: float = 0.55
FONT_THICKNESS: int = 2

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
