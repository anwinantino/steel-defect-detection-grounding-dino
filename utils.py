"""
utils.py — Visualization and helper utilities.

Provides functions to draw bounding boxes, labels, severity banners,
and FPS overlays on OpenCV frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

import config
from inference import Detection, InferenceResult

logger = logging.getLogger(__name__)


# ── Severity key helper ────────────────────────────────────────────
def _severity_key(status: str) -> str:
    """Map a human-readable status string back to a banner-colour key."""
    for key, label in config.SEVERITY_LABELS.items():
        if label == status:
            return key
    return "none"


# ── Drawing ────────────────────────────────────────────────────────
def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
    """
    Draw bounding boxes and labels with confidence on *frame* (in-place).
    """
    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det.bbox]
        label_text = f"{det.label} {det.confidence:.2f}"

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.BBOX_COLOR, 2)

        # Label background
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
        )
        cv2.rectangle(
            frame,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            config.BBOX_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label_text,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            config.TEXT_COLOR,
            config.FONT_THICKNESS,
        )
    return frame


def draw_severity_banner(
    frame: np.ndarray,
    result: InferenceResult,
) -> np.ndarray:
    """Draw a coloured status banner at the top of the frame."""
    key = _severity_key(result.status)
    colour = config.BANNER_COLORS.get(key, (128, 128, 128))

    banner_h = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_h), colour, cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    text = f"{result.status}  |  Detections: {result.defect_count}"
    cv2.putText(
        frame,
        text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        config.TEXT_COLOR,
        2,
    )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw an FPS counter in the bottom-left corner."""
    text = f"FPS: {fps:.1f}"
    h = frame.shape[0]
    cv2.putText(
        frame,
        text,
        (10, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    return frame


def annotate_frame(
    frame: np.ndarray,
    result: InferenceResult,
    fps: float | None = None,
) -> np.ndarray:
    """All-in-one annotation: boxes + banner + optional FPS."""
    draw_detections(frame, result.detections)
    draw_severity_banner(frame, result)
    if fps is not None:
        draw_fps(frame, fps)
    return frame


# ── I/O helpers ────────────────────────────────────────────────────
def save_annotated_image(
    frame: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Write an annotated frame to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    logger.info("Saved annotated image → %s", output_path)
    return output_path


def setup_logging() -> None:
    """Configure root logger according to config."""
    logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
