"""
inference.py — Grounding DINO inference engine.

Provides `DefectDetector` for single-image and batch inference, returning
structured detection results together with a severity classification.

Includes NMS deduplication and confidence-based filtering for accuracy.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchvision
from PIL import Image

import config
from model_loader import ModelLoader

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────
@dataclass
class Detection:
    """Single bounding-box detection."""

    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coords


@dataclass
class InferenceResult:
    """Aggregated output for one image."""

    detections: List[Detection] = field(default_factory=list)
    defect_count: int = 0
    status: str = config.SEVERITY_LABELS["none"]
    inference_time_ms: float = 0.0


# ── Detector ────────────────────────────────────────────────────────
class DefectDetector:
    """
    Zero-shot steel-surface defect detector backed by Grounding DINO.

    Usage
    -----
    >>> detector = DefectDetector()
    >>> result = detector.detect("path/to/image.jpg")
    >>> print(result.status, result.defect_count)
    """

    def __init__(
        self,
        prompt: Optional[str] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> None:
        loader = ModelLoader()
        if not loader.is_loaded:
            loader.load()

        self.model = loader.model
        self.processor = loader.processor
        self.device = loader.device

        self.prompt = prompt or config.PROMPT_TEXT
        self.box_threshold = box_threshold or config.BOX_THRESHOLD
        self.text_threshold = text_threshold or config.TEXT_THRESHOLD

    # ── public API ──────────────────────────────────────────────────
    def detect(self, source: Union[str, Path, np.ndarray, Image.Image]) -> InferenceResult:
        """Run detection on a single image and return `InferenceResult`."""
        image = self._load_image(source)
        return self._run(image)

    def detect_batch(self, sources: List[Union[str, Path]]) -> List[InferenceResult]:
        """Run detection on a list of image paths."""
        results: List[InferenceResult] = []
        for src in sources:
            results.append(self.detect(src))
        return results

    # ── internals ───────────────────────────────────────────────────
    @staticmethod
    def _load_image(source: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
        """Convert various input types to a PIL RGB Image."""
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        if isinstance(source, np.ndarray):
            return Image.fromarray(source[..., ::-1]).convert("RGB")  # BGR→RGB
        return Image.open(str(source)).convert("RGB")

    @staticmethod
    def _apply_nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: list,
        iou_threshold: float,
    ) -> tuple:
        """Apply Non-Maximum Suppression to remove overlapping detections."""
        if len(boxes) == 0:
            return boxes, scores, labels

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        keep = keep_indices.numpy()

        return boxes[keep], scores[keep], [labels[i] for i in keep]

    @torch.no_grad()
    def _run(self, image: Image.Image) -> InferenceResult:
        """Core inference routine with NMS post-processing."""
        t0 = time.perf_counter()

        # Tokenise + encode
        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
        ).to(self.device)

        # Forward pass
        outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=target_sizes,
        )[0]

        # Extract raw detections
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        # Apply NMS to remove duplicate / overlapping boxes
        boxes, scores, labels = self._apply_nms(
            boxes, scores, labels, config.NMS_IOU_THRESHOLD
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build detection list (sorted by confidence, highest first)
        detections: List[Detection] = []
        sort_idx = np.argsort(-scores)  # descending confidence
        for i in sort_idx:
            detections.append(
                Detection(
                    label=labels[i],
                    confidence=round(float(scores[i]), 4),
                    bbox=[round(float(c), 2) for c in boxes[i]],
                )
            )

        defect_count = len(detections)
        status = self._classify_severity(defect_count)

        logger.info(
            "Inference done in %.1f ms — %d detections → %s",
            elapsed_ms,
            defect_count,
            status,
        )

        return InferenceResult(
            detections=detections,
            defect_count=defect_count,
            status=status,
            inference_time_ms=round(elapsed_ms, 2),
        )

    # ── severity logic ──────────────────────────────────────────────
    @staticmethod
    def _classify_severity(count: int) -> str:
        """Map detection count to a severity label."""
        if count == 0:
            return config.SEVERITY_LABELS["none"]
        if count <= config.SEVERITY_MINOR_MAX:
            return config.SEVERITY_LABELS["minor"]
        return config.SEVERITY_LABELS["critical"]
