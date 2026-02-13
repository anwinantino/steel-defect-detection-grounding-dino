"""
evaluator.py — Batch evaluation pipeline using Roboflow dataset.

Downloads the dataset from Roboflow via the SDK, runs Grounding DINO
inference on every image, and computes:
    * Per-image detection counts & classifications
    * Precision, Recall, mAP@0.5  (when ground truth available)
    * Exports results to CSV
    * Prints a confusion matrix
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from roboflow import Roboflow
from tqdm import tqdm

import config
from inference import DefectDetector, InferenceResult

logger = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────
@dataclass
class GTBox:
    """Ground truth bounding box from COCO annotation."""
    category: str
    bbox: List[float]  # [x, y, w, h] COCO format


@dataclass
class EvalImageResult:
    """Evaluation result for a single image."""
    filename: str
    defect_count: int
    status: str
    inference_time_ms: float
    detections: list = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0


# ── IoU helper ──────────────────────────────────────────────────────
def _iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Compute IoU between two boxes.
    box_a: [x1, y1, x2, y2]  (prediction format)
    box_b: [x, y, w, h]      (COCO ground-truth format → converted)
    """
    # Convert COCO [x,y,w,h] → [x1,y1,x2,y2]
    bx1 = box_b[0]
    by1 = box_b[1]
    bx2 = box_b[0] + box_b[2]
    by2 = box_b[1] + box_b[3]

    ax1, ay1, ax2, ay2 = box_a
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Roboflow dataset download ──────────────────────────────────────
def download_roboflow_dataset(
    api_key: str | None = None,
    workspace: str | None = None,
    project: str | None = None,
    version: int | None = None,
    export_format: str | None = None,
) -> Path:
    """
    Download the dataset from Roboflow and return the local path.

    Uses credentials from ``config`` by default; all can be overridden.
    """
    api_key = api_key or config.ROBOFLOW_API_KEY
    workspace = workspace or config.ROBOFLOW_WORKSPACE
    project_name = project or config.ROBOFLOW_PROJECT
    version_num = version or config.ROBOFLOW_VERSION
    export_format = export_format or config.ROBOFLOW_EXPORT_FORMAT

    logger.info("Connecting to Roboflow …")
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project_name)
    version_obj = project_obj.version(version_num)

    logger.info(
        "Downloading dataset: %s/%s v%d (format=%s) …",
        workspace, project_name, version_num, export_format,
    )
    dataset = version_obj.download(export_format)

    dataset_path = Path(dataset.location)
    logger.info("Dataset downloaded to: %s", dataset_path)
    return dataset_path


# ── Evaluator ───────────────────────────────────────────────────────
class Evaluator:
    """Batch evaluation engine."""

    def __init__(
        self,
        detector: DefectDetector,
        dataset_dir: str | Path | None = None,
        iou_threshold: float | None = None,
        download: bool = True,
    ) -> None:
        self.detector = detector
        self.iou_threshold = iou_threshold or config.EVALUATION_IOU_THRESHOLD
        self._annotations: Optional[dict] = None
        self._category_map: Dict[int, str] = {}
        self._download = download

        # If a local path was explicitly supplied, use it; otherwise
        # we'll download from Roboflow in ``run()``.
        self._explicit_dir = Path(dataset_dir) if dataset_dir else None
        self.dataset_dir: Path = self._explicit_dir or Path(".")

    # ── Loading ─────────────────────────────────────────────────────
    def _ensure_dataset(self) -> None:
        """Download the dataset from Roboflow if it hasn't been provided locally."""
        if self._download and self._explicit_dir is None:
            self.dataset_dir = download_roboflow_dataset()
            logger.info("Using downloaded dataset at: %s", self.dataset_dir)

    def _load_annotations(self) -> Optional[dict]:
        """Attempt to load a COCO-format _annotations.coco.json file."""
        candidates = list(self.dataset_dir.rglob("*annotations*.json"))
        if not candidates:
            logger.warning(
                "No COCO annotation file found in %s — "
                "evaluation will run without ground truth.",
                self.dataset_dir,
            )
            return None
        ann_path = candidates[0]
        logger.info("Loading annotations from %s", ann_path)
        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Build category id → name map
        for cat in data.get("categories", []):
            self._category_map[cat["id"]] = cat["name"]
        return data

    def _gt_for_image(self, image_id: int) -> List[GTBox]:
        """Return ground-truth boxes for a given image id."""
        if self._annotations is None:
            return []
        boxes: List[GTBox] = []
        for ann in self._annotations.get("annotations", []):
            if ann["image_id"] == image_id:
                cat = self._category_map.get(ann["category_id"], "unknown")
                boxes.append(GTBox(category=cat, bbox=ann["bbox"]))
        return boxes

    # ── Per-image matching ──────────────────────────────────────────
    def _match(
        self,
        result: InferenceResult,
        gt_boxes: List[GTBox],
    ) -> Tuple[int, int, int]:
        """Match predictions to ground truth → (TP, FP, FN)."""
        matched_gt = set()
        tp = 0
        for det in result.detections:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                iou_val = _iou(det.bbox, gt.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = idx
            if best_iou >= self.iou_threshold and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)

        fp = len(result.detections) - tp
        fn = len(gt_boxes) - tp
        return tp, fp, fn

    # ── Main ────────────────────────────────────────────────────────
    def run(self, output_csv: str | Path | None = None) -> Dict:
        """
        Evaluate all images in the dataset.

        If no local dataset directory was supplied, the Roboflow dataset
        is downloaded automatically before evaluation begins.

        Returns a dict with overall metrics.
        """
        # Step 1 — ensure dataset is available
        self._ensure_dataset()

        self._annotations = self._load_annotations()
        output_csv = Path(output_csv or config.EVALUATION_RESULTS_CSV)

        # Collect image paths
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = sorted(
            p for p in self.dataset_dir.rglob("*") if p.suffix.lower() in image_extensions
        )
        if not image_paths:
            logger.error("No images found in %s", self.dataset_dir)
            return {}

        logger.info("Found %d images for evaluation.", len(image_paths))

        # Build an image-filename → image-id map (from COCO data)
        fname_to_id: Dict[str, int] = {}
        if self._annotations:
            for img_info in self._annotations.get("images", []):
                fname_to_id[img_info["file_name"]] = img_info["id"]

        total_tp, total_fp, total_fn = 0, 0, 0
        all_results: List[EvalImageResult] = []
        confusion: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for img_path in tqdm(image_paths, desc="Evaluating", unit="img"):
            result = self.detector.detect(img_path)
            image_id = fname_to_id.get(img_path.name, -1)
            gt_boxes = self._gt_for_image(image_id) if image_id >= 0 else []

            tp, fp, fn = (0, 0, 0)
            prec, rec = 0.0, 0.0
            if gt_boxes:
                tp, fp, fn = self._match(result, gt_boxes)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Confusion bookkeeping
                has_gt_defect = len(gt_boxes) > 0
                actual_status = "DEFECT" if has_gt_defect else "NO DEFECT"
                pred_label = "DEFECT" if result.defect_count > 0 else "NO DEFECT"
                confusion[actual_status][pred_label] += 1

            eval_result = EvalImageResult(
                filename=img_path.name,
                defect_count=result.defect_count,
                status=result.status,
                inference_time_ms=result.inference_time_ms,
                detections=[
                    {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
                    for d in result.detections
                ],
                precision=round(prec, 4),
                recall=round(rec, 4),
            )
            all_results.append(eval_result)

        # Overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        # Simplified mAP@0.5 approximation (micro-average P×R)
        map_at_50 = overall_precision * overall_recall

        metrics = {
            "total_images": len(image_paths),
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "mAP@0.5": round(map_at_50, 4),
        }

        # Save CSV
        self._save_csv(all_results, output_csv)

        # Print confusion matrix
        self._print_confusion(confusion)

        # Print summary
        logger.info("=" * 55)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 55)
        for k, v in metrics.items():
            logger.info("  %-20s : %s", k, v)
        logger.info("=" * 55)

        return metrics

    # ── CSV export ──────────────────────────────────────────────────
    @staticmethod
    def _save_csv(results: List[EvalImageResult], path: Path) -> None:
        fieldnames = [
            "filename",
            "defect_count",
            "status",
            "inference_time_ms",
            "precision",
            "recall",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(
                    {
                        "filename": r.filename,
                        "defect_count": r.defect_count,
                        "status": r.status,
                        "inference_time_ms": r.inference_time_ms,
                        "precision": r.precision,
                        "recall": r.recall,
                    }
                )
        logger.info("Results saved to %s", path)

    # ── Confusion matrix ────────────────────────────────────────────
    @staticmethod
    def _print_confusion(confusion: Dict) -> None:
        if not confusion:
            logger.info("No ground-truth data — confusion matrix skipped.")
            return
        labels = sorted(
            {k for k in confusion} | {v for row in confusion.values() for v in row}
        )
        logger.info("\nConfusion Matrix:")
        header = f"{'':>15}" + "".join(f"{lbl:>15}" for lbl in labels)
        logger.info(header)
        for actual in labels:
            row = f"{actual:>15}"
            for pred in labels:
                row += f"{confusion[actual][pred]:>15}"
            logger.info(row)
