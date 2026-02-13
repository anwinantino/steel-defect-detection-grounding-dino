"""
evaluate.py — Evaluate Grounding DINO against the NEU COCO validation set.

Computes:
  • Detection rate (did the model find ≥1 defect per image?)
  • Per-class metrics (precision, recall, F1, AP)
  • Overall mAP@0.50
  • Confusion-style stats
  • Average inference time

Results are saved to  static/evaluation_results.json  so the frontend can
display them.

Usage
-----
  python evaluate.py                    # full evaluation (all 360 images)
  python evaluate.py --limit 20        # quick test on 20 images
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

import config
from inference import DefectDetector
from model_loader import ModelLoader
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

DATASET_DIR = Path("NEU-Surface-Defect-Dataset-1/valid")
ANNO_FILE = DATASET_DIR / "_annotations.coco.json"
OUTPUT_FILE = Path("static/evaluation_results.json")

# ── Label normalisation ──────────────────────────────────────────────
# Grounding DINO returns free-form text labels.  We map them to the 6
# canonical NEU categories so we can compare fairly.
CANONICAL = ["crazing", "inclusion", "patches", "pitted_surface",
             "rolled-in_scale", "scratches"]

LABEL_MAP: dict[str, str] = {}
for c in CANONICAL:
    LABEL_MAP[c] = c
    LABEL_MAP[c.replace("_", " ")] = c
    LABEL_MAP[c.replace("-", " ")] = c
# Extra mappings for common Grounding DINO outputs
LABEL_MAP.update({
    "pitted surface":   "pitted_surface",
    "pitted surface on steel": "pitted_surface",
    "pitted":           "pitted_surface",
    "rolled in scale":  "rolled-in_scale",
    "rolled-in scale":  "rolled-in_scale",
    "rolled in scale on steel": "rolled-in_scale",
    "rolled-in scale on steel": "rolled-in_scale",
    "crazing on steel surface": "crazing",
    "crazing on steel": "crazing",
    "inclusion defect on steel": "inclusion",
    "inclusion defect": "inclusion",
    "patches on steel surface": "patches",
    "patches on steel": "patches",
    "scratches on steel surface": "scratches",
    "scratches on steel": "scratches",
    "steel surface crack": "scratches",
    "surface crack":    "scratches",
    "crack":            "scratches",
    "rust":             "inclusion",
    "corrosion":        "inclusion",
    "hole":             "pitted_surface",
    "dent":             "pitted_surface",
    "surface defect":   None,  # too vague to map
    "steel surface":    None,
    "surface steel":    None,
    "surface steel steel": None,
    "steel":            None,
    "surface":          None,
    "patches surface steel": "patches",
})


def normalise_label(raw: str) -> str | None:
    """Map a raw Grounding DINO label to a canonical NEU class, or None."""
    raw_lower = raw.strip().lower()
    if raw_lower in LABEL_MAP:
        return LABEL_MAP[raw_lower]
    # Substring matching fallback
    for canonical in CANONICAL:
        if canonical.replace("_", " ") in raw_lower or canonical.replace("-", " ") in raw_lower:
            return canonical
    return None


def compute_iou(box_a: list, box_b: list) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def coco_bbox_to_xyxy(bbox: list) -> list:
    """Convert COCO [x, y, w, h] → [x1, y1, x2, y2]."""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def compute_ap(precisions: list, recalls: list) -> float:
    """Compute Average Precision using 11-point interpolation."""
    if not precisions or not recalls:
        return 0.0
    # Add sentinel values
    recalls = [0.0] + recalls + [1.0]
    precisions = [1.0] + precisions + [0.0]
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = 0.0
        for r, pr in zip(recalls, precisions):
            if r >= t:
                p = max(p, pr)
        ap += p / 11.0
    return ap


def run_evaluation(limit: int | None = None):
    """Run full evaluation and return results dict."""
    # ── Load annotations ─────────────────────────────────────────────
    with open(ANNO_FILE) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {im["id"]: im for im in coco["images"]}

    # Build per-image ground-truth
    gt_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        cat_name = cat_map[ann["category_id"]]
        if cat_name == "surface-defects":
            continue  # skip the parent category
        gt_by_image[ann["image_id"]].append({
            "category": cat_name,
            "bbox": coco_bbox_to_xyxy(ann["bbox"]),
        })

    images = list(coco["images"])

    # ── Stratified sampling when --limit is used ─────────────────────
    # Pick equal images per class so every defect type is represented.
    if limit and limit < len(images):
        import random
        random.seed(42)  # reproducible

        # Group images by their dominant GT class
        class_to_images: dict[str, list] = defaultdict(list)
        for img_info in images:
            gt_list = gt_by_image.get(img_info["id"], [])
            if gt_list:
                # Use the first GT annotation's class as the "dominant" class
                dominant = gt_list[0]["category"]
                class_to_images[dominant].append(img_info)

        per_class_count = max(1, limit // len(CANONICAL))
        sampled = []
        for cls in CANONICAL:
            pool = class_to_images.get(cls, [])
            random.shuffle(pool)
            sampled.extend(pool[:per_class_count])

        # Fill remaining slots if limit wasn't evenly divisible
        remaining = limit - len(sampled)
        if remaining > 0:
            sampled_ids = {im["id"] for im in sampled}
            extras = [im for im in images if im["id"] not in sampled_ids]
            random.shuffle(extras)
            sampled.extend(extras[:remaining])

        images = sampled
        logger.info(f"Stratified sample: {per_class_count} images/class "
                    f"→ {len(images)} total")

    total_images = len(images)
    logger.info(f"Evaluating on {total_images} images …")

    # ── Load model ───────────────────────────────────────────────────
    ModelLoader().load()
    detector = DefectDetector()

    # ── Per-class tracking ───────────────────────────────────────────
    # For each class: list of (confidence, is_tp) tuples
    all_detections: dict[str, list] = {c: [] for c in CANONICAL}
    gt_counts: dict[str, int] = {c: 0 for c in CANONICAL}

    detection_rate_hits = 0
    total_inference_ms = 0.0
    per_image_results = []

    for idx, img_info in enumerate(images):
        img_path = DATASET_DIR / img_info["file_name"]
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")
        result = detector.detect(image)
        total_inference_ms += result.inference_time_ms

        # Ground truth for this image
        gt_list = gt_by_image.get(img_info["id"], [])
        gt_matched = [False] * len(gt_list)

        # Count GT per class
        for gt in gt_list:
            if gt["category"] in gt_counts:
                gt_counts[gt["category"]] += 1

        # Did model detect anything?
        has_gt = len(gt_list) > 0
        has_pred = result.defect_count > 0
        if has_gt and has_pred:
            detection_rate_hits += 1
        elif not has_gt and not has_pred:
            detection_rate_hits += 1

        # Match predictions to GT
        for det in result.detections:
            norm_label = normalise_label(det.label)
            if norm_label is None:
                continue  # unmappable label

            is_tp = False
            best_iou = 0.0
            best_gt_idx = -1

            for gi, gt in enumerate(gt_list):
                if gt_matched[gi]:
                    continue
                if gt["category"] != norm_label:
                    continue
                iou = compute_iou(det.bbox, gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_iou >= config.EVALUATION_IOU_THRESHOLD and best_gt_idx >= 0:
                is_tp = True
                gt_matched[best_gt_idx] = True

            all_detections[norm_label].append((det.confidence, is_tp))

        progress = f"[{idx+1}/{total_images}]"
        logger.info(f"{progress} {img_info['file_name']} → "
                    f"{result.defect_count} preds, {len(gt_list)} GT, "
                    f"{result.inference_time_ms:.0f}ms")

    # ── Compute metrics ──────────────────────────────────────────────
    per_class = {}
    all_ap = []

    for cls in CANONICAL:
        dets = sorted(all_detections[cls], key=lambda x: -x[0])  # sort by conf desc
        total_gt = gt_counts[cls]
        tp_count = sum(1 for _, tp in dets if tp)
        fp_count = len(dets) - tp_count

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Compute AP
        precisions_list = []
        recalls_list = []
        running_tp = 0
        running_fp = 0
        for conf, is_tp in dets:
            if is_tp:
                running_tp += 1
            else:
                running_fp += 1
            precisions_list.append(running_tp / (running_tp + running_fp))
            recalls_list.append(running_tp / total_gt if total_gt > 0 else 0.0)

        ap = compute_ap(precisions_list, recalls_list)
        all_ap.append(ap)

        per_class[cls] = {
            "ground_truth_count": total_gt,
            "true_positives": tp_count,
            "false_positives": fp_count,
            "missed": total_gt - tp_count,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "ap": round(ap, 4),
        }

    mean_ap = float(np.mean(all_ap)) if all_ap else 0.0
    avg_time = total_inference_ms / total_images if total_images > 0 else 0.0

    total_tp = sum(v["true_positives"] for v in per_class.values())
    total_fp = sum(v["false_positives"] for v in per_class.values())
    total_gt_all = sum(v["ground_truth_count"] for v in per_class.values())
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / total_gt_all if total_gt_all > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0.0

    results = {
        "model": config.MODEL_ID,
        "mode": "zero-shot (pre-trained, no fine-tuning)",
        "dataset": "NEU Surface Defect Dataset (validation split)",
        "total_images": total_images,
        "total_ground_truth": total_gt_all,
        "thresholds": {
            "box_threshold": config.BOX_THRESHOLD,
            "text_threshold": config.TEXT_THRESHOLD,
            "nms_iou_threshold": config.NMS_IOU_THRESHOLD,
            "eval_iou_threshold": config.EVALUATION_IOU_THRESHOLD,
        },
        "overall": {
            "mAP_50": round(mean_ap, 4),
            "precision": round(overall_precision, 4),
            "recall": round(overall_recall, 4),
            "f1_score": round(overall_f1, 4),
            "detection_rate": round(detection_rate_hits / total_images, 4) if total_images > 0 else 0,
            "avg_inference_ms": round(avg_time, 1),
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_missed": total_gt_all - total_tp,
        },
        "per_class": per_class,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Save ─────────────────────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model       : {config.MODEL_ID}")
    print(f"  Images      : {total_images}")
    print(f"  mAP@0.50    : {mean_ap:.4f}")
    print(f"  Precision   : {overall_precision:.4f}")
    print(f"  Recall      : {overall_recall:.4f}")
    print(f"  F1 Score    : {overall_f1:.4f}")
    print(f"  Det. Rate   : {detection_rate_hits}/{total_images}")
    print(f"  Avg Time    : {avg_time:.0f} ms / image")
    print("-" * 60)
    print(f"  {'Class':<20} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AP':>6} {'TP':>5} {'FP':>5} {'Miss':>5}")
    print("-" * 60)
    for cls in CANONICAL:
        m = per_class[cls]
        print(f"  {cls:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1_score']:>6.3f} {m['ap']:>6.3f} "
              f"{m['true_positives']:>5} {m['false_positives']:>5} {m['missed']:>5}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Grounding DINO on NEU dataset")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of images to evaluate (for quick tests)")
    args = parser.parse_args()
    run_evaluation(limit=args.limit)
