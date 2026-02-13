"""
app.py — CLI entry point for the Steel Defect Detection System.

Modes
-----
  image     Run inference on a single image or a directory of images.
  camera    Start real-time defect detection from a live camera.
  evaluate  Batch-evaluate a dataset and produce metrics + CSV.

Usage
-----
  python app.py --mode image   --source test_images/sample.jpg
  python app.py --mode camera
  python app.py --mode evaluate --dataset dataset/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

import config
from camera import LiveCamera
from evaluator import Evaluator
from inference import DefectDetector
from model_loader import ModelLoader
from utils import annotate_frame, save_annotated_image, setup_logging

logger = logging.getLogger(__name__)


# ── CLI arguments ───────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Steel Surface Defect Detection using Grounding DINO",
    )
    parser.add_argument(
        "--mode",
        choices=["image", "camera", "evaluate"],
        required=True,
        help="Operating mode.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to an image or directory of images (for 'image' mode).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to a local dataset directory (skips Roboflow download).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=config.CAMERA_INDEX,
        help="Camera device index.",
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=config.BOX_THRESHOLD,
        help="Bounding-box confidence threshold.",
    )
    parser.add_argument(
        "--text-threshold",
        type=float,
        default=config.TEXT_THRESHOLD,
        help="Text-prompt confidence threshold.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for annotated image(s).",
    )
    return parser


# ── Mode handlers ───────────────────────────────────────────────────
def _run_image_mode(args: argparse.Namespace, detector: DefectDetector) -> None:
    """Process a single image or all images in a directory."""
    source = Path(args.source)
    if not source.exists():
        logger.error("Source path does not exist: %s", source)
        sys.exit(1)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    if source.is_file():
        paths = [source]
    else:
        paths = sorted(p for p in source.iterdir() if p.suffix.lower() in image_extensions)

    if not paths:
        logger.error("No images found at %s", source)
        sys.exit(1)

    for img_path in paths:
        logger.info("Processing: %s", img_path)
        result = detector.detect(img_path)

        # Print result
        print(f"\n{'='*50}")
        print(f"  Image   : {img_path.name}")
        print(f"  Status  : {result.status}")
        print(f"  Defects : {result.defect_count}")
        print(f"  Time    : {result.inference_time_ms:.1f} ms")
        for det in result.detections:
            print(f"    → {det.label:20s}  conf={det.confidence:.3f}  bbox={det.bbox}")
        print(f"{'='*50}\n")

        # Annotate and save / display
        frame = cv2.imread(str(img_path))
        if frame is not None:
            annotated = annotate_frame(frame, result)
            if args.output:
                out = Path(args.output)
                if out.is_dir():
                    out = out / f"annotated_{img_path.name}"
                save_annotated_image(annotated, out)
            else:
                cv2.imshow("Defect Detection", annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def _run_camera_mode(args: argparse.Namespace, detector: DefectDetector) -> None:
    cam = LiveCamera(detector, camera_index=args.camera_index)
    cam.run()


def _run_evaluate_mode(args: argparse.Namespace, detector: DefectDetector) -> None:
    # If user supplied --dataset, use it directly (no Roboflow download).
    # Otherwise, download from Roboflow automatically.
    if args.dataset:
        evaluator = Evaluator(detector, dataset_dir=args.dataset, download=False)
    else:
        evaluator = Evaluator(detector, download=True)
    evaluator.run()


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    logger.info("Initializing model …")
    ModelLoader().load()

    detector = DefectDetector(
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    mode_dispatch = {
        "image": _run_image_mode,
        "camera": _run_camera_mode,
        "evaluate": _run_evaluate_mode,
    }

    try:
        mode_dispatch[args.mode](args, detector)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception:
        logger.exception("Unhandled error — see traceback above.")
        sys.exit(1)
    finally:
        ModelLoader().unload()


if __name__ == "__main__":
    main()
