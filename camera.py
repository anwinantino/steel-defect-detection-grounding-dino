"""
camera.py — Real-time live camera defect detection.

Opens an OpenCV video capture, performs Grounding DINO inference on
every N-th frame (configurable via ``config.CAMERA_FRAME_SKIP``), draws
annotated overlays, and displays an FPS counter.

Press **q** to quit.
"""

from __future__ import annotations

import logging
import time

import cv2

import config
from inference import DefectDetector, InferenceResult
from utils import annotate_frame

logger = logging.getLogger(__name__)


class LiveCamera:
    """Manage the live camera feed with real-time defect detection."""

    def __init__(
        self,
        detector: DefectDetector,
        camera_index: int | None = None,
        frame_skip: int | None = None,
    ) -> None:
        self.detector = detector
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.frame_skip = frame_skip if frame_skip is not None else config.CAMERA_FRAME_SKIP
        self._cap: cv2.VideoCapture | None = None

    # ── public ──────────────────────────────────────────────────────
    def run(self) -> None:
        """Start the camera loop.  Blocks until the user presses 'q'."""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            logger.error("Cannot open camera index %d", self.camera_index)
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        logger.info(
            "Camera %d opened — frame skip = %d.  Press 'q' to quit.",
            self.camera_index,
            self.frame_skip,
        )

        frame_count = 0
        last_result = InferenceResult()  # empty until first inference
        fps = 0.0
        prev_time = time.perf_counter()

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("Failed to read frame — exiting.")
                    break

                # Run inference only every N-th frame
                if frame_count % self.frame_skip == 0:
                    last_result = self.detector.detect(frame)

                # Compute FPS (wall-clock per displayed frame)
                now = time.perf_counter()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                # Draw annotations
                annotated = annotate_frame(frame, last_result, fps=fps)
                cv2.imshow("Steel Defect Detection — Live", annotated)

                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit key pressed.")
                    break

                frame_count += 1
        finally:
            self.release()

    def release(self) -> None:
        """Release the camera and destroy OpenCV windows."""
        if self._cap is not None:
            self._cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released.")
