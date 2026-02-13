"""
model_loader.py — Thread-safe singleton loader for Grounding DINO.

Loads `AutoProcessor` and `AutoModelForZeroShotObjectDetection` once,
optionally casts to FP16 on CUDA, and exposes them through the
`ModelLoader` class.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import config

logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton that holds the Grounding DINO model and processor."""

    _instance: Optional["ModelLoader"] = None
    _lock: threading.Lock = threading.Lock()

    # ── public attributes (set after load) ──────────────────────────
    processor: AutoProcessor
    model: AutoModelForZeroShotObjectDetection
    device: str

    # ── singleton access ────────────────────────────────────────────
    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # double-checked locking
                    instance = super().__new__(cls)
                    instance._loaded = False
                    cls._instance = instance
        return cls._instance

    # ── loading ─────────────────────────────────────────────────────
    def load(self) -> "ModelLoader":
        """Download / cache and prepare the model.  Idempotent."""
        if self._loaded:
            logger.info("Model already loaded — skipping.")
            return self

        logger.info("Loading model  : %s", config.MODEL_ID)
        logger.info("Target device  : %s", config.DEVICE)
        logger.info("Half precision : %s", config.USE_HALF_PRECISION)

        self.device = config.DEVICE

        self.processor = AutoProcessor.from_pretrained(config.MODEL_ID)

        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config.MODEL_ID
        )

        if config.USE_HALF_PRECISION:
            self.model = self.model.half()
            logger.info("Model cast to FP16.")

        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info("Model loaded successfully.")
        return self

    # ── convenience ────────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        """Release GPU memory."""
        if self._loaded:
            del self.model
            del self.processor
            if self.device == "cuda":
                torch.cuda.empty_cache()
            self._loaded = False
            logger.info("Model unloaded and GPU cache cleared.")
