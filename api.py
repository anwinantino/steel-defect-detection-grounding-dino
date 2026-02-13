"""
api.py — FastAPI wrapper for the Steel Defect Detection System.

Exposes
-------
  GET  /          Frontend UI (steel defect detection dashboard)
  POST /predict   Upload an image → JSON with detections + classification.
  GET  /health    Liveness check.

Run
---
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from inference import DefectDetector
from model_loader import ModelLoader
from utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ── Shared state ────────────────────────────────────────────────────
_detector: DefectDetector | None = None


# ── Lifespan (replaces deprecated on_event) ─────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global _detector
    logger.info("Loading model for API …")
    ModelLoader().load()
    _detector = DefectDetector()
    logger.info("API ready — open http://localhost:8000 in your browser")
    yield
    # Shutdown
    ModelLoader().unload()
    logger.info("API shut down.")


# ── FastAPI app ─────────────────────────────────────────────────────
app = FastAPI(
    title="Steel Defect Detection API",
    description="Zero-shot steel surface defect detection powered by Grounding DINO.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow frontend on any origin during dev) ──────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Response schemas ────────────────────────────────────────────────
class DetectionItem(BaseModel):
    label: str
    confidence: float
    bbox: List[float]


class PredictResponse(BaseModel):
    status: str
    defect_count: int
    inference_time_ms: float
    detections: List[DetectionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Endpoints ───────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Accept an image upload and return defect detections + classification.

    **Request**: multipart/form-data with a ``file`` field (JPEG/PNG).

    **Response**: JSON with ``status``, ``defect_count``, ``detections``.
    """
    if _detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    result = _detector.detect(image)

    return PredictResponse(
        status=result.status,
        defect_count=result.defect_count,
        inference_time_ms=result.inference_time_ms,
        detections=[
            DetectionItem(
                label=d.label,
                confidence=d.confidence,
                bbox=d.bbox,
            )
            for d in result.detections
        ],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness / readiness probe."""
    return HealthResponse(
        status="ok",
        model_loaded=ModelLoader().is_loaded,
    )


@app.get("/evaluation")
async def evaluation():
    """Return saved evaluation results (run evaluate.py first)."""
    eval_file = STATIC_DIR / "evaluation_results.json"
    if not eval_file.exists():
        raise HTTPException(status_code=404, detail="Evaluation not run yet. Run: python evaluate.py")
    import json
    with open(eval_file) as f:
        return json.load(f)


# ── Mount static files (after routes so /predict takes priority) ────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
