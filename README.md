# Steel Surface Defect Detection — Grounding DINO

Production-ready AI system for detecting steel surface defects using **IDEA-Research/grounding-dino-base** (zero-shot object detection).

| Feature | Details |
|---|---|
| **Model** | `IDEA-Research/grounding-dino-base` via Hugging Face Transformers |
| **Detection** | Cracks, rust, dents, scratches, holes, corrosion, pitted surfaces |
| **Classification** | `NO DEFECT` · `MINOR DEFECT` · `CRITICAL DEFECT` |
| **Modes** | Static image · Live camera · Batch evaluation · REST API |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU**: Install the CUDA-enabled PyTorch build for your system first:
> https://pytorch.org/get-started/locally/

### 2. Run on an image

```bash
python app.py --mode image --source test_images/sample.jpg
```

### 3. Run on a live camera

```bash
python app.py --mode camera
```

Press **q** to quit.

### 4. Batch evaluation

The dataset is downloaded automatically from Roboflow using the credentials in `config.py`:

```bash
python app.py --mode evaluate
```

Results are saved to `evaluation_results.csv`.

> **Override**: To evaluate a local directory instead, pass `--dataset path/to/local/`.

### 5. REST API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**POST** `/predict` — upload an image:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@test_images/sample.jpg"
```

Response:

```json
{
  "status": "CRITICAL DEFECT",
  "defect_count": 4,
  "inference_time_ms": 312.5,
  "detections": [
    {
      "label": "crack",
      "confidence": 0.87,
      "bbox": [120.0, 45.0, 380.0, 210.0]
    }
  ]
}
```

**GET** `/health` — liveness check.

---

## Project Structure

```
steel-defect-detection-grounding-dino/
├── app.py              # CLI entry point (image / camera / evaluate)
├── api.py              # FastAPI server (POST /predict)
├── config.py           # All configurable parameters + Roboflow credentials
├── model_loader.py     # Singleton model loader (CUDA / CPU)
├── inference.py        # Zero-shot detection + severity classification
├── camera.py           # Real-time camera feed with overlays
├── evaluator.py        # Roboflow download + batch evaluation
├── utils.py            # Visualization & helpers
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Configuration

All tunables live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_ID` | `IDEA-Research/grounding-dino-base` | HF model identifier |
| `BOX_THRESHOLD` | `0.30` | Minimum box confidence |
| `TEXT_THRESHOLD` | `0.25` | Minimum text-prompt confidence |
| `SEVERITY_MINOR_MAX` | `2` | Max detections for MINOR |
| `SEVERITY_CRITICAL_MIN` | `3` | Min detections for CRITICAL |
| `CAMERA_INDEX` | `0` | OpenCV camera device |
| `CAMERA_FRAME_SKIP` | `3` | Process every N-th frame |
| `USE_HALF_PRECISION` | auto | FP16 on CUDA, FP32 on CPU |

Override at runtime with CLI flags:

```bash
python app.py --mode image --source img.jpg --box-threshold 0.4 --text-threshold 0.3
```

---

## Docker

```bash
# Build
docker build -t steel-defect-detection .

# Run API
docker run --gpus all -p 8000:8000 steel-defect-detection

# Run CLI inside container
docker run --gpus all steel-defect-detection \
  python app.py --mode image --source test_images/sample.jpg
```

---

## Classification Logic

| Detections | Status |
|---|---|
| 0 | `NO DEFECT` |
| 1 – 2 | `MINOR DEFECT` |
| ≥ 3 | `CRITICAL DEFECT` |

Thresholds are configurable in `config.py`.

---

## Evaluation Metrics

When ground-truth annotations are available (COCO JSON):

- **Precision** — fraction of predicted boxes that match ground truth
- **Recall** — fraction of ground-truth boxes that are detected
- **mAP@0.5** — mean average precision at IoU ≥ 0.5
- **Confusion Matrix** — printed to console
- **CSV Export** — per-image results

---

## Architecture Flow

```
Image / Camera Input
  → Preprocessing (PIL / OpenCV)
    → Grounding DINO Inference (zero-shot)
      → Post-processing (NMS, threshold filtering)
        → Defect Severity Logic
          → Visualization (boxes, labels, banner)
            → Output (display / save / JSON API)
```

---

## License

MIT
