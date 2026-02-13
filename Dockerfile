# ── Stage 1: base image with CUDA support ──────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Python dependencies ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: application code ──────────────────────────────────────
COPY . .

# Pre-download model weights into the image (optional — remove to
# download at first run instead).
# RUN python -c "from model_loader import ModelLoader; ModelLoader().load()"

# ── Expose API port ────────────────────────────────────────────────
EXPOSE 8000

# Default: run the FastAPI server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
