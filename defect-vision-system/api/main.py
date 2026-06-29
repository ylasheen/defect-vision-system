"""
FastAPI serving layer for the Defect Vision System.

Run with:
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from tensorflow import keras

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config
from src.models.gradcam import make_gradcam_heatmap, overlay_heatmap

ROOT = Path(__file__).resolve().parents[1]
config = load_config()
MODEL_PATH = ROOT / config["model"]["saved_model_path"]
CLASS_NAMES_PATH = ROOT / config["model"]["class_names_path"]
TARGET_SIZE = tuple(config["model"]["target_size"])

app = FastAPI(
    title="Defect Vision API",
    description="Classifies product-surface images as 'good' or 'defective' and "
                 "returns a Grad-CAM explainability heatmap.",
    version="1.0.0",
)

_model = None
_class_names = None


def get_model():
    global _model, _class_names
    if _model is None:
        _model = keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH) as f:
            _class_names = json.load(f)
    return _model, _class_names


@app.get("/")
def root():
    return {"status": "ok", "message": "Defect Vision API is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model, class_names = get_model()

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize(TARGET_SIZE)
    arr = np.array(img).astype("float32")
    batch = arr[np.newaxis, ...]

    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]

    return JSONResponse({
        "prediction": pred_label,
        "probabilities": {cls: float(p) for cls, p in zip(class_names, probs)},
        "confidence": float(probs[pred_idx]),
    })


@app.post("/predict_with_heatmap")
async def predict_with_heatmap(file: UploadFile = File(...)):
    """Same as /predict but also returns a base64-encoded Grad-CAM overlay image."""
    model, class_names = get_model()

    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB").resize(TARGET_SIZE)
    arr = np.array(img).astype("float32")
    batch = arr[np.newaxis, ...]

    probs = model.predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]

    heatmap = make_gradcam_heatmap(batch, model, pred_index=pred_idx)
    overlay = overlay_heatmap(heatmap, arr.astype("uint8"))

    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format="PNG")
    overlay_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return JSONResponse({
        "prediction": pred_label,
        "probabilities": {cls: float(p) for cls, p in zip(class_names, probs)},
        "confidence": float(probs[pred_idx]),
        "heatmap_overlay_base64": overlay_b64,
    })
