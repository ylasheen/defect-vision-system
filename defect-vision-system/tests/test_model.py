"""Unit tests for the CNN model architecture and Grad-CAM module."""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.train_model import build_model
from src.models.gradcam import make_gradcam_heatmap, find_last_conv_layer, overlay_heatmap


def test_build_model_output_shape():
    model = build_model(input_shape=(64, 64, 3), num_classes=2, learning_rate=0.001)
    dummy = np.random.randint(0, 255, size=(1, 64, 64, 3)).astype("float32")
    preds = model.predict(dummy, verbose=0)
    assert preds.shape == (1, 2)
    # softmax outputs should sum to ~1
    assert abs(preds.sum() - 1.0) < 1e-4


def test_find_last_conv_layer():
    model = build_model(input_shape=(64, 64, 3), num_classes=2, learning_rate=0.001)
    last_conv = find_last_conv_layer(model)
    assert "conv2d" in last_conv


def test_gradcam_heatmap_shape_and_range():
    model = build_model(input_shape=(64, 64, 3), num_classes=2, learning_rate=0.001)
    dummy = np.random.randint(0, 255, size=(1, 64, 64, 3)).astype("float32")
    heatmap = make_gradcam_heatmap(dummy, model)
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-5


def test_overlay_heatmap_shape():
    heatmap = np.random.rand(8, 8)
    original = np.random.randint(0, 255, size=(64, 64, 3)).astype("uint8")
    overlay = overlay_heatmap(heatmap, original)
    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8
