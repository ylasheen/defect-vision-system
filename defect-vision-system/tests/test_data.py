"""Unit tests for the synthetic image dataset generator."""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.make_dataset import generate_image, _base_surface


def test_base_surface_shape():
    rng = np.random.default_rng(0)
    img = _base_surface(64, rng)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_generate_good_image():
    rng = np.random.default_rng(1)
    img = generate_image(64, defective=False, rng=rng)
    assert img.size == (64, 64)


def test_generate_defective_image():
    rng = np.random.default_rng(2)
    img = generate_image(64, defective=True, rng=rng)
    assert img.size == (64, 64)


def test_defective_images_have_darker_pixels_on_average():
    """Defective images should contain darker (defect) pixels than clean ones,
    since scratches/dents/pits are drawn darker than the base surface."""
    rng_good = np.random.default_rng(42)
    rng_bad = np.random.default_rng(42)

    good_mins = []
    bad_mins = []
    for _ in range(10):
        good_img = np.array(generate_image(64, defective=False, rng=rng_good))
        bad_img = np.array(generate_image(64, defective=True, rng=rng_bad))
        good_mins.append(good_img.min())
        bad_mins.append(bad_img.min())

    assert np.mean(bad_mins) < np.mean(good_mins)
