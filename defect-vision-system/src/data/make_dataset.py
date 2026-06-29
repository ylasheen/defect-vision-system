"""
make_dataset.py
----------------
Generates a statistically realistic synthetic image dataset simulating a
manufacturing surface-inspection camera: 'good' (clean surface) vs
'defective' (scratches / dents / pits) product images.

This keeps the project fully runnable offline, the same way the tabular
churn-intelligence-system project used a synthetic-but-realistic dataset.
Swap data/raw/<class>/ with real inspection photos any time -- the rest of
the pipeline (loader, training, Grad-CAM, API) is unaffected as long as the
folder-per-class structure is kept.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, get_logger

logger = get_logger("make_dataset")
ROOT = Path(__file__).resolve().parents[2]


def _base_surface(size: int, rng: np.random.Generator) -> Image.Image:
    """Create a brushed-metal-like base texture with subtle directional noise."""
    base_gray = rng.integers(120, 180)
    arr = np.full((size, size), base_gray, dtype=np.float32)

    # directional brushed-metal streaks
    streaks = rng.normal(0, 6, size=(size, 1)).repeat(size, axis=1)
    arr += streaks

    # fine grain noise
    arr += rng.normal(0, 5, size=(size, size))

    # soft vignette / lighting gradient
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = rng.uniform(0.3, 0.7) * size, rng.uniform(0.3, 0.7) * size
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    light = 18 * (1 - dist / dist.max())
    arr += light

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    return img


def _add_scratch(draw: ImageDraw.ImageDraw, size: int, rng: np.random.Generator):
    x1, y1 = rng.integers(0, size, 2)
    length = rng.integers(size // 4, size)
    angle = rng.uniform(0, np.pi)
    x2 = int(np.clip(x1 + length * np.cos(angle), 0, size - 1))
    y2 = int(np.clip(y1 + length * np.sin(angle), 0, size - 1))
    darkness = int(rng.integers(20, 70))
    width = int(rng.integers(1, 3))
    draw.line([(x1, y1), (x2, y2)], fill=(darkness, darkness, darkness), width=width)


def _add_dent(draw: ImageDraw.ImageDraw, size: int, rng: np.random.Generator):
    r = rng.integers(2, max(3, size // 10))
    cx, cy = rng.integers(r, size - r, 2)
    darkness = int(rng.integers(30, 90))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(darkness, darkness, darkness))


def _add_pit_cluster(draw: ImageDraw.ImageDraw, size: int, rng: np.random.Generator):
    cx, cy = rng.integers(size // 6, 5 * size // 6, 2)
    n_pits = rng.integers(4, 10)
    for _ in range(n_pits):
        ox, oy = rng.integers(-size // 8, size // 8, 2)
        r = rng.integers(1, 3)
        x, y = int(np.clip(cx + ox, r, size - r)), int(np.clip(cy + oy, r, size - r))
        darkness = int(rng.integers(40, 100))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(darkness, darkness, darkness))


def generate_image(size: int, defective: bool, rng: np.random.Generator) -> Image.Image:
    img = _base_surface(size, rng)

    if defective:
        draw = ImageDraw.Draw(img)
        n_defects = rng.integers(1, 4)
        for _ in range(n_defects):
            defect_type = rng.choice(["scratch", "dent", "pits"], p=[0.45, 0.35, 0.20])
            if defect_type == "scratch":
                _add_scratch(draw, size, rng)
            elif defect_type == "dent":
                _add_dent(draw, size, rng)
            else:
                _add_pit_cluster(draw, size, rng)

    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.6)))
    return img


def main():
    config = load_config()
    size = config["data"]["image_size"]
    n_per_class = config["data"]["n_images_per_class"]
    random_state = config["data"]["random_state"]
    classes = config["data"]["classes"]
    raw_dir = ROOT / config["data"]["raw_dir"]

    rng = np.random.default_rng(random_state)

    for cls in classes:
        cls_dir = raw_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating {n_per_class} '{cls}' images -> {cls_dir}")
        for i in range(n_per_class):
            img = generate_image(size, defective=(cls == "defective"), rng=rng)
            img.save(cls_dir / f"{cls}_{i:04d}.png")

    total = n_per_class * len(classes)
    logger.info(f"Done. Generated {total} images across {len(classes)} classes at {raw_dir}")


if __name__ == "__main__":
    main()
