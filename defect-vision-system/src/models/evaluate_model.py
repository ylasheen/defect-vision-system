"""
evaluate_model.py
------------------
Evaluates the trained CNN on the held-out test set, generates Grad-CAM
sample visualizations, and translates classification performance into a
business cost report (cost of missed defects vs false alarms on a
manufacturing line).
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, get_logger
from src.models.gradcam import make_gradcam_heatmap, overlay_heatmap

logger = get_logger("evaluate_model")
ROOT = Path(__file__).resolve().parents[2]


def load_test_set(processed_dir: Path, class_names: list, target_size: tuple):
    images, labels, paths = [], [], []
    for idx, cls in enumerate(class_names):
        for f in sorted((processed_dir / "test" / cls).glob("*.png")):
            img = Image.open(f).convert("RGB").resize(target_size)
            images.append(np.array(img))
            labels.append(idx)
            paths.append(f)
    return np.array(images, dtype="float32"), np.array(labels), paths


def main():
    config = load_config()
    model_cfg = config["model"]
    biz = config["business"]

    model_path = ROOT / model_cfg["saved_model_path"]
    class_names_path = ROOT / model_cfg["class_names_path"]
    processed_dir = ROOT / config["data"]["processed_dir"]
    target_size = tuple(model_cfg["target_size"])

    logger.info(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    with open(class_names_path) as f:
        class_names = json.load(f)
    logger.info(f"Classes: {class_names}")

    X_test, y_test, paths = load_test_set(processed_dir, class_names, target_size)
    logger.info(f"Test set: {X_test.shape[0]} images")

    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"Accuracy: {report['accuracy']:.4f}")

    # ---- Business cost analysis -------------------------------------------------
    # 'defective' is assumed to be class index for "defective"
    defective_idx = class_names.index("defective")
    good_idx = class_names.index("good")

    # Missed defect: true=defective, predicted=good (false negative for defective)
    missed_defects = np.sum((y_test == defective_idx) & (y_pred == good_idx))
    # False alarm: true=good, predicted=defective (false positive for defective)
    false_alarms = np.sum((y_test == good_idx) & (y_pred == defective_idx))

    cost_missed = missed_defects * biz["cost_per_missed_defect"]
    cost_false_alarm = false_alarms * biz["cost_per_false_alarm"]
    total_cost = cost_missed + cost_false_alarm

    # Compare to a "no AI" baseline: manual inspection catches defects at ~85%
    # (typical human visual-inspection detection rate cited in QA literature)
    manual_catch_rate = 0.85
    n_test_defects = np.sum(y_test == defective_idx)
    manual_missed = n_test_defects * (1 - manual_catch_rate)
    manual_cost = manual_missed * biz["cost_per_missed_defect"]

    savings_vs_manual = manual_cost - cost_missed
    daily_extrapolation = (savings_vs_manual / len(y_test)) * biz["units_inspected_per_day"]

    logger.info(f"Missed defects: {missed_defects} | False alarms: {false_alarms}")
    logger.info(f"Estimated daily savings vs manual inspection: ${daily_extrapolation:,.2f}")

    # ---- Grad-CAM sample sheet ----------------------------------------------------
    figures_dir = ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(model_cfg["random_state"])
    sample_idx = rng.choice(len(X_test), size=min(8, len(X_test)), replace=False)

    fig, axes = plt.subplots(2, len(sample_idx), figsize=(2.2 * len(sample_idx), 4.6))
    for col, idx in enumerate(sample_idx):
        img = X_test[idx]
        batch = img[np.newaxis, ...]
        heatmap = make_gradcam_heatmap(batch, model)
        overlay = overlay_heatmap(heatmap, img.astype("uint8"))

        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]
        correct = "✓" if true_label == pred_label else "✗"

        axes[0, col].imshow(img.astype("uint8"))
        axes[0, col].axis("off")
        axes[0, col].set_title(f"{true_label}", fontsize=9)

        axes[1, col].imshow(overlay)
        axes[1, col].axis("off")
        axes[1, col].set_title(f"{correct} pred:{pred_label}", fontsize=8)

    plt.tight_layout()
    plt.savefig(figures_dir / "gradcam_samples.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Grad-CAM sample sheet -> {figures_dir / 'gradcam_samples.png'}")

    # ---- Confusion matrix plot -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ---- Reports --------------------------------------------------------------------
    reports_dir = ROOT / "reports"
    with open(reports_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    with open(reports_dir / "summary.md", "w") as f:
        f.write("# Defect Vision System — Business Summary\n\n")
        f.write(f"**Test accuracy:** {report['accuracy']:.2%}\n\n")
        f.write(f"**Missed defects (test set):** {missed_defects} / {n_test_defects}\n\n")
        f.write(f"**False alarms (test set):** {false_alarms}\n\n")
        f.write(f"**Estimated daily savings vs. manual inspection** "
                f"(at {biz['units_inspected_per_day']} units/day): "
                f"**${daily_extrapolation:,.2f}**\n\n")
        f.write("See `figures/gradcam_samples.png` for visual explainability and "
                "`figures/confusion_matrix.png` for the full confusion matrix.\n")

    logger.info(f"Saved business summary -> {reports_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
