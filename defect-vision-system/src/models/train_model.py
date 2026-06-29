"""
train_model.py
---------------
Trains a small CNN to classify product-surface images as 'good' or
'defective', tracks training/validation metrics, and saves the trained
model + class name mapping.
"""
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.utils.config import load_config, get_logger

logger = get_logger("train_model")
ROOT = Path(__file__).resolve().parents[2]


def build_model(input_shape, num_classes: int, learning_rate: float) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, name="logits"),
        layers.Activation("softmax", name="predictions"),
    ], name="defect_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    config = load_config()
    model_cfg = config["model"]
    processed_dir = ROOT / config["data"]["processed_dir"]
    target_size = tuple(model_cfg["target_size"])
    batch_size = model_cfg["batch_size"]

    tf.random.set_seed(model_cfg["random_state"])

    train_ds = keras.utils.image_dataset_from_directory(
        processed_dir / "train", image_size=target_size, batch_size=batch_size, label_mode="int", shuffle=True,
        seed=model_cfg["random_state"],
    )
    val_ds = keras.utils.image_dataset_from_directory(
        processed_dir / "val", image_size=target_size, batch_size=batch_size, label_mode="int", shuffle=False,
    )

    class_names = train_ds.class_names
    logger.info(f"Classes: {class_names}")

    input_shape = target_size + (3,)
    model = build_model(input_shape, num_classes=len(class_names), learning_rate=model_cfg["learning_rate"])
    model.summary(print_fn=lambda line: logger.info(line))

    early_stop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True)

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow_runs' / 'mlflow.db'}")
    (ROOT / "mlflow_runs").mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("defect-vision-system")

    with mlflow.start_run(run_name="defect_cnn"):
        mlflow.log_params({
            "epochs": model_cfg["epochs"],
            "batch_size": batch_size,
            "learning_rate": model_cfg["learning_rate"],
            "image_size": target_size[0],
            "architecture": "small_cnn_3conv",
        })

        history = model.fit(
            train_ds, validation_data=val_ds, epochs=model_cfg["epochs"],
            callbacks=[early_stop], verbose=2,
        )

        for epoch in range(len(history.history["accuracy"])):
            mlflow.log_metrics({
                "train_accuracy": history.history["accuracy"][epoch],
                "train_loss": history.history["loss"][epoch],
                "val_accuracy": history.history["val_accuracy"][epoch],
                "val_loss": history.history["val_loss"][epoch],
            }, step=epoch)

        best_val_acc = max(history.history["val_accuracy"])
        mlflow.log_metric("best_val_accuracy", best_val_acc)

    saved_model_path = ROOT / model_cfg["saved_model_path"]
    saved_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(saved_model_path)
    logger.info(f"Saved model -> {saved_model_path}")

    class_names_path = ROOT / model_cfg["class_names_path"]
    class_names_path.parent.mkdir(parents=True, exist_ok=True)
    with open(class_names_path, "w") as f:
        json.dump(class_names, f)
    logger.info(f"Saved class names -> {class_names_path}")

    final_val_acc = max(history.history["val_accuracy"])
    logger.info(f"Best validation accuracy: {final_val_acc:.4f}")

    history_path = ROOT / "reports" / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)
    logger.info(f"Saved training history -> {history_path}")


if __name__ == "__main__":
    main()
