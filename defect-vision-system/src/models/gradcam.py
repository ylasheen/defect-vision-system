"""
gradcam.py
----------
Grad-CAM implementation: produces a heatmap showing which pixels of an
input image most influenced the CNN's prediction. This is the visual
equivalent of SHAP for image models -- it answers "why did the model
flag this surface as defective?"
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_last_conv_layer(model: keras.Model) -> str:
    """Find the name of the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model.")


def make_gradcam_heatmap(img_array: np.ndarray, model: keras.Model, last_conv_layer_name: str = None,
                          pred_index: int = None) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a single preprocessed image batch (1, H, W, 3).
    Returns a 2D heatmap normalized to [0, 1].
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    # Rebuild a fresh functional graph by replaying each layer in order.
    # This is needed because models reloaded from a .keras file don't retain
    # the original call history, so `model.output` / `layer.output` raise
    # AttributeError until the graph is freshly constructed like this.
    #
    # We backprop from the pre-softmax "logits" rather than the final softmax
    # probabilities: once a confident model outputs ~1.0 / ~0.0, the softmax
    # gradient saturates to ~0 everywhere, which would make Grad-CAM blank.
    # Logits don't saturate, so gradients stay informative.
    inputs = keras.Input(shape=img_array.shape[1:])
    x = inputs
    conv_output_tensor = None
    logits_tensor = None
    final_tensor = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output_tensor = x
        if layer.name == "logits":
            logits_tensor = x
        final_tensor = x
    if logits_tensor is None:
        logits_tensor = final_tensor  # fallback for older saved models
    grad_model = keras.models.Model(inputs, [conv_output_tensor, logits_tensor, final_tensor])

    with tf.GradientTape() as tape:
        conv_output, logits, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = logits[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(heatmap: np.ndarray, original_img: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap (values in [0,1]) on top of the original RGB
    image (uint8, H x W x 3). Returns a uint8 RGB image.
    """
    import matplotlib.pyplot as plt

    h, w = original_img.shape[:2]
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (h, w)).numpy().squeeze()

    jet = plt.get_cmap("jet")
    jet_colors = jet(heatmap_resized)[:, :, :3]
    jet_colors = (jet_colors * 255).astype(np.uint8)

    overlaid = (jet_colors * alpha + original_img * (1 - alpha)).astype(np.uint8)
    return overlaid
