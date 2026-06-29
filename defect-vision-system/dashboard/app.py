"""
Streamlit Dashboard — Defect Vision System
--------------------------------------------
Run with:
    streamlit run dashboard/app.py

Lets a non-technical user upload a product-surface photo and see a live
defect prediction plus a Grad-CAM heatmap explaining the decision.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils.config import load_config
from src.models.gradcam import make_gradcam_heatmap, overlay_heatmap

st.set_page_config(page_title="Defect Vision System", page_icon="🔍", layout="wide")

config = load_config()


@st.cache_resource
def load_model_and_classes():
    model = keras.models.load_model(ROOT / config["model"]["saved_model_path"])
    with open(ROOT / config["model"]["class_names_path"]) as f:
        class_names = json.load(f)
    return model, class_names


@st.cache_data
def load_training_history():
    path = ROOT / "reports" / "training_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data
def load_classification_report():
    path = ROOT / "reports" / "classification_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


model, class_names = load_model_and_classes()
target_size = tuple(config["model"]["target_size"])

st.sidebar.title("🔍 Defect Vision")
st.sidebar.caption("Built by Youssef Lasheen — AI & ML Engineer")
page = st.sidebar.radio("Navigate", ["Live Inspector", "Model Performance", "Business Impact"])

# ---------------------------------------------------------------------------
# PAGE: Live Inspector
# ---------------------------------------------------------------------------
if page == "Live Inspector":
    st.title("Visual Quality Inspection System")
    st.caption(
        "Upload a product-surface photo. The CNN classifies it as good/defective "
        "and Grad-CAM highlights exactly which pixels drove the decision."
    )

    uploaded = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    sample_col1, sample_col2 = st.columns(2)
    use_sample = None
    with sample_col1:
        if st.button("Try a sample 'good' image", use_container_width=True):
            use_sample = "good"
    with sample_col2:
        if st.button("Try a sample 'defective' image", use_container_width=True):
            use_sample = "defective"

    img = None
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
    elif use_sample is not None:
        sample_dir = ROOT / "data" / "processed" / "test" / use_sample
        sample_files = sorted(sample_dir.glob("*.png"))
        if sample_files:
            idx = np.random.randint(0, len(sample_files))
            img = Image.open(sample_files[idx]).convert("RGB")

    if img is not None:
        img_resized = img.resize(target_size)
        arr = np.array(img_resized).astype("float32")
        batch = arr[np.newaxis, ...]

        probs = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        heatmap = make_gradcam_heatmap(batch, model, pred_index=pred_idx)
        overlay = overlay_heatmap(heatmap, arr.astype("uint8"))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_resized, caption="Input image", use_container_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM — what the model is looking at", use_container_width=True)
        with col3:
            if pred_label == "defective":
                st.error(f"🔴 DEFECTIVE  ({confidence:.1%} confidence)")
            else:
                st.success(f"🟢 GOOD  ({confidence:.1%} confidence)")

            st.markdown("**Class probabilities**")
            for cls, p in zip(class_names, probs):
                st.progress(float(p), text=f"{cls}: {p:.1%}")

    else:
        st.info("Upload an image above, or click one of the sample buttons to try the system.")

# ---------------------------------------------------------------------------
# PAGE: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("📈 Model Performance")

    history = load_training_history()
    report = load_classification_report()

    if history:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history["accuracy"], name="Train accuracy"))
            fig.add_trace(go.Scatter(y=history["val_accuracy"], name="Val accuracy"))
            fig.update_layout(title="Accuracy over training", xaxis_title="Epoch", yaxis_title="Accuracy")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history["loss"], name="Train loss"))
            fig.add_trace(go.Scatter(y=history["val_loss"], name="Val loss"))
            fig.update_layout(title="Loss over training", xaxis_title="Epoch", yaxis_title="Loss")
            st.plotly_chart(fig, use_container_width=True)

    if report:
        st.subheader("Classification report (test set)")
        rows = []
        for cls in class_names:
            if cls in report:
                rows.append({
                    "class": cls,
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1-score": report[cls]["f1-score"],
                    "support": report[cls]["support"],
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.metric("Overall test accuracy", f"{report['accuracy']:.2%}")

    figures_dir = ROOT / "reports" / "figures"
    col1, col2 = st.columns(2)
    with col1:
        cm_path = figures_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix", use_container_width=True)
    with col2:
        gradcam_path = figures_dir / "gradcam_samples.png"
        if gradcam_path.exists():
            st.image(str(gradcam_path), caption="Grad-CAM samples", use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: Business Impact
# ---------------------------------------------------------------------------
elif page == "Business Impact":
    st.title("💰 Business Impact")

    summary_path = ROOT / "reports" / "summary.md"
    if summary_path.exists():
        st.markdown(summary_path.read_text())
    else:
        st.info("Run `python src/models/evaluate_model.py` first to generate the business summary.")

    biz = config["business"]
    st.subheader("Cost assumptions")
    st.json(biz)
