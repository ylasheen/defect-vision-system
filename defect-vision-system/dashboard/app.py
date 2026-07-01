"""
Streamlit Dashboard — Defect Vision System (Enterprise Edition)
------------------------------------------------------------------
Run with:
    streamlit run dashboard/app.py

Enterprise-grade visual quality inspection console: single-image
inspection, batch processing, video analysis, live KPIs, ROI modeling,
ROC/PR diagnostics, session history, and exportable reports — all
built on top of the existing CNN + Grad-CAM pipeline.

Visual identity: instrumentation / andon-light control panel — the
same red / amber / green signal colors used on a factory status
tower, paired with a monospace readout face for every number.
"""
import json
import sys
import tempfile
from datetime import datetime
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

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="Defect Vision System", layout="wide")

config = load_config()

# ---------------------------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------------------------
BG = "#0a0e14"
PANEL = "#121821"
PANEL_ALT = "#161d28"
BORDER = "#232b38"
TEXT_PRIMARY = "#e8ebf1"
TEXT_SECONDARY = "#8891a1"

ACCENT = "#4f8cff"        # interactive / data accent — charts, links, focus
GOOD_COLOR = "#22c55e"    # andon green  — pass
WARN_COLOR = "#f59e0b"    # andon amber  — caution / alert
DEFECT_COLOR = "#ef4444"  # andon red    — fail

FONT_SANS = "'Inter', -apple-system, sans-serif"
FONT_MONO = "'IBM Plex Mono', 'SFMono-Regular', monospace"

# ---------------------------------------------------------------------------
# GLOBAL STYLE
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] {{
        font-family: {FONT_SANS};
    }}

    .stApp {{
        background-color: {BG};
        background-image:
            linear-gradient(rgba(79, 140, 255, 0.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(79, 140, 255, 0.045) 1px, transparent 1px);
        background-size: 34px 34px;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {PANEL};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        font-family: {FONT_SANS};
    }}

    /* ---- Brand block ---- */
    .brand-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 4px;
    }}
    .brand-mark {{
        width: 34px;
        height: 34px;
        border-radius: 50%;
        border: 2px solid {ACCENT};
        position: relative;
        flex-shrink: 0;
    }}
    .brand-mark::after {{
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 9px;
        height: 9px;
        background: {ACCENT};
        border-radius: 50%;
        transform: translate(-50%, -50%);
    }}
    .brand-title {{
        color: {TEXT_PRIMARY};
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        line-height: 1.2;
    }}
    .brand-sub {{
        color: {TEXT_SECONDARY};
        font-size: 0.74rem;
        margin-top: 1px;
    }}
    .sidebar-eyebrow {{
        font-family: {FONT_MONO};
        color: {TEXT_SECONDARY};
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 18px 0 6px 0;
    }}

    /* ---- Page header ---- */
    .page-header {{
        border-bottom: 1px solid {BORDER};
        padding-bottom: 14px;
        margin-bottom: 22px;
    }}
    .page-eyebrow {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: {FONT_MONO};
        font-size: 0.72rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: {ACCENT};
        margin-bottom: 8px;
    }}
    .page-eyebrow .dot {{
        width: 6px;
        height: 6px;
        background: {ACCENT};
    }}
    .page-title {{
        color: {TEXT_PRIMARY};
        font-size: 1.7rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }}
    .page-subtitle {{
        color: {TEXT_SECONDARY};
        font-size: 0.9rem;
        margin-top: 6px;
        max-width: 780px;
        line-height: 1.5;
    }}

    /* ---- Status strip (system health) ---- */
    .status-strip {{
        display: flex;
        align-items: center;
        gap: 12px;
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 24px;
        font-family: {FONT_MONO};
    }}
    .status-strip-led {{
        width: 9px;
        height: 9px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    .status-strip-label {{
        font-weight: 700;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
    }}
    .status-strip-msg {{
        color: {TEXT_SECONDARY};
        font-size: 0.78rem;
        flex-grow: 1;
    }}
    .status-strip-time {{
        color: {TEXT_SECONDARY};
        opacity: 0.55;
        font-size: 0.7rem;
    }}

    /* ---- KPI cards ---- */
    .kpi-card {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT};
        border-radius: 6px;
        padding: 16px 18px;
        height: 100%;
    }}
    .kpi-label {{
        color: {TEXT_SECONDARY};
        font-family: {FONT_MONO};
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        color: {TEXT_PRIMARY};
        font-family: {FONT_MONO};
        font-size: 1.75rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }}
    .kpi-sub {{
        color: {TEXT_SECONDARY};
        font-size: 0.76rem;
        margin-top: 5px;
    }}

    /* ---- Status badge (LED readout) ---- */
    .status-badge {{
        display: flex;
        align-items: center;
        gap: 12px;
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 16px 18px;
    }}
    .status-led {{
        width: 13px;
        height: 13px;
        border-radius: 50%;
        flex-shrink: 0;
    }}
    .status-text {{
        font-family: {FONT_MONO};
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.06em;
        line-height: 1.2;
    }}
    .status-confidence {{
        font-family: {FONT_MONO};
        color: {TEXT_SECONDARY};
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        margin-top: 3px;
    }}

    .alert-banner {{
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid {WARN_COLOR};
        color: #fcd34d;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: {FONT_MONO};
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 16px;
    }}

    hr {{ border-color: {BORDER}; }}

    /* ---- Widget polish ---- */
    div[data-testid="stMetricValue"] {{
        font-family: {FONT_MONO};
        color: {TEXT_PRIMARY};
    }}
    div[data-testid="stMetricLabel"] {{
        font-family: {FONT_MONO};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.7rem;
    }}
    div[data-testid="stFileUploader"] section {{
        background: {PANEL_ALT};
        border: 1px dashed {BORDER};
        border-radius: 8px;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 8px;
        overflow: hidden;
    }}
    div[data-testid="stAlert"] {{
        border-radius: 8px;
        font-size: 0.88rem;
    }}
    .stButton > button, .stDownloadButton > button {{
        font-family: {FONT_SANS};
        background: {PANEL_ALT};
        color: {TEXT_PRIMARY};
        border: 1px solid {BORDER};
        border-radius: 6px;
        font-weight: 500;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        border-color: {ACCENT};
        color: {ACCENT};
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------------------------
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


@st.cache_data
def load_predictions():
    """Optional file: reports/predictions.json with {"y_true": [...], "y_prob": [...]}
    Needed for the ROC / Precision-Recall page. See note on that page for how
    to generate it from evaluate_model.py."""
    path = ROOT / "reports" / "predictions.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


model, class_names = load_model_and_classes()
target_size = tuple(config["model"]["target_size"])
DEFECT_IDX = class_names.index("defective") if "defective" in class_names else None
GOOD_IDX = class_names.index("good") if "good" in class_names else None

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []
if "last_video_results" not in st.session_state:
    st.session_state.last_video_results = None


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def run_inference(img_pil: Image.Image, threshold: float = 0.5):
    """Runs the model + Grad-CAM on a single PIL image and returns a result dict."""
    img_resized = img_pil.resize(target_size)
    arr = np.array(img_resized).astype("float32")
    batch = arr[np.newaxis, ...]
    probs = model.predict(batch, verbose=0)[0]

    if DEFECT_IDX is not None:
        defect_prob = float(probs[DEFECT_IDX])
        if defect_prob >= threshold:
            label = "defective"
            confidence = defect_prob
        else:
            label = "good"
            confidence = 1.0 - defect_prob
        pred_idx = DEFECT_IDX if label == "defective" else GOOD_IDX
    else:
        pred_idx = int(np.argmax(probs))
        label = class_names[pred_idx]
        confidence = float(probs[pred_idx])

    heatmap = make_gradcam_heatmap(batch, model, pred_index=pred_idx)
    overlay = overlay_heatmap(heatmap, arr.astype("uint8"))

    return {
        "label": label,
        "confidence": confidence,
        "probs": probs,
        "resized_image": img_resized,
        "overlay": overlay,
    }


def log_inspection(source: str, label: str, confidence: float):
    st.session_state.inspection_history.append(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
            "label": label,
            "confidence": round(confidence, 4),
        }
    )


def status_badge(label: str, confidence: float, container=None):
    target = container if container is not None else st
    color = DEFECT_COLOR if label == "defective" else GOOD_COLOR
    text = "DEFECTIVE" if label == "defective" else "GOOD"
    target.markdown(
        f"""
        <div class="status-badge" style="border-color:{color}55;">
            <span class="status-led" style="background:{color};box-shadow:0 0 10px {color};"></span>
            <div>
                <div class="status-text" style="color:{color};">{text}</div>
                <div class="status-confidence">CONFIDENCE&nbsp;{confidence:.1%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", container=None, accent: str = None):
    target = container if container is not None else st
    border_color = accent if accent else ACCENT
    target.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{border_color};">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(eyebrow: str, title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-eyebrow"><span class="dot"></span>{eyebrow}</div>
            <div class="page-title">{title}</div>
            {f'<div class="page-subtitle">{subtitle}</div>' if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def history_dataframe() -> pd.DataFrame:
    if not st.session_state.inspection_history:
        return pd.DataFrame(columns=["timestamp", "source", "label", "confidence"])
    return pd.DataFrame(st.session_state.inspection_history)


def apply_chart_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SANS, color=TEXT_PRIMARY),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


def build_html_report(defect_rate, avg_conf, total, report, biz) -> str:
    rows_html = ""
    if report:
        for cls in class_names:
            if cls in report:
                r = report[cls]
                rows_html += (
                    f"<tr><td>{cls}</td><td>{r['precision']:.4f}</td>"
                    f"<td>{r['recall']:.4f}</td><td>{r['f1-score']:.4f}</td>"
                    f"<td>{r['support']}</td></tr>"
                )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: 'Inter', Arial, sans-serif; background:{BG}; color:{TEXT_PRIMARY}; padding:32px; }}
        h1 {{ color:{TEXT_PRIMARY}; font-weight:700; }}
        h2 {{ color:{TEXT_PRIMARY}; border-bottom:1px solid {BORDER}; padding-bottom:6px; }}
        table {{ border-collapse: collapse; width:100%; margin-top:12px; font-family: 'IBM Plex Mono', monospace; font-size:0.85rem; }}
        th, td {{ border:1px solid {BORDER}; padding:8px 12px; text-align:left; }}
        th {{ background:{PANEL}; color:{TEXT_SECONDARY}; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.06em; }}
        .metric {{ display:inline-block; background:{PANEL}; border:1px solid {BORDER}; border-left:3px solid {ACCENT};
                   border-radius:6px; padding:14px 18px; margin-right:12px; margin-bottom:12px; }}
        .metric .label {{ color:{TEXT_SECONDARY}; font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
                           font-family:'IBM Plex Mono', monospace; }}
        .metric .value {{ color:{TEXT_PRIMARY}; font-size:1.4rem; font-weight:700; font-family:'IBM Plex Mono', monospace; }}
    </style>
    </head>
    <body>
        <h1>Defect Vision System — Inspection Report</h1>
        <p style="color:{TEXT_SECONDARY};">Generated: {generated_at}</p>
        <div>
            <div class="metric"><div class="label">Session Inspections</div><div class="value">{total}</div></div>
            <div class="metric"><div class="label">Defect Rate (session)</div><div class="value">{defect_rate:.1%}</div></div>
            <div class="metric"><div class="label">Avg. Confidence</div><div class="value">{avg_conf:.1%}</div></div>
        </div>
        <h2>Model Test-Set Performance</h2>
        <table>
            <tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>
            {rows_html if rows_html else "<tr><td colspan='5'>No classification report found.</td></tr>"}
        </table>
        <h2>Cost Assumptions</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            {''.join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in biz.items())}
        </table>
    </body>
    </html>
    """
    return html


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div class="brand-row">
        <div class="brand-mark"></div>
        <div>
            <div class="brand-title">Defect Vision System</div>
            <div class="brand-sub">Youssef Lasheen — AI &amp; ML Engineer</div>
        </div>
    </div>
    <div class="sidebar-eyebrow">Modules</div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Live Inspector",
        "Batch Processing",
        "Video Analysis",
        "Model Performance",
        "ROC / PR Analysis",
        "ROI Calculator",
        "Inspection History",
        "Business Impact",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown('<div class="sidebar-eyebrow">Calibration</div>', unsafe_allow_html=True)
threshold = st.sidebar.slider(
    "Decision threshold (defective class)", min_value=0.05, max_value=0.95, value=0.50, step=0.05
)
alert_threshold = st.sidebar.slider(
    "Alert on defect rate above", min_value=0.05, max_value=0.90, value=0.20, step=0.05
)
st.sidebar.caption(
    "Threshold controls apply to Live Inspector, Batch Processing and Video Analysis."
)

# ---------------------------------------------------------------------------
# SYSTEM STATUS STRIP (shown above every module)
# ---------------------------------------------------------------------------
def render_status_strip():
    hist_df = history_dataframe()
    total = len(hist_df)
    if total == 0:
        status, color, msg = "STANDBY", TEXT_SECONDARY, "No inspections logged in this session yet"
    else:
        defect_rate = (hist_df["label"] == "defective").mean()
        if defect_rate > alert_threshold:
            status, color = "ALERT", DEFECT_COLOR
            msg = f"Defect rate {defect_rate:.1%} exceeds the {alert_threshold:.0%} threshold"
        else:
            status, color = "NOMINAL", GOOD_COLOR
            msg = f"Defect rate {defect_rate:.1%} — within threshold ({total} inspections logged)"
    st.markdown(
        f"""
        <div class="status-strip" style="border-color:{color}55;">
            <span class="status-strip-led" style="background:{color};box-shadow:0 0 8px {color};"></span>
            <span class="status-strip-label" style="color:{color};">{status}</span>
            <span class="status-strip-msg">{msg}</span>
            <span class="status-strip-time">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_status_strip()

# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------
if page == "Overview":
    page_header("Monitoring", "Operations Overview", "Live session KPIs across all inspections run in this dashboard session.")

    hist_df = history_dataframe()
    total = len(hist_df)
    defect_rate = (hist_df["label"] == "defective").mean() if total else 0.0
    avg_conf = hist_df["confidence"].mean() if total else 0.0
    status_color = DEFECT_COLOR if (total and defect_rate > alert_threshold) else GOOD_COLOR

    if total and defect_rate > alert_threshold:
        st.markdown(
            f'<div class="alert-banner">ALERT — Session defect rate ({defect_rate:.1%}) '
            f'exceeds the configured threshold ({alert_threshold:.1%}).</div>',
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    kpi_card("Total Inspections", f"{total}", "This session", c1)
    kpi_card("Defect Rate", f"{defect_rate:.1%}", "Session average", c2, accent=status_color if total else None)
    kpi_card("Average Confidence", f"{avg_conf:.1%}" if total else "—", "All inspections", c3)
    kpi_card(
        "Status",
        "ALERT" if (total and defect_rate > alert_threshold) else "NORMAL",
        f"Threshold {alert_threshold:.0%}",
        c4,
        accent=status_color,
    )

    st.markdown("###")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Recent Inspections")
        if total:
            st.dataframe(hist_df.tail(15).iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.info("No inspections logged yet. Use Live Inspector, Batch Processing or Video Analysis.")
    with col_b:
        st.subheader("Defect Rate Gauge")
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=defect_rate * 100 if total else 0,
                number={"suffix": "%", "font": {"family": FONT_MONO}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": ACCENT},
                    "bgcolor": PANEL,
                    "bordercolor": BORDER,
                    "steps": [
                        {"range": [0, alert_threshold * 100], "color": "#14532d"},
                        {"range": [alert_threshold * 100, 100], "color": "#7f1d1d"},
                    ],
                },
            )
        )
        gauge.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
        apply_chart_theme(gauge)
        st.plotly_chart(gauge, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: Live Inspector
# ---------------------------------------------------------------------------
elif page == "Live Inspector":
    page_header(
        "Inspection",
        "Live Inspector",
        "Upload a product-surface photo. The CNN classifies it as good or defective "
        "and Grad-CAM highlights exactly which pixels drove the decision.",
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
        # Prefer the committed demo samples (always present in the repo);
        # fall back to the generated test split if it exists locally.
        sample_dir = ROOT / "dashboard" / "assets" / "samples" / use_sample
        sample_files = sorted(sample_dir.glob("*.png"))
        if not sample_files:
            sample_dir = ROOT / "data" / "processed" / "test" / use_sample
            sample_files = sorted(sample_dir.glob("*.png"))
        if sample_files:
            idx = np.random.randint(0, len(sample_files))
            img = Image.open(sample_files[idx]).convert("RGB")
        else:
            st.warning("No sample images found for this class.")

    if img is not None:
        result = run_inference(img, threshold=threshold)
        log_inspection("live_inspector", result["label"], result["confidence"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(result["resized_image"], caption="Input image", use_container_width=True)
        with col2:
            st.image(result["overlay"], caption="Grad-CAM — model attention", use_container_width=True)
        with col3:
            status_badge(result["label"], result["confidence"])
            st.markdown("**Class probabilities**")
            for cls, p in zip(class_names, result["probs"]):
                st.progress(float(p), text=f"{cls}: {p:.1%}")
    else:
        st.info("Upload an image above, or click one of the sample buttons to try the system.")

# ---------------------------------------------------------------------------
# PAGE: Batch Processing
# ---------------------------------------------------------------------------
elif page == "Batch Processing":
    page_header("Inspection", "Batch Processing", "Upload multiple images to run inspection across an entire batch at once.")

    files = st.file_uploader(
        "Upload images (PNG/JPG)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
    )

    if files:
        rows = []
        thumbs = []
        progress = st.progress(0.0, text="Processing batch...")
        for i, f in enumerate(files):
            img = Image.open(f).convert("RGB")
            result = run_inference(img, threshold=threshold)
            log_inspection(f"batch:{f.name}", result["label"], result["confidence"])
            rows.append(
                {
                    "filename": f.name,
                    "label": result["label"],
                    "confidence": round(result["confidence"], 4),
                }
            )
            thumbs.append((f.name, result["resized_image"], result["label"], result["confidence"]))
            progress.progress((i + 1) / len(files), text=f"Processing batch... ({i + 1}/{len(files)})")
        progress.empty()

        batch_df = pd.DataFrame(rows)
        defect_rate = (batch_df["label"] == "defective").mean()

        if defect_rate > alert_threshold:
            st.markdown(
                f'<div class="alert-banner">ALERT — Batch defect rate ({defect_rate:.1%}) '
                f'exceeds the configured threshold ({alert_threshold:.1%}).</div>',
                unsafe_allow_html=True,
            )

        c1, c2, c3 = st.columns(3)
        kpi_card("Batch Size", f"{len(files)}", "Images processed", c1)
        kpi_card("Defect Rate", f"{defect_rate:.1%}", "This batch", c2, accent=DEFECT_COLOR if defect_rate > alert_threshold else GOOD_COLOR)
        kpi_card("Avg. Confidence", f"{batch_df['confidence'].mean():.1%}", "This batch", c3)

        st.subheader("Results")
        st.dataframe(batch_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download results (CSV)",
            batch_df.to_csv(index=False).encode("utf-8"),
            file_name="batch_results.csv",
            mime="text/csv",
        )

        st.subheader("Thumbnails")
        grid_cols = st.columns(4)
        for i, (name, thumb, label, conf) in enumerate(thumbs):
            with grid_cols[i % 4]:
                st.image(thumb, use_container_width=True)
                color = DEFECT_COLOR if label == "defective" else GOOD_COLOR
                st.markdown(
                    f'<div style="text-align:center;color:{color};font-family:{FONT_MONO};'
                    f'font-weight:600;font-size:0.82rem;">{label.upper()} ({conf:.0%})</div>'
                    f'<div style="text-align:center;color:{TEXT_SECONDARY};font-size:0.72rem;">{name}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Upload two or more images to run a batch inspection.")

# ---------------------------------------------------------------------------
# PAGE: Video Analysis
# ---------------------------------------------------------------------------
elif page == "Video Analysis":
    page_header(
        "Inspection",
        "Video Analysis",
        "Upload a video of the production line. Frames are sampled at a fixed interval, "
        "each frame is run through the model, and results are plotted as a timeline.",
    )

    if not CV2_AVAILABLE:
        st.warning(
            "This feature requires OpenCV, which is not installed in the current "
            "environment. Add `opencv-python-headless` to requirements.txt and redeploy "
            "to enable video analysis. Every other page on this dashboard works without it."
        )
    else:
        video_file = st.file_uploader("Upload a video (MP4/AVI/MOV)", type=["mp4", "avi", "mov", "mkv"])
        col_a, col_b = st.columns(2)
        with col_a:
            sample_interval = st.slider("Sample every N seconds", 0.5, 5.0, 1.0, 0.5)
        with col_b:
            max_frames = st.slider("Max frames to analyze", 10, 150, 60, 10)

        if video_file is not None and st.button("Run video analysis", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = max(int(fps * sample_interval), 1)

            rows = []
            flagged_frames = []
            frame_idx = 0
            analyzed = 0
            progress = st.progress(0.0, text="Analyzing video...")

            while cap.isOpened() and analyzed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_frame = Image.fromarray(rgb)
                    result = run_inference(pil_frame, threshold=threshold)
                    timestamp_sec = frame_idx / fps
                    rows.append(
                        {
                            "timestamp_sec": round(timestamp_sec, 2),
                            "label": result["label"],
                            "confidence": round(result["confidence"], 4),
                        }
                    )
                    if result["label"] == "defective":
                        flagged_frames.append((timestamp_sec, result["resized_image"], result["confidence"]))
                    log_inspection(f"video:{video_file.name}@{timestamp_sec:.1f}s", result["label"], result["confidence"])
                    analyzed += 1
                    progress.progress(min(analyzed / max_frames, 1.0), text=f"Analyzing video... ({analyzed}/{max_frames} frames)")
                frame_idx += 1

            cap.release()
            progress.empty()

            if not rows:
                st.error("No frames could be extracted from this video.")
            else:
                video_df = pd.DataFrame(rows)
                st.session_state.last_video_results = video_df
                defect_rate = (video_df["label"] == "defective").mean()

                if defect_rate > alert_threshold:
                    st.markdown(
                        f'<div class="alert-banner">ALERT — {len(flagged_frames)} defective frame(s) '
                        f'detected ({defect_rate:.1%} of analyzed frames).</div>',
                        unsafe_allow_html=True,
                    )

                c1, c2, c3 = st.columns(3)
                kpi_card("Frames Analyzed", f"{len(video_df)}", f"Every {sample_interval}s", c1)
                kpi_card("Defect Rate", f"{defect_rate:.1%}", "Across sampled frames", c2, accent=DEFECT_COLOR if defect_rate > alert_threshold else GOOD_COLOR)
                kpi_card("Flagged Frames", f"{len(flagged_frames)}", "Defective detections", c3)

                st.subheader("Defect Timeline")
                timeline = go.Figure()
                colors = [DEFECT_COLOR if l == "defective" else GOOD_COLOR for l in video_df["label"]]
                timeline.add_trace(
                    go.Scatter(
                        x=video_df["timestamp_sec"],
                        y=video_df["confidence"],
                        mode="markers+lines",
                        marker=dict(color=colors, size=9),
                        line=dict(color=BORDER),
                        name="Confidence",
                    )
                )
                timeline.update_layout(xaxis_title="Time (seconds)", yaxis_title="Confidence")
                apply_chart_theme(timeline)
                st.plotly_chart(timeline, use_container_width=True)

                if flagged_frames:
                    st.subheader("Flagged Frames")
                    grid_cols = st.columns(4)
                    for i, (ts, thumb, conf) in enumerate(flagged_frames[:12]):
                        with grid_cols[i % 4]:
                            st.image(thumb, use_container_width=True)
                            st.markdown(
                                f'<div style="text-align:center;color:{DEFECT_COLOR};'
                                f'font-family:{FONT_MONO};font-size:0.75rem;">t = {ts:.1f}s ({conf:.0%})</div>',
                                unsafe_allow_html=True,
                            )

                st.download_button(
                    "Download frame-level results (CSV)",
                    video_df.to_csv(index=False).encode("utf-8"),
                    file_name="video_analysis_results.csv",
                    mime="text/csv",
                )
        elif st.session_state.last_video_results is not None:
            st.info("Showing results from the last run. Upload a new video and click Run to re-analyze.")
            st.dataframe(st.session_state.last_video_results, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# PAGE: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    page_header("Analytics", "Model Performance", "Training curves and test-set diagnostics for the current model.")

    history = load_training_history()
    report = load_classification_report()

    if history:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history["accuracy"], name="Train accuracy", line=dict(color=ACCENT)))
            fig.add_trace(go.Scatter(y=history["val_accuracy"], name="Val accuracy", line=dict(color=GOOD_COLOR)))
            fig.update_layout(title="Accuracy over training", xaxis_title="Epoch", yaxis_title="Accuracy")
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history["loss"], name="Train loss", line=dict(color=ACCENT)))
            fig.add_trace(go.Scatter(y=history["val_loss"], name="Val loss", line=dict(color=WARN_COLOR)))
            fig.update_layout(title="Loss over training", xaxis_title="Epoch", yaxis_title="Loss")
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    if report:
        st.subheader("Classification Report (Test Set)")
        rows = []
        for cls in class_names:
            if cls in report:
                rows.append(
                    {
                        "class": cls,
                        "precision": report[cls]["precision"],
                        "recall": report[cls]["recall"],
                        "f1-score": report[cls]["f1-score"],
                        "support": report[cls]["support"],
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
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
# PAGE: ROC / PR Analysis
# ---------------------------------------------------------------------------
elif page == "ROC / PR Analysis":
    page_header("Analytics", "ROC / Precision-Recall Analysis", "Threshold-independent diagnostics for the defective class.")

    if not SKLEARN_AVAILABLE:
        st.warning("This page requires scikit-learn. Add `scikit-learn` to requirements.txt and redeploy.")
    else:
        preds = load_predictions()
        if not preds or "y_true" not in preds or "y_prob" not in preds:
            st.info(
                "No `reports/predictions.json` file found. ROC and Precision-Recall curves "
                "need the raw per-sample probabilities from the test set, which the current "
                "evaluate_model.py does not export. Add the following before it writes "
                "classification_report.json, using the defective class as the positive class:"
            )
            st.code(
                "import json\n"
                "y_prob_defective = probs[:, class_names.index('defective')].tolist()\n"
                "y_true_defective = (y_test == class_names.index('defective')).astype(int).tolist()\n"
                "with open(ROOT / 'reports' / 'predictions.json', 'w') as f:\n"
                "    json.dump({'y_true': y_true_defective, 'y_prob': y_prob_defective}, f)\n",
                language="python",
            )
        else:
            y_true = np.array(preds["y_true"])
            y_prob = np.array(preds["y_prob"])

            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)

            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC = {roc_auc:.3f})", line=dict(color=ACCENT)))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random baseline", line=dict(dash="dash", color=TEXT_SECONDARY)))
                fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                apply_chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recall, y=precision, name="Precision-Recall", line=dict(color=ACCENT)))
                fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
                apply_chart_theme(fig)
                st.plotly_chart(fig, use_container_width=True)

            st.metric("ROC AUC", f"{roc_auc:.4f}")

# ---------------------------------------------------------------------------
# PAGE: ROI Calculator
# ---------------------------------------------------------------------------
elif page == "ROI Calculator":
    page_header("Analytics", "ROI Calculator", "Model estimates prefill from the test-set classification report where available.")

    report = load_classification_report()
    biz = config["business"]

    default_recall = report["defective"]["recall"] if report and "defective" in report else 0.95
    default_defect_rate = 0.10
    if report and "defective" in report and "good" in report:
        d_support = report["defective"]["support"]
        g_support = report["good"]["support"]
        if d_support + g_support > 0:
            default_defect_rate = d_support / (d_support + g_support)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Volume & Defect Rate")
        daily_volume = st.number_input("Units inspected per day", min_value=1, value=int(biz.get("units_inspected_per_day", 500)))
        defect_rate_input = st.slider("Historical defect rate", 0.0, 1.0, float(round(default_defect_rate, 2)))
        model_recall = st.slider("Model recall (defect detection rate)", 0.0, 1.0, float(round(default_recall, 2)))
        model_false_alarm_rate = st.slider("Model false-alarm rate on good units", 0.0, 1.0, 0.0)
    with col2:
        st.subheader("Cost Assumptions")
        cost_missed = st.number_input("Cost per missed defect ($)", min_value=0.0, value=float(biz.get("cost_per_missed_defect", 85)))
        cost_false_alarm = st.number_input("Cost per false alarm ($)", min_value=0.0, value=float(biz.get("cost_per_false_alarm", 6)))
        manual_recall = st.slider("Manual inspection recall (baseline)", 0.0, 1.0, 1.0)
        manual_false_alarm_rate = st.slider("Manual inspection false-alarm rate (baseline)", 0.0, 1.0, 0.0)

    daily_defects = daily_volume * defect_rate_input
    daily_good = daily_volume - daily_defects

    model_missed = daily_defects * (1 - model_recall)
    model_false_alarms = daily_good * model_false_alarm_rate
    model_daily_cost = model_missed * cost_missed + model_false_alarms * cost_false_alarm

    manual_missed = daily_defects * (1 - manual_recall)
    manual_false_alarms = daily_good * manual_false_alarm_rate
    manual_daily_cost = manual_missed * cost_missed + manual_false_alarms * cost_false_alarm

    daily_savings = manual_daily_cost - model_daily_cost

    st.markdown("###")
    c1, c2, c3 = st.columns(3)
    kpi_card("Daily Savings", f"${daily_savings:,.2f}", "Model vs. manual baseline", c1)
    kpi_card("Monthly Savings", f"${daily_savings * 30:,.2f}", "30-day estimate", c2)
    kpi_card("Annual Savings", f"${daily_savings * 365:,.2f}", "365-day estimate", c3)

    st.subheader("Cost Comparison")
    fig = go.Figure(
        data=[
            go.Bar(name="Manual Inspection", x=["Daily Cost"], y=[manual_daily_cost], marker_color=TEXT_SECONDARY),
            go.Bar(name="Model-Assisted", x=["Daily Cost"], y=[model_daily_cost], marker_color=ACCENT),
        ]
    )
    fig.update_layout(barmode="group", yaxis_title="Estimated cost ($)")
    apply_chart_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Costs are estimated from missed defects and false alarms only, at the volume and "
        "rates configured above. They do not include labor, throughput, or downtime effects."
    )

# ---------------------------------------------------------------------------
# PAGE: Inspection History
# ---------------------------------------------------------------------------
elif page == "Inspection History":
    page_header("Monitoring", "Inspection History", "All inspections run in this browser session (Live Inspector, Batch Processing, Video Analysis).")

    hist_df = history_dataframe()

    if hist_df.empty:
        st.info("No inspections logged yet in this session.")
    else:
        col1, col2 = st.columns([1, 3])
        with col1:
            label_filter = st.selectbox("Filter by label", ["All", "defective", "good"])
        filtered = hist_df if label_filter == "All" else hist_df[hist_df["label"] == label_filter]

        st.dataframe(filtered.iloc[::-1], use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Download history (CSV)",
                hist_df.to_csv(index=False).encode("utf-8"),
                file_name="inspection_history.csv",
                mime="text/csv",
            )
        with col_b:
            if st.button("Clear history", use_container_width=True):
                st.session_state.inspection_history = []
                st.rerun()

        if len(hist_df) >= 3:
            st.subheader("Defect Rate Trend")
            hist_df["is_defective"] = (hist_df["label"] == "defective").astype(int)
            hist_df["rolling_rate"] = hist_df["is_defective"].expanding().mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hist_df["rolling_rate"], mode="lines", line=dict(color=ACCENT)))
            fig.update_layout(xaxis_title="Inspection #", yaxis_title="Cumulative defect rate")
            apply_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE: Business Impact
# ---------------------------------------------------------------------------
elif page == "Business Impact":
    page_header("Analytics", "Business Impact", "Cost model, evaluation artifacts, and exportable reporting.")

    summary_path = ROOT / "reports" / "summary.md"
    if summary_path.exists():
        st.markdown(summary_path.read_text())
    else:
        st.info("Run `python src/models/evaluate_model.py` first to generate the business summary.")

    figures_dir = ROOT / "reports" / "figures"
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        gradcam_path = figures_dir / "gradcam_samples.png"
        if gradcam_path.exists():
            st.image(str(gradcam_path), caption="Grad-CAM samples", use_container_width=True)
    with fig_col2:
        cm_path = figures_dir / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix", use_container_width=True)

    biz = config["business"]
    st.subheader("Cost Assumptions")
    st.json(biz)

    st.subheader("Export Report")
    hist_df = history_dataframe()
    total = len(hist_df)
    defect_rate = (hist_df["label"] == "defective").mean() if total else 0.0
    avg_conf = hist_df["confidence"].mean() if total else 0.0
    report = load_classification_report()

    html_report = build_html_report(defect_rate, avg_conf, total, report, biz)
    st.download_button(
        "Download HTML report",
        html_report.encode("utf-8"),
        file_name=f"defect_vision_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
    )
    st.caption("The report includes session KPIs, the test-set classification report, and cost assumptions.")
