# ─────────────────────────────────────────────────────────────────────────────
# app.py — Streamlit Deployment
# Enhanced CNN CIFAR-10 Classifier
# Author: Hussain Samdani
# GitHub: github.com/Hussain-Innovator
# ─────────────────────────────────────────────────────────────────────────────

import os
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

import time
import numpy as np
import pandas as pd
import streamlit as st
import onnxruntime as ort
from PIL import Image
import plotly.graph_objects as go

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 CNN Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 15px;
    }
    .result-highlight {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3d4f7c;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #666;
        padding: 20px;
        border-top: 1px solid #2d3250;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CLASS_EMOJIS = {
    "airplane"   : "✈️",
    "automobile" : "🚗",
    "bird"       : "🐦",
    "cat"        : "🐱",
    "deer"       : "🦌",
    "dog"        : "🐶",
    "frog"       : "🐸",
    "horse"      : "🐴",
    "ship"       : "🚢",
    "truck"      : "🚛"
}

CLASS_METRICS = {
    "airplane"   : {"precision": 0.90, "recall": 0.92, "f1": 0.91},
    "automobile" : {"precision": 0.96, "recall": 0.94, "f1": 0.95},
    "bird"       : {"precision": 0.89, "recall": 0.88, "f1": 0.88},
    "cat"        : {"precision": 0.89, "recall": 0.77, "f1": 0.82},
    "deer"       : {"precision": 0.90, "recall": 0.91, "f1": 0.90},
    "dog"        : {"precision": 0.86, "recall": 0.87, "f1": 0.87},
    "frog"       : {"precision": 0.91, "recall": 0.95, "f1": 0.93},
    "horse"      : {"precision": 0.92, "recall": 0.93, "f1": 0.92},
    "ship"       : {"precision": 0.94, "recall": 0.96, "f1": 0.95},
    "truck"      : {"precision": 0.93, "recall": 0.95, "f1": 0.94},
}

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load ONNX model — works on any Python version."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(app_dir, "models", "model.onnx"),
        "models/model.onnx",
        "../models/model.onnx",
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                session = ort.InferenceSession(path)
                return session, path
            except Exception as e:
                st.warning(f"Failed to load {path}: {e}")
                continue
    return None, None

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(session, image: Image.Image):
    """Run inference using ONNX runtime."""
    image       = image.convert("RGB")
    img_resized = image.resize((32, 32), Image.LANCZOS)
    arr         = np.array(img_resized).astype("float32") / 255.0
    arr         = np.expand_dims(arr, axis=0)
    input_name  = session.get_inputs()[0].name
    start       = time.time()
    probs       = session.run(None, {input_name: arr})[0][0]
    inf_time    = (time.time() - start) * 1000
    return probs, inf_time


def plot_predictions(probs):
    labels = [
        f"{CLASS_EMOJIS[CLASS_NAMES[i]]} {CLASS_NAMES[i].capitalize()}"
        for i in range(10)
    ]
    colors = [
        "#ff7f0e" if i == np.argmax(probs) else "#1f77b4"
        for i in range(10)
    ]
    fig = go.Figure(go.Bar(
        x=probs * 100,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='outside',
    ))
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Confidence (%)",
        xaxis=dict(range=[0, 115]),
        height=400,
        margin=dict(l=10, r=60, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,33,48,0.8)',
        font=dict(color='white'),
        showlegend=False,
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    return fig


def plot_f1_radar():
    classes   = [f"{CLASS_EMOJIS[c]} {c.capitalize()}" for c in CLASS_NAMES]
    f1_scores = [CLASS_METRICS[c]["f1"] for c in CLASS_NAMES]
    fig = go.Figure(go.Scatterpolar(
        r=f1_scores + [f1_scores[0]],
        theta=classes + [classes[0]],
        fill='toself',
        fillcolor='rgba(31,119,180,0.3)',
        line=dict(color='#1f77b4', width=2),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0.75, 1.0],
                gridcolor='rgba(255,255,255,0.2)',
                color='white'
            ),
            angularaxis=dict(color='white'),
            bgcolor='rgba(30,33,48,0.8)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title="Per-Class F1 Score",
        showlegend=False,
        height=400,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1f3a, #2d3561);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #3d4f7c;
        text-align: center;
    ">
        <h2 style="color:#ffffff; margin:0; font-size:20px; font-weight:700;
                   letter-spacing:1px;">CIFAR-10 CNN</h2>
        <p style="color:#8892b0; margin:6px 0 0 0; font-size:13px;">
            Enhanced Hybrid Classifier
        </p>
        <hr style="border:none; border-top:1px solid #3d4f7c; margin:12px 0;">
        <p style="color:#64ffda; margin:0; font-size:12px; letter-spacing:2px;">
            DEEP LEARNING PROJECT
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#8892b0; font-size:11px; letter-spacing:2px;
              text-transform:uppercase; margin-bottom:10px;">
        Model Performance
    </p>
    """, unsafe_allow_html=True)

    stats = [
        ("Test Accuracy",  "90.93%",   "#64ffda"),
        ("Top-3 Accuracy", "98.71%",   "#64ffda"),
        ("Macro F1 Score", "0.91",     "#a8b2d8"),
        ("Best Epoch",     "86 / 100", "#a8b2d8"),
        ("Overfit Gap",    "3.4%",     "#a8b2d8"),
    ]
    for label, value, color in stats:
        st.markdown(f"""
        <div style="background-color:#1e2130; border-left:3px solid {color};
                    border-radius:6px; padding:10px 14px; margin-bottom:8px;">
            <span style="color:#8892b0; font-size:12px;">{label}</span>
            <span style="color:{color}; font-weight:700; font-size:14px;
                         float:right;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#8892b0; font-size:11px; letter-spacing:2px;
              text-transform:uppercase; margin-bottom:10px;">
        Architecture Components
    </p>
    """, unsafe_allow_html=True)

    components = [
        ("Residual Connections",     "Skip paths for gradient flow"),
        ("SE Attention Blocks",      "Channel-wise recalibration"),
        ("Depthwise Separable Conv", "8-9x parameter reduction"),
        ("Global Average Pooling",   "Replaces Dense(512) head"),
        ("MixUp Augmentation",       "Label interpolation training"),
    ]
    for title, desc in components:
        st.markdown(f"""
        <div style="background-color:#1e2130; border-radius:6px;
                    padding:10px 14px; margin-bottom:8px;
                    border:1px solid #2d3250;">
            <div style="color:#ccd6f6; font-size:12px;
                        font-weight:600; margin-bottom:2px;">{title}</div>
            <div style="color:#8892b0; font-size:11px;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#8892b0; font-size:11px; letter-spacing:2px;
              text-transform:uppercase; margin-bottom:10px;">
        Training Configuration
    </p>
    """, unsafe_allow_html=True)

    configs = [
        ("Optimizer",   "SGD + Nesterov"),
        ("LR Schedule", "Cosine Annealing"),
        ("Warmup",      "5 epochs linear"),
        ("Loss",        "CE + Smoothing 0.1"),
        ("Dataset",     "CIFAR-10 / 50K imgs"),
        ("Framework",   "TensorFlow 2.15"),
    ]
    for key, val in configs:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between;
                    padding:6px 0; border-bottom:1px solid #1e2130;">
            <span style="color:#8892b0; font-size:11px;">{key}</span>
            <span style="color:#ccd6f6; font-size:11px;
                         font-weight:500;">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#8892b0; font-size:11px; letter-spacing:2px;
              text-transform:uppercase; margin-bottom:10px;">
        References
    </p>
    """, unsafe_allow_html=True)

    refs = [
        ("ResNet",    "He et al., CVPR 2016"),
        ("SENet",     "Hu et al., CVPR 2018"),
        ("MobileNet", "Howard et al., 2017"),
        ("MixUp",     "Zhang et al., ICLR 2018"),
        ("BatchNorm", "Ioffe & Szegedy, 2015"),
    ]
    for name, cite in refs:
        st.markdown(f"""
        <div style="margin-bottom:6px;">
            <span style="background-color:#2d3561; color:#64ffda;
                         font-size:10px; font-weight:600; padding:2px 7px;
                         border-radius:10px; margin-right:6px;">{name}</span>
            <span style="color:#8892b0; font-size:11px;">{cite}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <p style="color:#8892b0; font-size:11px; letter-spacing:2px;
              text-transform:uppercase; margin-bottom:10px;">Connect</p>
    <div style="display:flex; gap:8px; flex-direction:column;">
        <a href="https://github.com/Hussain-Innovator" target="_blank"
           style="display:block; background-color:#1e2130; color:#ccd6f6;
                  text-decoration:none; padding:9px 14px; border-radius:6px;
                  font-size:12px; font-weight:500; border:1px solid #2d3250;
                  text-align:center;">
            GitHub — Hussain-Innovator
        </a>
        <a href="https://linkedin.com/in/hussain56" target="_blank"
           style="display:block; background-color:#1e2130; color:#ccd6f6;
                  text-decoration:none; padding:9px 14px; border-radius:6px;
                  font-size:12px; font-weight:500; border:1px solid #2d3250;
                  text-align:center;">
            LinkedIn — hussain56
        </a>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("# Enhanced CNN — CIFAR-10 Classifier")
st.markdown(
    "### Custom Hybrid Architecture: ResNet + SE Attention + DepthwiseSep + GAP"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading model..."):
    model, model_path = load_model()

if model is None:
    st.error("Model file not found. Place `model.onnx` in `models/` folder.")
    st.stop()
else:
    st.success("Model Loaded Successfully")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "Classify Image",
    "Model Performance",
    "Architecture"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLASSIFY IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload an image to classify")
    st.markdown(
        "The model classifies images into one of **10 CIFAR-10 categories**. "
        "Any image works — it will be resized to 32×32 internally."
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
        )
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"""
            **Image Info:**
            - Original size: `{image.size[0]}×{image.size[1]}`
            - Mode: `{image.mode}`
            - Resized to: `32×32` for inference
            """)

    with col2:
        if uploaded_file:
            with st.spinner("Running inference..."):
                probs, inf_time = predict_image(model, image)

            pred_idx   = np.argmax(probs)
            pred_class = CLASS_NAMES[pred_idx]
            pred_conf  = probs[pred_idx] * 100
            pred_emoji = CLASS_EMOJIS[pred_class]

            st.markdown(f"""
            <div class="result-highlight">
                <h1>{pred_emoji}</h1>
                <h2>{pred_class.upper()}</h2>
                <h3>Confidence: {pred_conf:.1f}%</h3>
                <p>Inference time: {inf_time:.1f} ms</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            top3_idx = np.argsort(probs)[::-1][:3]
            st.markdown("**Top-3 Predictions:**")
            for rank, idx in enumerate(top3_idx, 1):
                st.markdown(
                    f"`#{rank}` {CLASS_EMOJIS[CLASS_NAMES[idx]]} "
                    f"**{CLASS_NAMES[idx].capitalize()}** "
                    f"— {probs[idx]*100:.1f}%"
                )

            st.plotly_chart(
                plot_predictions(probs),
                use_container_width=True
            )

            if pred_conf >= 80:
                st.success(f"High confidence prediction: {pred_conf:.1f}%")
            elif pred_conf >= 50:
                st.warning(f"Moderate confidence: {pred_conf:.1f}%")
            else:
                st.error(f"Low confidence: {pred_conf:.1f}% — image may not match CIFAR-10 categories well.")
        else:
            st.info("Upload an image on the left to classify it.")
            st.markdown("**Supported classes:**")
            cols = st.columns(5)
            for i, name in enumerate(CLASS_NAMES):
                with cols[i % 5]:
                    st.markdown(f"- {name.capitalize()}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Performance Analysis")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Test Accuracy",  "90.93%", "+0.93% vs baseline")
    m2.metric("Top-3 Accuracy", "98.71%")
    m3.metric("Macro F1",       "0.91")
    m4.metric("Best Epoch",     "86")
    m5.metric("Overfit Gap",    "3.4%",   "healthy")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Training Curves")
        app_dir = os.path.dirname(os.path.abspath(__file__))
        curves  = os.path.join(app_dir, "assets", "training_curves.png")
        if os.path.exists(curves):
            st.image(curves, use_column_width=True)
        else:
            st.info("Add training_curves.png to assets/ folder")

    with col2:
        st.markdown("#### Confusion Matrix")
        cm = os.path.join(app_dir, "assets", "confusion_matrix.png")
        if os.path.exists(cm):
            st.image(cm, use_column_width=True)
        else:
            st.info("Add confusion_matrix.png to assets/ folder")

    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Per-Class Performance")
        df = pd.DataFrame([
            {
                "Class"    : f"{CLASS_EMOJIS[c]} {c.capitalize()}",
                "Precision": CLASS_METRICS[c]["precision"],
                "Recall"   : CLASS_METRICS[c]["recall"],
                "F1 Score" : CLASS_METRICS[c]["f1"],
            }
            for c in CLASS_NAMES
        ])
        st.dataframe(
            df.style.background_gradient(
                subset=["Precision", "Recall", "F1 Score"],
                cmap="Blues"
            ),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.plotly_chart(plot_f1_radar(), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Sample Predictions on Test Set")
    preds = os.path.join(app_dir, "assets", "sample_predictions.png")
    if os.path.exists(preds):
        st.image(preds, use_column_width=True)
    else:
        st.info("Add sample_predictions.png to assets/ folder")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Architecture Overview")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Network Design")
        st.code("""
Input (32×32×3)
│
▼
Stem: Conv2D(64) + BN + ReLU
│
▼
Block 1 ──────────────────────────
Conv2D(64) + BN + ReLU
Conv2D(64) + BN + SE Attention
Identity Shortcut (64→64)
Add → ReLU → MaxPool → Dropout(0.25)
Output: 16×16×64
│
▼
Block 2 ──────────────────────────
Conv2D(128) + BN + ReLU
DepthwiseSep(128) + BN + SE Attention
Projection Shortcut 1×1 (64→128)
Add → ReLU → MaxPool → Dropout(0.25)
Output: 8×8×128
│
▼
Block 3 ──────────────────────────
Conv2D(256) + BN + ReLU
DepthwiseSep(256) + BN + SE Attention
Projection Shortcut 1×1 (128→256)
Add → ReLU → MaxPool → Dropout(0.25)
Output: 4×4×256
│
▼
Global Average Pooling → 256-dim
Dropout(0.4)
Dense(10, Softmax)
        """, language="")

    with col2:
        st.markdown("#### Key Components")

        with st.expander("Residual Connections", expanded=True):
            st.markdown("""
            Skip connections add input directly to block output.
            Solves vanishing gradient problem.
            **Math:** `output = F(x) + x`
            **Reference:** He et al., ResNet, CVPR 2016
            """)

        with st.expander("Squeeze-and-Excitation (SE) Block"):
            st.markdown("""
            Channel attention — learns which feature maps matter.
            1. Squeeze: GlobalAvgPool → (1,1,C)
            2. Excitation: Dense(C/16) → Dense(C, sigmoid)
            3. Scale: Multiply back onto channels
            **Reference:** Hu et al., SENet, CVPR 2018
            """)

        with st.expander("Depthwise Separable Convolution"):
            st.markdown("""
            Splits conv into depthwise + pointwise.
            Reduces parameters ~8-9× vs standard conv.
            **Reference:** Howard et al., MobileNets 2017
            """)

        with st.expander("Projection Shortcut"):
            st.markdown("""
            1×1 Conv on skip path when channels change.
            Block 1: 64→64 (identity)
            Block 2: 64→128 (projection)
            Block 3: 128→256 (projection)
            """)

        with st.expander("Global Average Pooling"):
            st.markdown("""
            Replaces Flatten + Dense(512).
            4×4×256 → 256-dim vector directly.
            Reduces params, improves generalisation.
            """)

    st.markdown("---")
    st.markdown("#### Training Configuration")
    cfg1, cfg2, cfg3 = st.columns(3)

    with cfg1:
        st.markdown("""
        **Optimisation**
        - SGD + Nesterov (momentum=0.9)
        - Base LR: 0.1
        - Cosine Annealing
        - Linear Warmup (5 epochs)
        """)
    with cfg2:
        st.markdown("""
        **Regularisation**
        - Batch Normalisation
        - Dropout: 0.25 / 0.4
        - Label Smoothing: 0.1
        - MixUp: α = 0.4
        """)
    with cfg3:
        st.markdown("""
        **Data Augmentation**
        - Random Horizontal Flip
        - Random Crop (pad 4px)
        - Brightness Jitter
        - Saturation Jitter
        - MixUp Augmentation
        """)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>
        Built by <strong>Hussain Samdani</strong> |
        <a href="https://github.com/Hussain-Innovator"
           target="_blank">GitHub</a> |
        <a href="https://linkedin.com/in/hussain56"
           target="_blank">LinkedIn</a>
    </p>
    <p>ONNX Runtime · Streamlit · CIFAR-10 ·
       ResNet · SENet · MobileNets · MixUp</p>
</div>
""", unsafe_allow_html=True)