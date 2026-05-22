import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sea Animal Classifier",
    page_icon="🌊",
    layout="centered"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }

    .stApp {
        background: linear-gradient(160deg, #020c1b 0%, #041524 50%, #061e35 100%);
        color: #e0f2fe;
    }

    .hero-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #64748b;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    .card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(56,189,248,0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(8px);
    }

    .pred-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }

    .pred-value {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2rem;
        color: #38bdf8;
    }

    .conf-value {
        font-family: 'Space Mono', monospace;
        font-size: 1.1rem;
        color: #a5f3fc;
    }

    .warning-box {
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.3);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #fbbf24;
        font-family: 'Space Mono', monospace;
    }

    .info-box {
        background: rgba(56,189,248,0.06);
        border: 1px solid rgba(56,189,248,0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.82rem;
        color: #7dd3fc;
        font-family: 'Space Mono', monospace;
    }

    .footer-text {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #334155;
        text-align: center;
        margin-top: 3rem;
        letter-spacing: 0.06em;
    }

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(56,189,248,0.25);
        border-radius: 14px;
        padding: 1rem;
        background: rgba(56,189,248,0.03);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CLASS NAMES (23 classes)
# ─────────────────────────────────────────────
CLASSES = [
    'Clams', 'Corals', 'Crabs', 'Dolphin', 'Eel',
    'Fish', 'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus',
    'Otter', 'Penguin', 'Puffers', 'Sea Rays', 'Sea Urchins',
    'Seahorse', 'Seal', 'Sharks', 'Shrimp', 'Squid',
    'Starfish', 'Turtle_Tortoise', 'Whale'
]

CLASS_ICONS = {
    'Clams': '🐚', 'Corals': '🪸', 'Crabs': '🦀', 'Dolphin': '🐬',
    'Eel': '〰️', 'Fish': '🐟', 'Jelly Fish': '🪼', 'Lobster': '🦞',
    'Nudibranchs': '🐛', 'Octopus': '🐙', 'Otter': '🦦', 'Penguin': '🐧',
    'Puffers': '🐡', 'Sea Rays': '🌊', 'Sea Urchins': '🦔', 'Seahorse': '🐴',
    'Seal': '🦭', 'Sharks': '🦈', 'Shrimp': '🦐', 'Squid': '🦑',
    'Starfish': '⭐', 'Turtle_Tortoise': '🐢', 'Whale': '🐋'
}

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
MODEL_PATH = "efficientnet_sea_model.h5"

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ─────────────────────────────────────────────
# PREPROCESSING — matches EfficientNet training
# ─────────────────────────────────────────────
def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Resize to 224×224 and apply EfficientNet preprocessing
    (scales pixels to [-1, 1] range, NOT /255).
    """
    img = pil_img.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # EfficientNet expects tf.keras.applications.efficientnet.preprocess_input
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="hero-title">🌊 Sea Animal Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">EfficientNetB0 · 23 Classes · TensorFlow</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL STATUS
# ─────────────────────────────────────────────
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.error(f"❌ Model file `{MODEL_PATH}` not found. Make sure it is in the same directory as app.py.")
    st.stop()
else:
    st.markdown('<div class="info-box">✅ Model loaded successfully — ready to classify</div>', unsafe_allow_html=True)
    st.write("")

# ─────────────────────────────────────────────
# FILE UPLOADER
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a sea animal image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

if uploaded_file is not None:
    # Show image
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # ── PREDICT ──
    with st.spinner("Analyzing..."):
        processed = preprocess_image(pil_img)
        predictions = model.predict(processed, verbose=0)[0]      # shape: (23,)

    top_n = 5
    top_indices = np.argsort(predictions)[::-1][:top_n]
    top_classes = [CLASSES[i] for i in top_indices]
    top_probs   = [float(predictions[i]) * 100 for i in top_indices]

    best_class = top_classes[0]
    best_conf  = top_probs[0]
    best_icon  = CLASS_ICONS.get(best_class, "🌊")

    # ── RESULT CARD ──
    st.markdown(f"""
    <div class="card">
        <div class="pred-label">🔍 Prediction</div>
        <div class="pred-value">{best_icon} {best_class}</div>
        <br/>
        <div class="pred-label">Confidence</div>
        <div class="conf-value">{best_conf:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Low confidence warning
    if best_conf < 50:
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ Low confidence ({best_conf:.1f}%) — the model is uncertain.
            The correct answer may be among the Top 5 predictions below.
        </div>
        """, unsafe_allow_html=True)
        st.write("")

    # ── TOP 5 CHART ──
    st.markdown("### Top 5 Predictions")

    colors = ["#38bdf8" if i == 0 else "#1e3a5f" for i in range(top_n)]

    fig = go.Figure(go.Bar(
        x=top_probs[::-1],
        y=top_classes[::-1],
        orientation="h",
        marker=dict(color=colors[::-1], line=dict(width=0)),
        text=[f"{p:.1f}%" for p in top_probs[::-1]],
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11, family="Space Mono"),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#64748b", size=10, family="Space Mono"),
            title=dict(text="Confidence (%)", font=dict(color="#64748b", size=11)),
            range=[0, max(top_probs) * 1.2]
        ),
        yaxis=dict(
            tickfont=dict(color="#94a3b8", size=12, family="Syne"),
        ),
        margin=dict(l=10, r=60, t=10, b=30),
        height=260,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── FULL PROBABILITY TABLE (expandable) ──
    with st.expander("📊 View all class probabilities"):
        all_sorted = sorted(zip(CLASSES, predictions), key=lambda x: x[1], reverse=True)
        for rank, (cls, prob) in enumerate(all_sorted, 1):
            icon = CLASS_ICONS.get(cls, "•")
            pct  = prob * 100
            bar_fill = int(pct / 100 * 20)
            bar = "█" * bar_fill + "░" * (20 - bar_fill)
            st.markdown(
                f"<span style='font-family:Space Mono;font-size:0.78rem;color:#64748b;'>"
                f"#{rank:02d} {icon} <b style='color:#94a3b8;'>{cls:<18}</b> "
                f"<span style='color:#38bdf8;'>{bar}</span> {pct:5.2f}%"
                f"</span>",
                unsafe_allow_html=True
            )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer-text">
    Model: EfficientNetB0 &nbsp;·&nbsp; Input: 224×224 &nbsp;·&nbsp;
    Classes: 23 &nbsp;·&nbsp; Built with TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
            
