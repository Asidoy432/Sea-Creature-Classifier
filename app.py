import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

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
    html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
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
        margin-bottom: 1rem;
    }
    .danger-box {
        background: rgba(239,68,68,0.08);
        border: 1px solid rgba(239,68,68,0.35);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        font-size: 0.88rem;
        color: #fca5a5;
        font-family: 'Space Mono', monospace;
        margin-bottom: 1rem;
        text-align: center;
    }
    .danger-icon { font-size: 2.5rem; margin-bottom: 0.4rem; }
    .danger-title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.3rem;
        color: #f87171;
        margin-bottom: 0.5rem;
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
# NON-SEA-ANIMAL KEYWORDS
# Used to detect clearly non-ocean images by filename hint
# The real guard is entropy — keywords are a secondary soft signal
# ─────────────────────────────────────────────
NON_SEA_KEYWORDS = [
    'cat', 'dog', 'car', 'human', 'person', 'face', 'food',
    'building', 'tree', 'flower', 'bird', 'insect', 'house',
    'landscape', 'sky', 'mountain', 'city', 'indoor'
]

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
MODEL_PATH = "efficientnet_sea_model_improved.h5"

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

# ─────────────────────────────────────────────
# PREPROCESSING — correct EfficientNet scaling
# ─────────────────────────────────────────────
def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ─────────────────────────────────────────────
# NOT-A-SEA-ANIMAL DETECTION
# Only fires when BOTH conditions are true:
#   1. Top confidence extremely low  (< 5%)
#   2. Entropy nearly maximum        (> 0.98 of max)
# This is intentionally strict so real sea animals are never blocked.
# ─────────────────────────────────────────────
def is_not_sea_animal(predictions):
    top_conf = float(np.max(predictions)) * 100

    # Entropy: uniform distribution across 23 classes = totally random
    probs = predictions + 1e-9
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(CLASSES))
    normalized_entropy = entropy / max_entropy  # 0 = certain, 1 = random

    # Only reject if model is almost completely clueless
    return top_conf < 5.0 and normalized_entropy > 0.98

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
    st.error(f"❌ Model file `{MODEL_PATH}` not found. Place it in the same directory as app.py.")
    st.stop()
else:
    st.markdown('<div class="info-box">✅ Model loaded — ready to classify</div>', unsafe_allow_html=True)
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
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # ── PREDICT ──
    with st.spinner("Analyzing..."):
        processed   = preprocess_image(pil_img)
        predictions = model.predict(processed, verbose=0)[0]  # shape: (23,)

    # ── NOT A SEA ANIMAL CHECK (strict — only fires on clearly random output) ──
    if is_not_sea_animal(predictions):
        st.markdown("""
        <div class="danger-box">
            <div class="danger-icon">🚫</div>
            <div class="danger-title">Not a Sea Animal</div>
            This image does not appear to contain any of the 23 sea animal
            categories this model was trained on.<br><br>
            Please upload a clear image of a sea creature such as an
            octopus, shark, dolphin, crab, jellyfish, eel, sea urchin, etc.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── TOP 5 ──
    top_n       = 5
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

    # Low-confidence warning (shown but prediction still displayed)
    if best_conf < 50:
        st.markdown(f"""
        <div class="warning-box">
            ⚠️ Low confidence ({best_conf:.1f}%) — model is uncertain.
            The correct answer may appear in the Top 5 below.
        </div>
        """, unsafe_allow_html=True)

    # ── TOP 5 CHART — pure HTML, zero extra deps ──
    st.markdown("### Top 5 Predictions")

    max_prob  = max(top_probs) if max(top_probs) > 0 else 1
    bars_html = ""
    for i, (cls, prob) in enumerate(zip(top_classes, top_probs)):
        icon        = CLASS_ICONS.get(cls, "•")
        bar_pct     = prob / max_prob * 100
        bar_color   = "#38bdf8" if i == 0 else "#1e4a6e"
        label_color = "#e0f2fe" if i == 0 else "#94a3b8"
        bars_html  += f"""
        <div style="display:flex;align-items:center;margin-bottom:10px;gap:10px;">
            <div style="width:145px;text-align:right;font-family:'Space Mono',monospace;
                        font-size:0.76rem;color:{label_color};white-space:nowrap;flex-shrink:0;">
                {icon} {cls}
            </div>
            <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:6px;
                        overflow:hidden;height:20px;">
                <div style="width:{bar_pct:.1f}%;background:{bar_color};height:100%;
                            border-radius:6px;"></div>
            </div>
            <div style="width:50px;font-family:'Space Mono',monospace;font-size:0.74rem;
                        color:{label_color};text-align:right;flex-shrink:0;">
                {prob:.1f}%
            </div>
        </div>"""

    st.markdown(f'<div class="card" style="padding:1.2rem 1.4rem;">{bars_html}</div>',
                unsafe_allow_html=True)

    # ── FULL PROBABILITY TABLE (expandable) ──
    with st.expander("📊 View all 23 class probabilities"):
        all_sorted = sorted(zip(CLASSES, predictions), key=lambda x: x[1], reverse=True)
        for rank, (cls, prob) in enumerate(all_sorted, 1):
            icon     = CLASS_ICONS.get(cls, "•")
            pct      = prob * 100
            bar_fill = int(pct / 100 * 20)
            bar      = "█" * bar_fill + "░" * (20 - bar_fill)
            st.markdown(
                f"<span style='font-family:Space Mono,monospace;font-size:0.76rem;color:#64748b;'>"
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
    Classes: 23 &nbsp;·&nbsp; Built with TensorFlow &amp; Streamlit
</div>
""", unsafe_allow_html=True)
