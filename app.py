import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌊 Sea Creature Classifier",
    page_icon="🐠",
    layout="centered",
)

# ── Class labels (23 sea creature classes) ─────────────────────────────────────
CLASS_NAMES = [
    "Clams",
    "Corals",
    "Crabs",
    "Dolphin",
    "Eel",
    "Fish",
    "Jelly Fish",
    "Lobster",
    "Nudibranchs",
    "Octopus",
    "Otter",
    "Penguin",
    "Puffers",
    "Sea Rays",
    "Sea Urchins",
    "Seahorse",
    "Seal",
    "Sharks",
    "Shrimp",
    "Squid",
    "Starfish",
    "Turtle_Tortoise",
    "Whale",
]

# ── Model loading ──────────────────────────────────────────────────────────────
MODEL_PATH = "efficientnet_sea_model.h5"
IMG_SIZE = (224, 224)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"❌ Model file `{MODEL_PATH}` not found.\n\n"
            "Make sure the file is in the same directory as `app.py`."
        )
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalize, and expand dims for model input."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image):
    tensor = preprocess_image(image)
    preds = model.predict(tensor, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return CLASS_NAMES[top_idx], float(preds[top_idx]), preds


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🌊 Sea Creature Classifier")
st.markdown(
    "Upload an image of a sea creature and the model will identify it from **23 ocean species**."
)

model = load_model()

uploaded = st.file_uploader(
    "Choose an image…",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Classifying…"):
            label, confidence, all_preds = predict(model, image)

        st.markdown("### 🔍 Prediction")
        st.success(f"**{label}**")
        st.metric("Confidence", f"{confidence * 100:.1f}%")

        # Top-5 predictions bar chart
        st.markdown("#### Top 5 Predictions")
        top5_idx = np.argsort(all_preds)[::-1][:5]
        top5_labels = [CLASS_NAMES[i] for i in top5_idx]
        top5_scores = [float(all_preds[i]) * 100 for i in top5_idx]

        chart_data = {
            "Species": top5_labels,
            "Confidence (%)": top5_scores,
        }

        import pandas as pd

        df = pd.DataFrame(chart_data).set_index("Species")
        st.bar_chart(df, horizontal=True)

else:
    st.info("👆 Upload a JPG, PNG, or WebP image to get started.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: EfficientNet · Input: 224 × 224 · Classes: 23 · "
    "Built with TensorFlow & Streamlit"
)
