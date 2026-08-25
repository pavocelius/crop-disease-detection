import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌱", layout="centered")

# ── Custom styling injected directly into Streamlit ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@700;800&family=Nunito+Sans:wght@400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Nunito Sans', sans-serif;
}

.stApp {
    background-color: #FBF7EC;
}

h1 {
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    color: #1F4A29;
    text-align: center;
}

.subtitle {
    text-align: center;
    color: #5B6653;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #2F6B3C;
    border-radius: 20px;
    padding: 20px;
}

.result-card {
    background: white;
    border-radius: 20px;
    padding: 24px;
    margin-top: 20px;
    border: 2px solid #E4EFDF;
    text-align: center;
}
.result-card.healthy { border-color: #2F6B3C; background: #E4EFDF; }
.result-card.diseased { border-color: #D98A2B; background: #FBEBD4; }

.result-label {
    font-family: 'Nunito', sans-serif;
    font-weight: 800;
    font-size: 1.6rem;
    margin-bottom: 6px;
}
.result-label.healthy { color: #1F4A29; }
.result-label.diseased { color: #8A4A17; }

.result-confidence {
    color: #5B6653;
    font-size: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌱 Crop Disease Detection</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a leaf photo to check if your crop is healthy.</p>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/crop_disease_model.keras")

model = load_model()

# Full class list needed internally to map prediction index to a label
disease_labels = [...]  # paste your real class list here, same order as training printed

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded:
    st.image(uploaded, use_container_width=True)

    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype("float32")
    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing leaf..."):
        result = model.predict(img)[0]

    pred_idx = int(np.argmax(result))
    predicted_class = disease_labels[pred_idx]
    pct = float(np.max(result)) * 100

    is_healthy = "healthy" in predicted_class.lower()
    status = "Healthy" if is_healthy else "Diseased"
    css_class = "healthy" if is_healthy else "diseased"

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-label {css_class}">{status}</div>
        <div class="result-confidence">Confidence: {pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)