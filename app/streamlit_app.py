import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.set_page_config(page_title="Crop Disease Detection", page_icon="🔬", layout="centered")

# ── Custom styling: clean, modern agri-tech look ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0B1F1A 0%, #0F2922 320px, #F7FAF8 320px);
}

.hero-wrap {
    padding: 10px 0 4px;
}

.badge-row {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-bottom: 14px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(52, 211, 153, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.35);
    color: #34D399;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: 0.78rem;
    padding: 6px 14px;
    border-radius: 999px;
}

h1 {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    color: #F2FBF7;
    text-align: center;
    font-size: 2.1rem !important;
    margin-bottom: 6px !important;
}

.subtitle {
    text-align: center;
    color: #A9C4BA;
    font-size: 1rem;
    margin-bottom: 28px;
}

.stat-row {
    display: flex;
    justify-content: center;
    gap: 32px;
    margin-bottom: 30px;
}
.stat {
    text-align: center;
}
.stat-num {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.5rem;
    color: #34D399;
}
.stat-label {
    color: #A9C4BA;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 1px solid #E1E8E4;
    border-radius: 16px;
    padding: 22px;
    box-shadow: 0 8px 30px rgba(15, 41, 34, 0.08);
}

.result-card {
    background: white;
    border-radius: 16px;
    padding: 26px;
    margin-top: 20px;
    border: 1px solid #E1E8E4;
    box-shadow: 0 8px 30px rgba(15, 41, 34, 0.08);
    text-align: center;
}
.result-card.healthy { border-top: 4px solid #34D399; }
.result-card.diseased { border-top: 4px solid #F0973B; }

.result-tag {
    display: inline-block;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6B7A73;
    margin-bottom: 8px;
}

.result-label {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    margin-bottom: 10px;
}
.result-label.healthy { color: #0F7A54; }
.result-label.diseased { color: #B85A16; }

.result-confidence {
    color: #6B7A73;
    font-size: 0.95rem;
}

.conf-track {
    background: #EEF3F0;
    border-radius: 999px;
    height: 8px;
    margin-top: 12px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 999px;
}
.conf-fill.healthy { background: #34D399; }
.conf-fill.diseased { background: #F0973B; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <div class="badge-row">
        <span class="badge">AI-Powered</span>
        <span class="badge">Instant Diagnosis</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1>Crop Disease Detection</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a leaf photo for an instant, model-based health check.</p>', unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
    <div class="stat"><div class="stat-num">15</div><div class="stat-label">Classes Trained</div></div>
    <div class="stat"><div class="stat-num">MobileNetV2</div><div class="stat-label">Model</div></div>
    <div class="stat"><div class="stat-num">CNN</div><div class="stat-label">Architecture</div></div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/crop_disease_model.keras")

model = load_model()

# Full class list needed internally to map prediction index to a label
disease_labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

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
        <div class="result-tag">Diagnosis Result</div>
        <div class="result-label {css_class}">{status}</div>
        <div class="result-confidence">{pct:.1f}% confidence</div>
        <div class="conf-track"><div class="conf-fill {css_class}" style="width:{pct}%;"></div></div>
    </div>
    """, unsafe_allow_html=True)