import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🔬",
    layout="centered"
)

# ============================================================
# CUSTOM CSS — simple dark theme
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');

html, body {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #111816;
    color: #E8F0EC;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    text-align: center;
    font-size: 2.4rem !important;
    margin-bottom: 8px !important;
}

[data-testid="stCaptionContainer"] {
    color: #AFC8BE !important;
    text-align: center;
}

h2 {
    font-family: 'Manrope', sans-serif !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

[data-testid="stFileUploader"] {
    background-color: #1A2420;
    border: 2px dashed #2E7D32;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
}

[data-testid="stFileUploader"] label {
    color: #E8F0EC !important;
}

[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

p, label {
    color: #D5E1DC;
}

.unknown-box {
    background: #1A2420;
    border: 1px solid #554B2A;
    border-top: 4px solid #EAB308;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 25px;
}

.unknown-title {
    color: #EAB308;
    font-size: 25px;
    font-weight: 800;
    margin-bottom: 12px;
}

.unknown-text {
    color: #D5E1DC;
    font-size: 15px;
    line-height: 1.6;
}

.diagnosis-box {
    background: #1A2420;
    border: 1px solid #2B3934;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 25px;
}

.diagnosis-title {
    color: #AFC8BE;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

.diagnosis-name {
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 14px;
}

.diagnosis-info {
    color: #D5E1DC;
    font-size: 15px;
    margin: 7px 0;
}

.confidence-bar {
    background: #2A3531;
    height: 9px;
    border-radius: 10px;
    margin-top: 15px;
    overflow: hidden;
}

.confidence-fill {
    height: 100%;
    border-radius: 10px;
}

.symptom-box {
    background: #1A2420;
    border: 1px solid #2B3934;
    border-radius: 14px;
    padding: 18px 22px;
    text-align: left;
    margin-top: -10px;
    margin-bottom: 25px;
    color: #D5E1DC;
    font-size: 14.5px;
    line-height: 1.6;
}

.symptom-label {
    color: #AFC8BE;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
    display: block;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.title("🌱 AgroDetect")
st.caption(
    "Upload a crop leaf image and receive disease detection, "
    "expected symptoms, and treatment recommendations."
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/crop_disease_model.keras")

model = load_model()

# ============================================================
# DISEASE LABELS — 16 classes, 'Other' included (index 0)
# Must match the order printed by evaluate.py / train.py exactly.
# ============================================================

import json

with open("models/class_names.json", "r") as f:
    disease_labels = json.load(f)

# ============================================================
# DISEASE INFORMATION — severity, symptoms, treatment, improvements
# Symptom text adapted from the project report's Table 2 (crop leaf
# diseases with symptoms and pathogen categories).
# ============================================================

disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "severity": "Moderate", "pathogen": "Bacterial",
        "symptoms": "Small, dark, water-soaked spots on leaves that may have a yellow halo; spots can merge and cause leaves to drop.",
        "treatment": ["Remove infected leaves", "Avoid overhead watering", "Apply copper-based bactericide"],
        "improvements": ["Improve air circulation", "Avoid wet foliage", "Monitor plants weekly"]
    },
    "Pepper__bell___healthy": {
        "severity": "None", "pathogen": "—",
        "symptoms": "No visible spots, discoloration, or lesions. Leaf color and texture appear uniform.",
        "treatment": ["No treatment required"],
        "improvements": ["Maintain watering schedule", "Continue regular monitoring", "Apply balanced fertilizer"]
    },
    "Potato___Early_blight": {
        "severity": "Moderate", "pathogen": "Fungal",
        "symptoms": "Dark concentric rings ('target spot' pattern) surrounded by a yellow halo, usually starting on older, lower leaves.",
        "treatment": ["Remove infected leaves", "Apply fungicide", "Reduce leaf wetness"],
        "improvements": ["Increase plant spacing", "Improve drainage", "Add organic compost"]
    },
    "Potato___Late_blight": {
        "severity": "High", "pathogen": "Fungal",
        "symptoms": "Large, irregular dark brown or black patches that spread rapidly, often with a pale green-yellow border; leaves may rot quickly in humid conditions.",
        "treatment": ["Remove infected plants", "Apply fungicide immediately", "Inspect nearby plants"],
        "improvements": ["Improve airflow", "Avoid excess moisture", "Practice crop rotation"]
    },
    "Potato___healthy": {
        "severity": "None", "pathogen": "—",
        "symptoms": "No visible spots, discoloration, or lesions. Leaf color and texture appear uniform.",
        "treatment": ["No treatment required"],
        "improvements": ["Maintain healthy watering schedule", "Apply fertilizer as needed", "Continue monitoring"]
    },
    "Tomato_Bacterial_spot": {
        "severity": "Moderate", "pathogen": "Bacterial",
        "symptoms": "Small, dark, greasy-looking spots on leaves and stems, sometimes with a yellow halo; spots may crack in the center.",
        "treatment": ["Remove infected leaves", "Apply copper spray", "Avoid working with wet plants"],
        "improvements": ["Improve air circulation", "Reduce leaf moisture", "Inspect regularly"]
    },
    "Tomato_Early_blight": {
        "severity": "Moderate", "pathogen": "Fungal",
        "symptoms": "Dark concentric ring spots (target-like pattern), typically starting on lower/older leaves and moving upward.",
        "treatment": ["Remove infected leaves", "Apply fungicide", "Avoid overhead watering"],
        "improvements": ["Improve airflow", "Use mulch", "Monitor weekly"]
    },
    "Tomato_Late_blight": {
        "severity": "High", "pathogen": "Fungal",
        "symptoms": "Large, irregular greasy-looking dark patches that spread quickly across the leaf, often with pale edges; can affect the whole plant within days.",
        "treatment": ["Remove infected plants immediately", "Apply fungicide", "Separate infected plants"],
        "improvements": ["Improve drainage", "Reduce humidity", "Inspect nearby crops"]
    },
    "Tomato_Leaf_Mold": {
        "severity": "Moderate", "pathogen": "Fungal",
        "symptoms": "Pale green or yellow patches on the upper leaf surface, with an olive-green to grayish-purple fuzzy mold visible on the underside.",
        "treatment": ["Remove affected leaves", "Improve ventilation", "Use fungicide if necessary"],
        "improvements": ["Reduce humidity", "Increase spacing", "Prune dense foliage"]
    },
    "Tomato_Septoria_leaf_spot": {
        "severity": "Moderate", "pathogen": "Fungal",
        "symptoms": "Many small, circular spots with dark borders and lighter gray centers, usually starting on lower leaves.",
        "treatment": ["Remove infected leaves", "Apply fungicide", "Avoid splashing water"],
        "improvements": ["Use mulch", "Improve airflow", "Rotate crops"]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "severity": "Moderate", "pathogen": "Pest (mite)",
        "symptoms": "Fine yellow/white speckling on leaves, sometimes with faint webbing on the underside; leaves may look dry or bronzed.",
        "treatment": ["Spray leaves with water", "Apply insecticidal soap", "Remove heavily affected leaves"],
        "improvements": ["Increase humidity", "Inspect undersides of leaves", "Monitor regularly"]
    },
    "Tomato__Target_Spot": {
        "severity": "Moderate", "pathogen": "Fungal",
        "symptoms": "Brown lesions with concentric rings, similar to early blight but often with a more defined target pattern and can appear on stems and fruit too.",
        "treatment": ["Remove infected foliage", "Apply fungicide", "Avoid overcrowding"],
        "improvements": ["Increase spacing", "Improve drainage", "Monitor symptoms"]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "severity": "High", "pathogen": "Viral",
        "symptoms": "Upward curling and yellowing of leaves, stunted plant growth, and reduced leaf size — spread mainly by whiteflies.",
        "treatment": ["Remove infected plants", "Control whiteflies", "Prevent spread immediately"],
        "improvements": ["Use resistant varieties", "Monitor pests", "Keep field clean"]
    },
    "Tomato__Tomato_mosaic_virus": {
        "severity": "High", "pathogen": "Viral",
        "symptoms": "Mottled light and dark green mosaic pattern on leaves, sometimes with leaf curling or a fern-like distorted shape.",
        "treatment": ["Remove infected plants", "Disinfect tools", "Avoid plant-to-plant contact"],
        "improvements": ["Use certified seeds", "Control contamination", "Inspect plants frequently"]
    },
    "Tomato_healthy": {
        "severity": "None", "pathogen": "—",
        "symptoms": "No visible spots, discoloration, or lesions. Leaf color and texture appear uniform.",
        "treatment": ["No treatment required"],
        "improvements": ["Maintain balanced nutrition", "Continue monitoring", "Keep regular watering schedule"]
    }
}

# ============================================================
# UPLOAD IMAGE
# ============================================================

st.markdown("## 📤 Upload Leaf Image")
uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded:
    st.markdown("## 🖼️ Leaf Preview")
    st.image(uploaded, width="stretch")

    # ── Read image ──
    file_bytes = np.asarray(bytearray(uploaded.getvalue()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Unable to read this image. Please upload a valid JPG or PNG file.")
        st.stop()

    # ── Basic size check ──
    height, width = img.shape[:2]
    if height < 100 or width < 100:
        st.warning("⚠️ The image is too small. Please upload a clearer crop leaf image.")
        st.stop()

    # ── Convert to RGB for the model ──
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ── Model input ──
    model_img = cv2.resize(rgb_img, (224, 224)).astype("float32")
    model_img = np.expand_dims(model_img, axis=0)

    with st.spinner("Analyzing leaf..."):
        result = model.predict(model_img, verbose=0)[0]

        st.write("### Debug Predictions")
        for label, score in zip(disease_labels, result):
            st.write(f"{label}: {score * 100:.2f}%")

    top_index = int(np.argmax(result))
    predicted_class = disease_labels[top_index]

    confidence = float(np.max(result))
    top_confidence = confidence * 100

    UNKNOWN_THRESHOLD = 0.75
    if confidence < UNKNOWN_THRESHOLD:
        predicted_class = "Unknown / Not Supported"

    sorted_probs = np.sort(result)
    top1 = sorted_probs[-1]
    top2 = sorted_probs[-2]
    margin = top1 - top2

    MIN_CONFIDENCE = 0.80
    MIN_MARGIN = 0.20
    if top1 < MIN_CONFIDENCE or margin < MIN_MARGIN:
        predicted_class = "Unknown / Not Supported"

    st.markdown("## 🔍 Diagnosis")

    # ── Model itself judged this "Other" — not a supported leaf/condition ──
    if predicted_class in ["Other", "Unknown / Not Supported"]:
        st.markdown(f"""
        <div class="unknown-box">
            <div class="unknown-title">⚠️ Unknown / Not Supported</div>
            <div class="unknown-text">
                The model identified this image as outside the supported crop conditions
                (confidence: {top_confidence:.1f}%).
                <br><br>
                Please upload a clear image of a <b>tomato, potato, or pepper leaf</b> from a supported condition.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("💡 Supported crops: Tomato, Potato, and Pepper.")

    # ── Valid result ──
    else:
        is_healthy = "healthy" in predicted_class.lower()
        status = "Healthy" if is_healthy else "Diseased"
        display_name = predicted_class.replace("___", " ").replace("__", " ").replace("_", " ")
        info = disease_info.get(predicted_class, {
            "severity": "Unknown", "pathogen": "Unknown",
            "symptoms": "Not available for this class.",
            "treatment": ["Consult an agricultural specialist"],
            "improvements": ["Monitor plant condition"]
        })
        result_color = "#34D399" if is_healthy else "#F0973B"

        st.markdown(f"""
        <div class="diagnosis-box" style="border-top: 4px solid {result_color};">
            <div class="diagnosis-title">Diagnosis Result</div>
            <div class="diagnosis-name" style="color: {result_color};">{display_name}</div>
            <div class="diagnosis-info"><b>Status:</b> {status}</div>
            <div class="diagnosis-info"><b>Confidence:</b> {top_confidence:.1f}%</div>
            <div class="diagnosis-info"><b>Severity:</b> {info['severity']}</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {min(top_confidence, 100):.2f}%; background: {result_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="symptom-box">
            <span class="symptom-label">Typical Symptoms</span>
            {info['symptoms']}
        </div>
        """, unsafe_allow_html=True)

        if not is_healthy:
            st.markdown("## 💊 Treatment Recommendations")
            for treatment in info["treatment"]:
                st.success(treatment)

        st.markdown("## 🌱 Plant Health Improvement Plan")
        col1, col2 = st.columns(2)
        for i, tip in enumerate(info["improvements"]):
            with col1 if i % 2 == 0 else col2:
                st.info(tip)