import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title="Crop Disease Detection System",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "models/crop_disease_model_efficientnet.keras"

CLASS_NAMES = [
    "Other",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# Keep this dictionary aligned with the labels used by the trained model.
DISEASE_INFO = {
    "Tomato_healthy": {
        "title": "Tomato — Healthy",
        "category": "Healthy",
        "symptoms": "No major visible disease symptoms detected.",
        "action": "Continue normal crop monitoring and maintain good field hygiene.",
    },
    "Potato___Early_blight": {
        "title": "Potato — Early Blight",
        "category": "Fungal disease",
        "symptoms": "Dark lesions and target-like patterns may appear on leaves.",
        "action": "Remove severely affected leaves and improve airflow. Seek local agricultural guidance for treatment.",
    },
    "Potato___Late_blight": {
        "title": "Potato — Late Blight",
        "category": "Fungal disease",
        "symptoms": "Dark, water-soaked-looking lesions may occur on leaves.",
        "action": "Remove affected material and avoid prolonged leaf wetness. Confirm with an agricultural professional.",
    },
    "Tomato_Bacterial_spot": {
        "title": "Tomato — Bacterial Spot",
        "category": "Bacterial disease",
        "symptoms": "Small dark spots can develop on leaves.",
        "action": "Reduce leaf wetness, avoid unnecessary handling of plants, and remove badly affected material.",
    },
    "Tomato_Early_blight": {
        "title": "Tomato — Early Blight",
        "category": "Fungal disease",
        "symptoms": "Dark lesions with concentric or target-like patterns may appear.",
        "action": "Improve air circulation, avoid overhead watering, and remove badly affected leaves.",
    },
    "Tomato_Late_blight": {
        "title": "Tomato — Late Blight",
        "category": "Fungal disease",
        "symptoms": "Dark lesions can spread rapidly under favourable conditions.",
        "action": "Remove affected material and obtain professional confirmation before treatment.",
    },
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

def prepare_image(image):
    image = image.convert("RGB").resize((224, 224))
    arr = np.asarray(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

# Modern light plant theme + responsive mobile layout.
st.markdown("""
<style>
:root {
    color-scheme: light;
}

.stApp {
    background:
        radial-gradient(circle at 8% 4%, rgba(184,216,180,.28), transparent 25%),
        radial-gradient(circle at 92% 12%, rgba(230,244,228,.55), transparent 24%),
        #F7FBF5;
    color: #17351F;
}

[data-testid="stHeader"] {
    background: rgba(247,251,245,0.90);
}

.block-container {
    max-width: 1050px;
    padding-top: 1rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

/* ---------- Hero ---------- */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #DCEFD9 0%, #F8FBEF 58%, #FFFFFF 100%);
    border: 1px solid #B8D8B4;
    border-radius: 24px;
    padding: 2rem 1.6rem 1.7rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 28px rgba(32,92,54,.10);
}

.hero::after {
    content: "🌱";
    position: absolute;
    right: 2rem;
    top: .7rem;
    font-size: 5rem;
    opacity: .15;
    transform: rotate(-12deg);
}

.hero-badge {
    display: inline-block;
    background: #FFFFFF;
    color: #3F7D3A;
    border: 1px solid #B8D8B4;
    border-radius: 999px;
    padding: .35rem .7rem;
    font-size: .78rem;
    font-weight: 800;
    letter-spacing: .04em;
    margin-bottom: .7rem;
}

.hero h1 {
    color: #205C36;
    margin: 0 0 .35rem 0;
    font-size: clamp(2rem, 6vw, 3rem);
    line-height: 1.1;
}

.hero p {
    color: #365A3E;
    margin: 0;
    max-width: 650px;
    font-size: 1rem;
    line-height: 1.6;
}

/* ---------- Cards ---------- */
.card {
    background: rgba(255,255,255,.96);
    border: 1px solid #C9DDC6;
    border-radius: 18px;
    padding: 1.1rem;
    margin: .8rem 0;
    box-shadow: 0 5px 18px rgba(32,92,54,.07);
}

.section-title {
    color: #205C36;
    font-size: 1.15rem;
    font-weight: 800;
    margin-bottom: .25rem;
}

.small {
    color: #52705A;
    font-size: .9rem;
    line-height: 1.55;
}

/* ---------- Feature cards ---------- */
.feature-card {
    background: #FFFFFF;
    border: 1px solid #D2E3CF;
    border-radius: 16px;
    padding: .9rem;
    min-height: 100%;
    box-shadow: 0 3px 12px rgba(32,92,54,.05);
}

.feature-icon {
    font-size: 1.45rem;
    margin-bottom: .25rem;
}

.feature-title {
    color: #205C36;
    font-weight: 800;
    font-size: .96rem;
}

.feature-text {
    color: #52705A;
    font-size: .82rem;
    margin-top: .2rem;
    line-height: 1.4;
}

/* ---------- Upload area ---------- */
.upload-card {
    background: linear-gradient(180deg, #FFFFFF, #F8FCF6);
    border: 2px solid #B8D8B4;
    border-radius: 20px;
    padding: 1rem;
    margin-top: 1rem;
    box-shadow: 0 6px 20px rgba(32,92,54,.08);
}

.upload-heading {
    color: #205C36;
    font-size: 1.2rem;
    font-weight: 800;
    margin-bottom: .2rem;
}

/* Streamlit uploader - high contrast and visible on light theme */
[data-testid="stFileUploader"] {
    width: 100%;
}

[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF !important;
    border: 2px dashed #3F7D3A !important;
    border-radius: 15px !important;
    min-height: 145px !important;
}

[data-testid="stFileUploaderDropzone"] > div {
    color: #17351F !important;
}

[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: #3F7D3A !important;
    color: #FFFFFF !important;
    border: 1px solid #3F7D3A !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    box-shadow: none !important;
}

[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    background: #205C36 !important;
    color: #FFFFFF !important;
    border-color: #205C36 !important;
}

[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button span,
[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    color: #FFFFFF !important;
}

[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderDropzone"] small {
    color: #52705A !important;
}

[data-testid="stFileUploader"] label {
    color: #17351F !important;
    font-weight: 700 !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    width: 100%;
    min-height: 3rem;
    border-radius: 12px;
    font-weight: 800;
    border: 1px solid #3F7D3A;
}

/* ---------- Prediction result ---------- */
.result {
    background: linear-gradient(135deg, #F0F8ED, #FFFFFF);
    border: 1px solid #B8D8B4;
    border-left: 6px solid #3F7D3A;
}

.result-healthy {
    border-left-color: #3F7D3A;
}

.result-warning {
    border-left-color: #D08B28;
}

.result-label {
    display: inline-block;
    background: #E6F4E4;
    color: #205C36;
    border-radius: 999px;
    padding: .3rem .65rem;
    font-size: .75rem;
    font-weight: 800;
    margin-bottom: .45rem;
}

.metric {
    font-size: clamp(1.35rem, 4vw, 1.8rem);
    font-weight: 800;
    color: #205C36;
    line-height: 1.2;
}

.confidence-track {
    width: 100%;
    height: 10px;
    background: #DCE8D9;
    border-radius: 999px;
    overflow: hidden;
    margin: .45rem 0 .7rem;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #72A968, #3F7D3A);
    border-radius: 999px;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #6B816F;
    font-size: .78rem;
    padding: 1rem .5rem .25rem;
}

/* ---------- Mobile ---------- */
@media (max-width: 640px) {
    .block-container {
        padding: .65rem .55rem 1.2rem;
    }

    .hero {
        padding: 1.25rem 1rem 1.2rem;
        border-radius: 18px;
    }

    .hero::after {
        right: .5rem;
        top: .35rem;
        font-size: 3.5rem;
    }

    .hero p {
        font-size: .92rem;
        max-width: 85%;
    }

    .card,
    .upload-card {
        padding: .85rem;
        border-radius: 15px;
    }

    [data-testid="stFileUploaderDropzone"] {
        min-height: 130px !important;
    }

    .metric {
        font-size: 1.35rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <div class="hero-badge">🌿 FINAL YEAR PROJECT</div>
    <h1>Crop Disease Detection System</h1>
    <p>Upload a clear crop leaf image and let the trained model analyse the condition of your crop!</p>
</div>
""", unsafe_allow_html=True)

# ---------- Feature row ----------
f1, f2, f3 = st.columns(3)

with f1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">⚡</div>
        <div class="feature-title">Fast Analysis</div>
        <div class="feature-text">Get a prediction in seconds.</div>
    </div>
    """, unsafe_allow_html=True)

with f2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <div class="feature-title">Smart Detection</div>
        <div class="feature-text">EfficientNet-B0 image classification.</div>
    </div>
    """, unsafe_allow_html=True)

with f3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-icon">🌱</div>
        <div class="feature-title">16 Classes</div>
        <div class="feature-text">Supported crop health conditions.</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- How to use ----------
st.markdown("""
<div class="card">
    <div class="section-title">How it works</div>
    <div class="small">
        <b>01</b> Take a clear photo of one leaf &nbsp;→&nbsp;
        <b>02</b> Upload it &nbsp;→&nbsp;
        <b>03</b> Analyse the image &nbsp;→&nbsp;
        <b>04</b> Review the prediction and suggested action.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------- Upload ----------
st.markdown("""
<div class="upload-card">
    <div class="upload-heading">📷 Upload a Leaf Image</div>
    <div class="small">Use JPG or PNG. For best results, use one clear leaf with good lighting and minimal background clutter.</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="For best results, use one clear leaf with good lighting.",
)

if uploaded:
    image = Image.open(uploaded)

    # Keep uploaded images visually compact on desktop while allowing them to
    # scale naturally on smaller screens.
    preview = image.copy()
    preview.thumbnail((560, 560))

    st.markdown('<div class="card">', unsafe_allow_html=True)
    image_col_left, image_col, image_col_right = st.columns([1, 2, 1])
    with image_col:
        st.image(preview, caption="Leaf image ready for analysis", width=520)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:.2rem'></div>", unsafe_allow_html=True)

    if st.button("🔍  Analyse Leaf", type="primary"):
        try:
            with st.spinner("🌿 Analysing your leaf..."):
                model = load_model()
                x = prepare_image(image)
                probs = model.predict(x, verbose=0)[0]
                idx = int(np.argmax(probs))
                confidence = float(probs[idx]) * 100
                label = CLASS_NAMES[idx]

            info = DISEASE_INFO.get(label, {
                "title": label.replace("_", " "),
                "category": "Supported class",
                "symptoms": "Refer to the class name and verify the result with an agricultural professional.",
                "action": "Use the prediction as a screening aid rather than a definitive diagnosis.",
            })

            # Use native Streamlit components for the prediction details.
            # This avoids raw HTML tags appearing in the result on different
            # Streamlit versions while keeping the visual design polished.
            with st.container(border=True):
                st.markdown("<div class='result-label'>🌿 PREDICTION RESULTS </div>", unsafe_allow_html=True)
                st.markdown("<div class='small'>Predicted condition</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric'>{info['title']}</div>", unsafe_allow_html=True)

                st.write(f"**Confidence:** {confidence:.2f}%")
                st.progress(min(confidence / 100.0, 1.0), text=f"Model confidence: {confidence:.2f}%")

                st.write(f"**Category:** {info['category']}")
                st.write(f"**Visible indicators:** {info['symptoms']}")
                st.write(f"**Suggested action:** {info['action']}")

            if label == "Other":
                st.warning(
                    "The image was classified as Other. Try a clearer image of a single supported crop leaf. "
                    "The system should not be used to diagnose crops outside its trained classes."
                )

            st.markdown("""
            <div class="card">
                <div class="section-title">💡 What to do next...</div>
                <div class="small">
                    Treat this prediction as a screening aid. Compare the result with the visible symptoms
                    and confirm important agricultural decisions with a qualified professional.
                </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error("The image could not be analysed. Check that the model file and class list are correct.")
            st.caption(str(e))

# ---------- Disclaimer + footer ----------
st.markdown("""
<div class="card small">
    <b>Important:</b> This application is a machine-learning decision-support prototype.
    Predictions can be affected by image quality, lighting, background, crop variety and diseases not represented
    in the training data. Confirm important agricultural decisions with a qualified professional.
</div>

<div class="footer">
    🌿 Final Year Project &nbsp;•&nbsp; EfficientNet-B0 &nbsp;•&nbsp; UniMAP
</div>
""", unsafe_allow_html=True)
