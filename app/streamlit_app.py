import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title="Crop Disease Detection",
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

# Light theme + plant colours + responsive layout.
st.markdown("""
<style>
:root {
    color-scheme: light;
}
.stApp {
    background: #F7FBF5;
    color: #17351F;
}
[data-testid="stHeader"] {
    background: rgba(247,251,245,0.96);
}
.block-container {
    max-width: 1050px;
    padding-top: 1.4rem;
    padding-left: 1rem;
    padding-right: 1rem;
}
.hero {
    background: linear-gradient(135deg, #E6F4E4, #F8FBEF);
    border: 1px solid #B8D8B4;
    border-radius: 20px;
    padding: 1.3rem 1.2rem;
    margin-bottom: 1rem;
}
.hero h1 {
    color: #205C36;
    margin-bottom: .25rem;
    font-size: clamp(1.7rem, 5vw, 2.7rem);
}
.hero p {
    color: #365A3E;
    margin: 0;
}
.card {
    background: #FFFFFF;
    border: 1px solid #C9DDC6;
    border-radius: 16px;
    padding: 1rem;
    margin: .75rem 0;
    box-shadow: 0 3px 12px rgba(32,92,54,.08);
}
.result {
    background: #F0F8ED;
    border-left: 6px solid #3F7D3A;
}
.metric {
    font-size: 1.55rem;
    font-weight: 700;
    color: #205C36;
}
.small {
    color: #52705A;
    font-size: .92rem;
}
.stButton > button, .stDownloadButton > button {
    width: 100%;
    border-radius: 12px;
    min-height: 2.8rem;
    font-weight: 700;
}
@media (max-width: 640px) {
    .block-container {
        padding: .75rem .65rem 1.5rem;
    }
    .hero {
        padding: 1rem;
        border-radius: 16px;
    }
    .card {
        padding: .85rem;
        border-radius: 14px;
    }
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    img {
        max-width: 100%;
        height: auto;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🌿 Crop Disease Detection</h1>
    <p>Upload a crop leaf image to obtain an AI-assisted classification result.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<b>How to use</b><br>
1. Take a clear photograph of one leaf.<br>
2. Upload the image below.<br>
3. Check the preview and run the prediction.<br>
4. Read the result, confidence and suggested action.
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="For best results, use one clear leaf with good lighting.",
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded leaf image", use_container_width=True)

    if st.button("🔍 Analyse Leaf", type="primary"):
        try:
            with st.spinner("Analysing image..."):
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

            st.markdown(f"""
            <div class="card result">
                <div class="small">Predicted condition</div>
                <div class="metric">{info["title"]}</div>
                <p><b>Confidence:</b> {confidence:.2f}%</p>
                <p><b>Category:</b> {info["category"]}</p>
                <p><b>Visible indicators:</b> {info["symptoms"]}</p>
                <p><b>Suggested action:</b> {info["action"]}</p>
            </div>
            """, unsafe_allow_html=True)

            if label == "Other":
                st.warning(
                    "The image was classified as Other. Try a clearer image of a single supported crop leaf. "
                    "The system should not be used to diagnose crops outside its trained classes."
                )

        except Exception as e:
            st.error("The image could not be analysed. Check that the model file and class list are correct.")
            st.caption(str(e))

st.markdown("""
<div class="card small">
<b>Important:</b> This application is a machine-learning decision-support prototype. "
"Predictions can be affected by image quality, lighting, background, crop variety and diseases not represented "
"in the training data. Confirm important agricultural decisions with a qualified professional.
</div>
""", unsafe_allow_html=True)
