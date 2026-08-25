import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.set_page_config(page_title="Crop Disease Detection", page_icon="🔬", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark Streamlit background ── */
.stApp {
    background: #111816;
    color: #E8F0EC;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ── Main title ── */
h1 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    text-align: center;
    font-size: 2.4rem !important;
    margin-bottom: 8px !important;
}

/* Subtitle */
.stCaption {
    color: #AFC8BE !important;
    text-align: center;
}

/* ── Section headings ── */
h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #FFFFFF !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1A2420;
    border: 2px dashed #2E7D32;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}

[data-testid="stFileUploader"] label {
    color: #E8F0EC !important;
}

/* ── Image ── */
[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
}

/* ── Result card ── */
.result-card {
    background: #1A2420;
    border-radius: 18px;
    padding: 28px;
    margin-top: 20px;
    border: 1px solid #2B3934;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    text-align: center;
}

.result-card.healthy {
    border-top: 4px solid #34D399;
}

.result-card.diseased {
    border-top: 4px solid #F0973B;
}

/* Diagnosis Result */
.result-tag {
    display: block;
    font-family: 'Manrope', sans-serif;
    font-weight: 700;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #AFC8BE;
    margin-bottom: 10px;
}

/* Disease name */
.result-label {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    margin-bottom: 14px;
}

.result-label.healthy {
    color: #34D399;
}

.result-label.diseased {
    color: #F0973B;
}

/* Status */
.result-status {
    font-weight: 600;
    margin-bottom: 8px;
    color: #D5E1DC;
}

/* Confidence */
.result-confidence {
    color: #AFC8BE;
    font-size: 0.95rem;
}

/* Severity */
.result-severity {
    margin-top: 8px;
    font-weight: 600;
    color: #D5E1DC;
}

/* Confidence bar */
.conf-track {
    background: #2A3531;
    border-radius: 999px;
    height: 9px;
    margin-top: 14px;
    overflow: hidden;
}

.conf-fill {
    height: 100%;
    border-radius: 999px;
}

.conf-fill.healthy {
    background: #34D399;
}

.conf-fill.diseased {
    background: #F0973B;
}

/* ── Success / treatment boxes ── */
.stAlert {
    border-radius: 12px;
}

/* Text inside dark UI */
p, label {
    color: #D5E1DC;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 AgroDetect")

st.caption(
    "Upload a crop leaf image and receive disease diagnosis, "
    "treatment recommendations, and plant health improvement advice."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/crop_disease_model.keras")

model = load_model()

# Full class list needed internally to map prediction index to a label
disease_labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "severity": "Moderate",
        "treatment": [
            "Remove infected leaves",
            "Avoid overhead watering",
            "Apply copper-based bactericide"
        ],
        "improvements": [
            "Improve air circulation",
            "Avoid wet foliage",
            "Monitor plants weekly"
        ]
    },

    "Pepper__bell___healthy": {
        "severity": "None",
        "treatment": [
            "No treatment required"
        ],
        "improvements": [
            "Maintain watering schedule",
            "Continue regular monitoring",
            "Apply balanced fertilizer"
        ]
    },

    "Potato___Early_blight": {
        "severity": "Moderate",
        "treatment": [
            "Remove infected leaves",
            "Apply fungicide",
            "Reduce leaf wetness"
        ],
        "improvements": [
            "Increase plant spacing",
            "Improve drainage",
            "Add organic compost"
        ]
    },

    "Potato___Late_blight": {
        "severity": "High",
        "treatment": [
            "Remove infected plants",
            "Apply fungicide immediately",
            "Inspect nearby plants"
        ],
        "improvements": [
            "Improve airflow",
            "Avoid excess moisture",
            "Practice crop rotation"
        ]
    },

    "Potato___healthy": {
        "severity": "None",
        "treatment": [
            "No treatment required"
        ],
        "improvements": [
            "Maintain healthy watering schedule",
            "Apply fertilizer as needed",
            "Continue monitoring"
        ]
    },

    "Tomato_Bacterial_spot": {
        "severity": "Moderate",
        "treatment": [
            "Remove infected leaves",
            "Apply copper spray",
            "Avoid working with wet plants"
        ],
        "improvements": [
            "Improve air circulation",
            "Reduce leaf moisture",
            "Inspect regularly"
        ]
    },

    "Tomato_Early_blight": {
        "severity": "Moderate",
        "treatment": [
            "Remove infected leaves",
            "Apply fungicide",
            "Avoid overhead watering"
        ],
        "improvements": [
            "Improve airflow",
            "Use mulch",
            "Monitor weekly"
        ]
    },

    "Tomato_Late_blight": {
        "severity": "High",
        "treatment": [
            "Remove infected plants immediately",
            "Apply fungicide",
            "Separate infected plants"
        ],
        "improvements": [
            "Improve drainage",
            "Reduce humidity",
            "Inspect nearby crops"
        ]
    },

    "Tomato_Leaf_Mold": {
        "severity": "Moderate",
        "treatment": [
            "Remove affected leaves",
            "Improve ventilation",
            "Use fungicide if necessary"
        ],
        "improvements": [
            "Reduce humidity",
            "Increase spacing",
            "Prune dense foliage"
        ]
    },

    "Tomato_Septoria_leaf_spot": {
        "severity": "Moderate",
        "treatment": [
            "Remove infected leaves",
            "Apply fungicide",
            "Avoid splashing water"
        ],
        "improvements": [
            "Use mulch",
            "Improve airflow",
            "Rotate crops"
        ]
    },

    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "severity": "Moderate",
        "treatment": [
            "Spray leaves with water",
            "Apply insecticidal soap",
            "Remove heavily affected leaves"
        ],
        "improvements": [
            "Increase humidity",
            "Inspect undersides of leaves",
            "Monitor regularly"
        ]
    },

    "Tomato__Target_Spot": {
        "severity": "Moderate",
        "treatment": [
            "Remove infected foliage",
            "Apply fungicide",
            "Avoid overcrowding"
        ],
        "improvements": [
            "Increase spacing",
            "Improve drainage",
            "Monitor symptoms"
        ]
    },

    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "severity": "High",
        "treatment": [
            "Remove infected plants",
            "Control whiteflies",
            "Prevent spread immediately"
        ],
        "improvements": [
            "Use resistant varieties",
            "Monitor pests",
            "Keep field clean"
        ]
    },

    "Tomato__Tomato_mosaic_virus": {
        "severity": "High",
        "treatment": [
            "Remove infected plants",
            "Disinfect tools",
            "Avoid plant-to-plant contact"
        ],
        "improvements": [
            "Use certified seeds",
            "Control contamination",
            "Inspect plants frequently"
        ]
    },

    "Tomato_healthy": {
        "severity": "None",
        "treatment": [
            "No treatment required"
        ],
        "improvements": [
            "Maintain balanced nutrition",
            "Continue monitoring",
            "Keep regular watering schedule"
        ]
    }
}

st.markdown("## 📤 Upload Leaf Image")
uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:

    st.markdown("## 🖼️ Leaf Preview")
    st.image(uploaded, use_container_width=True)

    file_bytes = np.asarray(
        bytearray(uploaded.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (224, 224)
    ).astype("float32")

    img = np.expand_dims(
        img,
        axis=0
    )

    with st.spinner("Analyzing leaf..."):
        result = model.predict(img, verbose=0)[0]

    pred_idx = int(np.argmax(result))

    predicted_class = disease_labels[pred_idx]

    pct = float(np.max(result)) * 100

    is_healthy = "healthy" in predicted_class.lower()

    status = (
        "Healthy"
        if is_healthy
        else "Diseased"
    )

    css_class = (
        "healthy"
        if is_healthy
        else "diseased"
    )

    display_name = (
        predicted_class
        .replace("___", " ")
        .replace("__", " ")
        .replace("_", " ")
    )

    info = disease_info.get(
        predicted_class,
        {
            "severity": "Unknown",
            "treatment": [
                "Consult agricultural specialist"
            ],
            "improvements": [
                "Monitor plant condition"
            ]
        }
    )

    st.markdown("## 🔍 Diagnosis")

    # Diagnosis result card
    if is_healthy:
        border_color = "#34D399"
        label_color = "#34D399"
    else:
        border_color = "#F0973B"
        label_color = "#F0973B"

    st.markdown(
        f"""
        <style>
        .diagnosis-box {{
            background: #1A2420;
            border: 1px solid #2B3934;
            border-top: 4px solid {border_color};
            border-radius: 18px;
            padding: 28px;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 25px;
        }}

        .diagnosis-title {{
            color: #AFC8BE;
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}

        .diagnosis-name {{
            color: {label_color};
            font-size: 26px;
            font-weight: 800;
            margin-bottom: 14px;
        }}

        .diagnosis-info {{
            color: #D5E1DC;
            font-size: 15px;
            margin: 7px 0;
        }}

        .confidence-bar {{
            background: #2A3531;
            height: 9px;
            border-radius: 10px;
            margin-top: 15px;
            overflow: hidden;
        }}

        .confidence-fill {{
            background: {label_color};
            height: 100%;
            width: {min(pct, 100):.2f}%;
            border-radius: 10px;
        }}
        </style>

        <div class="diagnosis-box">

            <div class="diagnosis-title">
                Diagnosis Result
            </div>

            <div class="diagnosis-name">
                {display_name}
            </div>

            <div class="diagnosis-info">
                <b>Status:</b> {status}
            </div>

            <div class="diagnosis-info">
                <b>Confidence:</b> {pct:.1f}%
            </div>

            <div class="diagnosis-info">
                <b>Severity:</b> {info['severity']}
            </div>

            <div class="confidence-bar">
                <div class="confidence-fill"></div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## 💊 Treatment Recommendations")

    for treatment in info["treatment"]:
        st.success(treatment)

    st.markdown("## 🌱 Plant Health Improvement Plan")

    col1, col2 = st.columns(2)

    for i, tip in enumerate(info["improvements"]):
        if i % 2 == 0:
            with col1:
                st.info(tip)
        else:
            with col2:
                st.info(tip)