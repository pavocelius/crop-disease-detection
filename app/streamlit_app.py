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
    background: #F7FAF8;
}

h1 {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    color: #0F2922;
    text-align: center;
    font-size: 2.3rem !important;
    margin-bottom: 6px !important;
}

.subtitle {
    text-align: center;
    color: #64748B;
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

st.markdown(
    "<h1>🌱 AgroDetect</h1>",
    unsafe_allow_html=True
)

st.markdown(
    '''
    <p class="subtitle">
    Upload a crop leaf image and receive disease diagnosis,
    treatment recommendations, and plant health improvement advice.
    </p>
    ''',
    unsafe_allow_html=True
)

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

uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "png", "jpeg"]
)

if uploaded:

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

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-tag">
            Diagnosis Result
        </div>

        <div class="result-label {css_class}">
            {display_name}
        </div>

        <div style="
            font-weight:600;
            margin-bottom:8px;
            color:#6B7A73;">
            Status: {status}
        </div>

        <div class="result-confidence">
            Confidence: {pct:.1f}%
        </div>

        <div style="
            margin-top:8px;
            font-weight:600;
            color:#6B7A73;">
            Severity: {info['severity']}
        </div>

        <div class="conf-track">
            <div
                class="conf-fill {css_class}"
                style="width:{pct}%;">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 💊 Treatment Recommendations")

    for treatment in info["treatment"]:
        st.success(treatment)

    st.markdown("## 🌱 Plant Health Improvement Plan")

    col1, col2 = st.columns(2)

    for i, tip in enumerate(info["improvements"]):

        with col1 if i % 2 == 0 else col2:
            st.info(tip)