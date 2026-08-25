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
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

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


/* =========================================================
   TITLE
   ========================================================= */

h1 {
    font-family: 'Manrope', sans-serif !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    text-align: center;
    font-size: 2.4rem !important;
    margin-bottom: 8px !important;
}


/* =========================================================
   SUBTITLE
   ========================================================= */

[data-testid="stCaptionContainer"] {
    color: #AFC8BE !important;
    text-align: center;
}


/* =========================================================
   SECTION HEADINGS
   ========================================================= */

h2 {
    font-family: 'Manrope', sans-serif !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

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


/* =========================================================
   IMAGE
   ========================================================= */

[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
}


/* =========================================================
   SUCCESS BOXES
   ========================================================= */

[data-testid="stAlert"] {
    border-radius: 12px;
}


/* =========================================================
   INFO BOXES
   ========================================================= */

[data-testid="stAlert"] p {
    color: #E8F0EC !important;
}


/* =========================================================
   GENERAL TEXT
   ========================================================= */

p, label {
    color: #D5E1DC;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 AgroDetect")

st.caption(
    "Upload a crop leaf image and receive disease diagnosis, "
    "treatment recommendations, and plant health improvement advice."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/crop_disease_model.keras"
    )


model = load_model()


# ============================================================
# DISEASE LABELS
# ============================================================

disease_labels = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


# ============================================================
# DISEASE INFORMATION
# ============================================================

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


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown("## 📤 Upload Leaf Image")

uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "png", "jpeg"]
)


# ============================================================
# IMAGE PROCESSING + PREDICTION
# ============================================================

if uploaded:

    # --------------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------------

    st.markdown("## 🖼️ Leaf Preview")

    st.image(
        uploaded,
        use_container_width=True
    )


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with st.spinner("Analyzing leaf..."):

        result = model.predict(
            img,
            verbose=0
        )[0]


    # --------------------------------------------------------
    # GET PREDICTION
    # --------------------------------------------------------

    pred_idx = int(
        np.argmax(result)
    )

    predicted_class = disease_labels[pred_idx]

    pct = float(
        np.max(result)
    ) * 100


    # --------------------------------------------------------
    # HEALTH STATUS
    # --------------------------------------------------------

    is_healthy = (
        "healthy" in predicted_class.lower()
    )

    status = (
        "Healthy"
        if is_healthy
        else "Diseased"
    )


    # --------------------------------------------------------
    # DISPLAY NAME
    # --------------------------------------------------------

    display_name = (
        predicted_class
        .replace("___", " ")
        .replace("__", " ")
        .replace("_", " ")
    )


    # --------------------------------------------------------
    # DISEASE INFORMATION
    # --------------------------------------------------------

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


    # ========================================================
    # DIAGNOSIS
    # ========================================================

    st.markdown("## 🔍 Diagnosis")

    if is_healthy:
        st.success(
            f"🌿 **Diagnosis Result: {display_name}**"
        )
    else:
        st.warning(
            f"🦠 **Diagnosis Result: {display_name}**"
        )

    # Status
    st.write(
        f"**Status:** {status}"
    )

    # Confidence
    st.write(
        f"**Confidence:** {pct:.1f}%"
    )

    # Severity
    st.write(
        f"**Severity:** {info['severity']}"
    )

    # Confidence progress bar
    st.progress(
        min(pct / 100, 1.0)
    )


    # ========================================================
    # TREATMENT
    # ========================================================

    st.markdown("## 💊 Treatment Recommendations")

    for treatment in info["treatment"]:

        st.success(
            treatment
        )


    # ========================================================
    # PLANT HEALTH IMPROVEMENT
    # ========================================================

    st.markdown(
        "## 🌱 Plant Health Improvement Plan"
    )

    col1, col2 = st.columns(2)

    for i, tip in enumerate(
        info["improvements"]
    ):

        if i % 2 == 0:

            with col1:
                st.info(tip)

        else:

            with col2:
                st.info(tip)