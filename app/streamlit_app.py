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
   ALERTS
   ========================================================= */

[data-testid="stAlert"] {
    border-radius: 12px;
}


/* =========================================================
   GENERAL TEXT
   ========================================================= */

p, label {
    color: #D5E1DC;
}


/* =========================================================
   UNKNOWN RESULT
   ========================================================= */

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


/* =========================================================
   DIAGNOSIS RESULT
   ========================================================= */

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

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 AgroDetect")

st.caption(
    "Upload a crop leaf image and receive disease detection, "
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
# UPLOAD IMAGE
# ============================================================

st.markdown("## 📤 Upload Leaf Image")

uploaded = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "png", "jpeg"]
)


# ============================================================
# IMAGE PROCESSING
# ============================================================

if uploaded:

    st.markdown("## 🖼️ Leaf Preview")

    st.image(
        uploaded,
        use_container_width=True
    )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    file_bytes = np.asarray(
        bytearray(uploaded.getvalue()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    if img is None:

        st.error(
            "Unable to read this image. Please upload a valid JPG or PNG file."
        )

        st.stop()


    # --------------------------------------------------------
    # BASIC IMAGE VALIDATION
    # --------------------------------------------------------

    height, width = img.shape[:2]

    # Very tiny images are unlikely to provide useful information
    if height < 100 or width < 100:

        st.warning(
            "⚠️ The image is too small. "
            "Please upload a clearer crop leaf image."
        )

        st.stop()


    # --------------------------------------------------------
    # CONVERT TO RGB
    # --------------------------------------------------------

    rgb_img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )


    # ========================================================
    # SIMPLE LEAF IMAGE CHECK
    # ========================================================

    # Convert image to HSV
    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    # Green color range
    lower_green = np.array(
        [25, 30, 30]
    )

    upper_green = np.array(
        [95, 255, 255]
    )

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    green_ratio = (
        np.count_nonzero(green_mask)
        /
        green_mask.size
    )


    # ========================================================
    # MODEL INPUT
    # ========================================================

    model_img = cv2.resize(
        rgb_img,
        (224, 224)
    ).astype("float32")

    model_img = np.expand_dims(
        model_img,
        axis=0
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    with st.spinner("Analyzing leaf..."):

        result = model.predict(
            model_img,
            verbose=0
        )[0]


    # --------------------------------------------------------
    # SORT PREDICTIONS
    # --------------------------------------------------------

    sorted_indices = np.argsort(
        result
    )[::-1]

    top_index = int(
        sorted_indices[0]
    )

    second_index = int(
        sorted_indices[1]
    )


    # --------------------------------------------------------
    # TOP PREDICTION
    # --------------------------------------------------------

    predicted_class = disease_labels[
        top_index
    ]

    top_confidence = float(
        result[top_index]
    ) * 100

    second_confidence = float(
        result[second_index]
    ) * 100

    prediction_margin = (
        top_confidence
        -
        second_confidence
    )


    # ========================================================
    # PROTECTION SETTINGS
    # ========================================================

    # Minimum confidence required
    MIN_CONFIDENCE = 70.0

    # Minimum difference between first and second prediction
    MIN_MARGIN = 10.0

    # Minimum amount of green in image
    MIN_GREEN_RATIO = 0.05


    # ========================================================
    # PROTECTION CHECKS
    # ========================================================

    confidence_ok = (
        top_confidence >= MIN_CONFIDENCE
    )

    margin_ok = (
        prediction_margin >= MIN_MARGIN
    )

    leaf_like = (
        green_ratio >= MIN_GREEN_RATIO
    )


    # ========================================================
    # UNKNOWN / NOT SUPPORTED DETECTION
    # ========================================================

    prediction_valid = (
        confidence_ok
        and
        margin_ok
        and
        leaf_like
    )


    # ========================================================
    # DISPLAY DIAGNOSIS
    # ========================================================

    st.markdown("## 🔍 Diagnosis")


    # --------------------------------------------------------
    # UNKNOWN RESULT
    # --------------------------------------------------------

    if not prediction_valid:

        st.markdown(
            f"""
            <div class="unknown-box">

                <div class="unknown-title">
                    ⚠️ Unknown / Not Supported
                </div>

                <div class="unknown-text">
                    The image could not be confidently identified
                    as one of the supported crop conditions.
                    <br><br>
                    Please upload a clear image of a
                    <b>tomato, potato, or pepper leaf</b>
                    from a supported condition.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # Show diagnostic information
        with st.expander("Why was this image rejected?"):

            st.write(
                f"Model confidence: {top_confidence:.1f}%"
            )

            st.write(
                f"Difference from second prediction: "
                f"{prediction_margin:.1f}%"
            )

            st.write(
                f"Green-area ratio: "
                f"{green_ratio * 100:.1f}%"
            )

            if not confidence_ok:

                st.write(
                    "❌ Model confidence was too low."
                )

            if not margin_ok:

                st.write(
                    "❌ The model was not sufficiently "
                    "certain between its top predictions."
                )

            if not leaf_like:

                st.write(
                    "❌ The image does not appear sufficiently "
                    "leaf-like based on its color characteristics."
                )


        st.info(
            "💡 Supported crops: Tomato, Potato, and Pepper."
        )


    # --------------------------------------------------------
    # VALID RESULT
    # --------------------------------------------------------

    else:

        # ----------------------------------------------------
        # DETERMINE HEALTH STATUS
        # ----------------------------------------------------

        is_healthy = (
            "healthy"
            in predicted_class.lower()
        )

        status = (
            "Healthy"
            if is_healthy
            else "Diseased"
        )


        # ----------------------------------------------------
        # DISPLAY NAME
        # ----------------------------------------------------

        display_name = (
            predicted_class
            .replace("___", " ")
            .replace("__", " ")
            .replace("_", " ")
        )


        # ----------------------------------------------------
        # DISEASE INFORMATION
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # RESULT COLORS
        # ----------------------------------------------------

        if is_healthy:

            result_color = "#34D399"

        else:

            result_color = "#F0973B"


        # ----------------------------------------------------
        # DIAGNOSIS CARD
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="diagnosis-box"
                 style="border-top: 4px solid {result_color};">

                <div class="diagnosis-title">
                    Diagnosis Result
                </div>

                <div class="diagnosis-name"
                     style="color: {result_color};">

                    {display_name}

                </div>

                <div class="diagnosis-info">
                    <b>Status:</b> {status}
                </div>

                <div class="diagnosis-info">
                    <b>Confidence:</b> {top_confidence:.1f}%
                </div>

                <div class="diagnosis-info">
                    <b>Severity:</b> {info['severity']}
                </div>

                <div class="confidence-bar">

                    <div class="confidence-fill"
                         style="
                            width: {min(top_confidence, 100):.2f}%;
                            background: {result_color};
                         ">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # TREATMENT
        # ====================================================

        st.markdown(
            "## 💊 Treatment Recommendations"
        )

        for treatment in info["treatment"]:

            st.success(
                treatment
            )


        # ====================================================
        # PLANT HEALTH IMPROVEMENT
        # ====================================================

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