import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.title("Crop Disease Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/crop_disease_model.keras")

model = load_model()
class_names = [...]  # paste your real 15-class list here, same order as training

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])

if uploaded:
    st.image(uploaded)
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype("float32")
    img = np.expand_dims(img, axis=0)

    result = model.predict(img)[0]
    pred_idx = int(np.argmax(result))
    st.write(f"**Prediction:** {class_names[pred_idx]}")
    st.write(f"**Confidence:** {float(np.max(result)):.2%}")