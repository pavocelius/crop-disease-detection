from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import os
import tensorflow as tf
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../FINAL YEAR PROJECT/app
PROJECT_ROOT = os.path.dirname(BASE_DIR)                        # .../FINAL YEAR PROJECT
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crop_disease_model.keras")

model = tf.keras.models.load_model(MODEL_PATH)

import json

with open(
    os.path.join(PROJECT_ROOT, "models", "class_names.json"),
    "r"
) as f:
    disease_labels = json.load(f)

print("Loaded disease labels:")
for i, label in enumerate(disease_labels):
    print(f"{i}: {label}")

MIN_CONFIDENCE = 70.0   # top prediction must be at least this confident
MIN_MARGIN = 10.0       # top prediction must beat the runner-up by at least this much

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "Error", "message": "Unable to read this image. Please upload a valid JPG or PNG."}

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype("float32")
    img = np.expand_dims(img, axis=0)

    result = model.predict(img, verbose=0)[0]

    sorted_indices = np.argsort(result)[::-1]
    top_index = int(sorted_indices[0])
    second_index = int(sorted_indices[1])

    top_confidence = float(result[top_index]) * 100
    second_confidence = float(result[second_index]) * 100
    margin = top_confidence - second_confidence

    confidence_ok = top_confidence >= MIN_CONFIDENCE
    margin_ok = margin >= MIN_MARGIN

    if not (confidence_ok and margin_ok):
        return {
            "status": "Unknown / Not Supported",
            "confidence": round(top_confidence, 1),
            "margin": round(margin, 1),
            "message": "The model wasn't confident enough to give a reliable diagnosis for this image."
        }

    predicted_class = disease_labels[top_index]
    status = "Healthy" if "healthy" in predicted_class.lower() else "Diseased"

    return {
        "status": status,
        "disease": predicted_class,
        "confidence": round(top_confidence, 1)
    }