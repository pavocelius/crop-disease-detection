import tensorflow as tf
import numpy as np
import cv2
import json

# ==============================
# Load model
# ==============================

model = tf.keras.models.load_model(
    "models/crop_disease_model.keras"
)

# ==============================
# Load class names
# ==============================

with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

print("\n===== CLASS NAMES =====")
for i, name in enumerate(class_names):
    print(i, ":", name)

# ==============================
# PUT YOUR HEALTHY IMAGE HERE
# ==============================

IMAGE_PATH = "PUT_YOUR_HEALTHY_IMAGE_PATH_HERE"

# Example:
# IMAGE_PATH = "data/processed/test/Tomato_healthy/image1.jpg"

# ==============================
# Read image
# ==============================

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("\nERROR: Image could not be loaded.")
    exit()

print("\nOriginal image shape:", img.shape)

# ==============================
# Convert to RGB
# ==============================

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ==============================
# Resize
# ==============================

img = cv2.resize(img, (224, 224)).astype("float32")

# ==============================
# Add batch dimension
# ==============================

img = np.expand_dims(img, axis=0)

# ==============================
# Prediction
# ==============================

prediction = model.predict(img, verbose=0)[0]

# ==============================
# Display ALL predictions
# ==============================

print("\n===== PREDICTIONS =====")

for i, probability in enumerate(prediction):
    print(
        f"{i}: {class_names[i]:50s} "
        f"{probability * 100:.6f}%"
    )

# ==============================
# Top prediction
# ==============================

top_index = np.argmax(prediction)

print("\n===== FINAL RESULT =====")
print("Predicted class :", class_names[top_index])
print("Class index     :", top_index)
print("Confidence      :", prediction[top_index] * 100, "%")