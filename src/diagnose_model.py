import os
import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = "models/crop_disease_model.keras"  # change this to test the other two models
SAMPLE_FOLDER = "data/processed/train/Tomato_healthy"  # a class the model was trained on

model = tf.keras.models.load_model(MODEL_PATH)

# Get the class order the SAME way evaluate.py does
val_ds = tf.keras.utils.image_dataset_from_directory("data/processed/val", image_size=(224, 224))
class_names = val_ds.class_names
print("Class order:", class_names)

# Grab one image from a known training class
sample_file = os.listdir(SAMPLE_FOLDER)[0]
sample_path = os.path.join(SAMPLE_FOLDER, sample_file)
print(f"\nTesting on: {sample_path}")

img = cv2.imread(sample_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)).astype("float32")
img = np.expand_dims(img, axis=0)

pred = model.predict(img, verbose=0)[0]
top_idx = int(np.argmax(pred))

print(f"\nPredicted class: {class_names[top_idx]}")
print(f"Confidence: {pred[top_idx] * 100:.2f}%")
print(f"Expected class: Tomato_healthy")
print(f"\nFull prediction vector: {pred}")