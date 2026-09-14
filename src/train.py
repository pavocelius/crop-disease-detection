import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/processed/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/processed/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE
)
class_names = train_ds.class_names
print("Classes:", class_names)

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

model = models.Sequential([
    layers.Rescaling(1./255),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# ============================================================
# DIAGNOSTIC: test prediction IMMEDIATELY, before saving/reloading,
# to isolate whether a save/reload bug is corrupting the model.
# ============================================================
print("\n" + "="*60)
print("IN-PROCESS SANITY CHECK (before saving)")
print("="*60)

sample_folder = "data/processed/train/Tomato_healthy"
sample_file = os.listdir(sample_folder)[0]
sample_path = os.path.join(sample_folder, sample_file)

img = cv2.imread(sample_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)).astype("float32")
img = np.expand_dims(img, axis=0)

pred = model.predict(img, verbose=0)[0]
top_idx = int(np.argmax(pred))
print(f"Predicted (in-process, no save/reload): {class_names[top_idx]} ({pred[top_idx]*100:.2f}%)")
print(f"Expected: Tomato_healthy")
print("="*60 + "\n")

model.save("models/crop_disease_model.keras")
print("Saved to models/crop_disease_model.keras")

# ============================================================
# DIAGNOSTIC PART 2: reload immediately and test again, in the
# SAME script run, to isolate the save/reload step specifically.
# ============================================================
print("\n" + "="*60)
print("POST-RELOAD SANITY CHECK (fresh load_model call)")
print("="*60)
reloaded = tf.keras.models.load_model("models/crop_disease_model.keras")
pred2 = reloaded.predict(img, verbose=0)[0]
top_idx2 = int(np.argmax(pred2))
print(f"Predicted (after save + reload): {class_names[top_idx2]} ({pred2[top_idx2]*100:.2f}%)")
print(f"Expected: Tomato_healthy")
print("="*60 + "\n")