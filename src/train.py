import tensorflow as tf
import json
from keras.applications import MobileNetV2
from keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load data directly from folders (folder names = class labels)
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/processed/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/processed/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

print("\n===== CLASS ORDER USED FOR TRAINING =====")
for i, cls in enumerate(class_names):
    print(f"{i}: {cls}")
print("=========================================\n")

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)

# Base model (pretrained on ImageNet, frozen)
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

model.save("models/crop_disease_model.keras")