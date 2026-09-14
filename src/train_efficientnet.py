import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models

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

# Base model pretrained on ImageNet, frozen.
# EfficientNet has rescaling BUILT IN — expects raw 0-255 pixel values,
# so no external Rescaling layer here (unlike MobileNetV2).
base_model = EfficientNetB0(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),  # normalizes EfficientNet's pooled feature scale before the head — without this, the head's gradients are too small to learn anything
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save("models/crop_disease_model_efficientnet.keras")
print("Saved to models/crop_disease_model_efficientnet.keras")