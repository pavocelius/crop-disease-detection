import tensorflow as tf
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

# A simple CNN built from scratch — no pre-trained weights, unlike MobileNetV2/EfficientNet.
# This gives you a genuine baseline to compare transfer learning against.
model = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save("models/crop_disease_model_cnn.keras")
print("Saved to models/crop_disease_model_cnn.keras")