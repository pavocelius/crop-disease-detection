import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = tf.keras.models.load_model("models/crop_disease_model.keras")
test_ds = tf.keras.utils.image_dataset_from_directory("data/processed/val", image_size=(224,224))

y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))