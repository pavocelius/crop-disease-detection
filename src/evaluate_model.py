import sys
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Usage: python src/evaluate_model.py models/crop_disease_model_cnn.keras
if len(sys.argv) < 2:
    print("Usage: python src/evaluate_model.py <path_to_model.keras>")
    sys.exit(1)

model_path = sys.argv[1]
print(f"Evaluating: {model_path}")

model = tf.keras.models.load_model(model_path)
val_ds = tf.keras.utils.image_dataset_from_directory("data/processed/val", image_size=(224, 224))
class_names = val_ds.class_names

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))
print(confusion_matrix(y_true, y_pred))