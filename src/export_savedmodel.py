import tensorflow as tf

model = tf.keras.models.load_model("models/crop_disease_model.keras")
model.export("models/crop_disease_model_savedmodel")