from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("models/crop_disease_model.keras")
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).astype("float32")
    img = np.expand_dims(img, axis=0)  # model expects a batch dimension

    result = model.predict(img)[0]
    pred_idx = int(np.argmax(result))

    return {
        "disease": class_names[pred_idx],
        "confidence": float(np.max(result))
    }