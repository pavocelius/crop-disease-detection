import cv2
import numpy as np

def preprocess_image(path, size=(224, 224)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)  # denoise
    img = img.astype("float32") / 255.0  # normalize to 0-1
    return img