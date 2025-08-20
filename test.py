import tensorflow as tf
import numpy as np
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

SIZE = 256
MODEL_PATH = os.getenv("MODEL_PATH")

def load_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

def preprocess_image(file_bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    x = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (SIZE, SIZE))
    x = x / 255.0
    x = x.astype(np.float32)
    return np.expand_dims(x, axis=0)

def postprocess_classification(cls_pred):
    x = int(cls_pred[0] > 0.5)
    if x == 1:
        return "Prêt pour l'incubation"
    return "N'est pas prêt pour l'incubation"