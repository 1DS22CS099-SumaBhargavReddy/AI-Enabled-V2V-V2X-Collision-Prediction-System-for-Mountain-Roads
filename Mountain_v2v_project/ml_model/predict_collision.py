import numpy as np
import tensorflow as tf
import joblib

MODEL_PATH = "../ml_model/trained_model/final_model.h5"
SCALER_PATH = "../ml_model/trained_model/scaler.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

SEQUENCE_LENGTH = 10

buffer = []

def predict_collision(speed, accel, ttc):
    global buffer

    # Scale input
    inp = scaler.transform([[speed, accel, ttc]])[0]

    buffer.append(inp)
    if len(buffer) < SEQUENCE_LENGTH:
        return 0  # not enough data

    seq = np.array(buffer[-SEQUENCE_LENGTH:]).reshape(1, SEQUENCE_LENGTH, 3)
    prob = model.predict(seq)[0][0]

    return 1 if prob > 0.5 else 0
