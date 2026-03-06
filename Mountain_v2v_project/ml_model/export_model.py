import tensorflow as tf
import numpy as np
import joblib
import os

# ----------------- FIX PATHS -----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "ml_model", "trained_model")
SEQ_DIR = os.path.join(BASE_DIR, "ml_model", "sequences")

BEST_MODEL = os.path.join(MODEL_DIR, "best_model.h5")
FINAL_MODEL = os.path.join(MODEL_DIR, "final_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# ----------------- PICK BEST MODEL -----------------
if os.path.exists(BEST_MODEL):
    MODEL_PATH = BEST_MODEL
    print("✔ Using best_model.h5")
elif os.path.exists(FINAL_MODEL):
    MODEL_PATH = FINAL_MODEL
    print("✔ Using final_model.h5")
else:
    raise FileNotFoundError("❌ No model file found!")

print("MODEL:", MODEL_PATH)

# ----------------- LOAD MODEL -----------------
model = tf.keras.models.load_model(MODEL_PATH)

# ----------------- LOAD SCALER (if needed later) -----------------
scaler = joblib.load(SCALER_PATH)

# ----------------- GET INPUT SHAPE -----------------
X = np.load(os.path.join(SEQ_DIR, "X.npy"))
input_shape = X.shape[1:]
print("Input shape:", input_shape)

# ----------------- TFLITE CONVERTER FIX FOR LSTM -----------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Allow TF Select Ops (needed for LSTM TensorList)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Disable lowering of TensorList ops (CRITICAL)
converter._experimental_lower_tensor_list_ops = False

# Optional optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# ----------------- Convert -----------------
tflite_model = converter.convert()

# ----------------- Save -----------------
OUT_PATH = os.path.join(MODEL_DIR, "collision_model.tflite")
with open(OUT_PATH, "wb") as f:
    f.write(tflite_model)

print("=======================================")
print("✔ TFLite Conversion Successful!")
print("Saved at:", OUT_PATH)
print("=======================================")
