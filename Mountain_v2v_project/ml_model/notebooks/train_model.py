import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import joblib
import pickle

# =====================================================
# 🔹 GPU / CPU DETECTION
# =====================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] GPU available: {gpus}")
    except RuntimeError as e:
        print(f"[WARNING] GPU setup failed: {e}")
else:
    print("[INFO] No GPU detected, using CPU instead.")

# Optional: Mixed precision
try:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("[INFO] Mixed precision enabled (float16 computations).")
except Exception as e:
    print(f"[INFO] Mixed precision not enabled: {e}")

# =====================================================
# 📂 PATH SETUP
# =====================================================
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_DIR, "ml_model", "dataset")
FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")

MODEL_DIR = os.path.join(PROJECT_DIR, "ml_model", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_v2v.h5")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
HISTORY_FILE = os.path.join(MODEL_DIR, "history.pkl")

# =====================================================
# 📊 LOAD DATA
# =====================================================
features = pd.read_csv(FEATURES_FILE).values
labels = pd.read_csv(LABELS_FILE).values.flatten()  # flatten to 1D array

print(f"[INFO] Loaded {features.shape[0]} samples with {features.shape[1]} features.")

# =====================================================
# ⚙️ PREPROCESSING
# =====================================================
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
joblib.dump(scaler, SCALER_FILE)
print(f"[INFO] Scaler saved to {SCALER_FILE}")

# Reshape for LSTM (samples, timesteps, features)
X = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
y = labels

# =====================================================
# 🔹 Compute class weights to handle imbalance
# =====================================================
weights = class_weight.compute_class_weight(class_weight='balanced',
                                            classes=np.unique(y),
                                            y=y)
class_weights = {i: w for i, w in enumerate(weights)}
print(f"[INFO] Computed class weights: {class_weights}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# =====================================================
# 🧠 BUILD MODEL
# =====================================================
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid', dtype='float32')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# =====================================================
# 🏋️ TRAIN MODEL
# =====================================================
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    class_weight=class_weights,
    verbose=1
)

# Save training history
with open(HISTORY_FILE, 'wb') as f:
    pickle.dump(history.history, f)
print(f"[INFO] Training history saved to {HISTORY_FILE}")

# =====================================================
# 💾 SAVE MODEL
# =====================================================
model.save(MODEL_FILE)
print(f"[INFO] LSTM model saved to {MODEL_FILE}")

# =====================================================
# 📈 EVALUATE
# =====================================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Test Accuracy: {acc:.4f}, Test Loss: {loss:.4f}")

# =====================================================
# 🔹 ADDITIONAL METRICS
# =====================================================
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("[INFO] Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Training curves
plt.figure(figsize=(6,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(6,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
