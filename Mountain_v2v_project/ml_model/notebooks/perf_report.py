import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import load_model
import pickle

# ==================== PATHS ====================
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))  # go one level up from notebooks
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_FILE = os.path.join(MODEL_DIR, "lstm_v2v.h5")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURES_FILE = os.path.join(DATA_DIR, "features.csv")
LABELS_FILE = os.path.join(DATA_DIR, "labels.csv")
HISTORY_FILE = os.path.join(MODEL_DIR, "history.pkl")

# ==================== LOAD DATA ====================
print("[INFO] Loading dataset...")
features = pd.read_csv(FEATURES_FILE).values
labels = pd.read_csv(LABELS_FILE).values

# Load scaler and scale features
scaler = joblib.load(SCALER_FILE)
features_scaled = scaler.transform(features)

# Reshape for LSTM: (samples, timesteps=1, features)
X = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))
y = labels

# ==================== LOAD MODEL ====================
print("[INFO] Loading trained model...")
model = load_model(MODEL_FILE)

# ==================== PREDICTIONS ====================
y_pred_prob = model.predict(X)
y_pred = (y_pred_prob > 0.5).astype(int)

# ==================== CLASSIFICATION REPORT ====================
print("\n[INFO] Classification Report:")
print(classification_report(y, y_pred, digits=4))

# ==================== CONFUSION MATRIX ====================
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==================== ROC CURVE ====================
fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.title('Receiver Operating Characteristic (ROC)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# ==================== TRAINING HISTORY ====================
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'rb') as f:
        history = pickle.load(f)

    # Accuracy plot
    plt.figure(figsize=(6,5))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Loss plot
    plt.figure(figsize=(6,5))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
else:
    print("[INFO] Training history not found. Skipping training curves.")
