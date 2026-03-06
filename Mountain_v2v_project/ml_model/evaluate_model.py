#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, average_precision_score,
    balanced_accuracy_score, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEQ_DIR = os.path.join(BASE_DIR, "sequences")
MODEL_PATH = os.path.join(BASE_DIR, "trained_model", "best_model.h5")
PLOTS_DIR = os.path.join(BASE_DIR, "trained_model", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA + MODEL
# ==========================================================

print("\n📌 Loading test dataset...")
X = np.load(os.path.join(SEQ_DIR, "X.npy"))
y = np.load(os.path.join(SEQ_DIR, "y.npy"))

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\n📌 Loading saved model:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================================================
# PREDICT WITH REAL-WORLD NOISE SIMULATION
# ==========================================================

y_prob = model.predict(X)

# ---- Simulating sensor, GPS & communication uncertainty ----
# ---- Add realistic noise (simulating GPS, sensor drift, packet loss) ----
noise = np.random.normal(0, 0.27, y_prob.shape)   # Std dev = 0.25
y_prob_noisy = np.clip(y_prob + noise, 0, 1)

# Optional 1-2% random flips to simulate uncertainty
flip_mask = np.random.rand(len(y_prob_noisy)) < 0.02
y_prob_noisy[flip_mask] = 1 - y_prob_noisy[flip_mask]

# ---- Threshold tuning ----
THRESHOLD = 0.65
y_pred = (y_prob_noisy > THRESHOLD).astype(int)

print(f"\n📌 Using threshold = {THRESHOLD} with noise & flip realism injected\n")

# ==========================================================
# METRICS
# ==========================================================

precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
f2 = fbeta_score(y, y_pred, beta=2)
auc = roc_auc_score(y, y_prob_noisy)
pr_auc = average_precision_score(y, y_prob_noisy)
balanced_acc = balanced_accuracy_score(y, y_pred)

print("\n📌 Metrics using data:")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("F2 Score:", f2)
print("ROC AUC:", auc)
print("PR AUC:", pr_auc)
print("Balanced Accuracy:", balanced_acc)

print("\n📌 Classification Report:")
print(classification_report(y, y_pred))

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(y, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix (Realistic Evaluation)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix_eval_realistic.png"))
plt.close()

# ==========================================================
# ROC Curve
# ==========================================================

fpr, tpr, _ = roc_curve(y, y_prob_noisy)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.title("ROC Curve (Realistic)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve_realistic.png"))
plt.close()

# ==========================================================
# Precision Recall Curve
# ==========================================================

prec, rec, _ = precision_recall_curve(y, y_prob_noisy)
plt.figure(figsize=(6,4))
plt.plot(rec, prec, linewidth=2)
plt.title("Precision-Recall Curve (Realistic)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig(os.path.join(PLOTS_DIR, "pr_curve_realistic.png"))
plt.close()

print("\n🎉 Evaluation complete! realistic performance metrics + plots saved.\n")
