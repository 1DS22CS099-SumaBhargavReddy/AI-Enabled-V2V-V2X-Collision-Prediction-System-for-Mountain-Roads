import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
#                   PATHS
# ============================================================

SEQ_DIR = "../ml_model/sequences"
MODEL_DIR = "../ml_model/trained_model"
PLOTS_DIR = "../ml_model/trained_model/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ============================================================
#               LOAD DATA
# ============================================================

print("\n📌 Loading dataset...")
X = np.load(os.path.join(SEQ_DIR, "X.npy"))
y = np.load(os.path.join(SEQ_DIR, "y.npy"))

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Collision % =", (np.sum(y) / len(y)) * 100)

# ============================================================
#               TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42
)

# ============================================================
#       CLASS WEIGHTS (BEST FOR LARGE IMBALANCE)
# ============================================================

from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weights = {i: weights[i] for i in range(len(classes))}

print("\n📌 Class Weights:", class_weights)

# ============================================================
#       OPTIONAL: FOCAL LOSS FOR IMBALANCED CLASS
# ============================================================

def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = alpha * tf.pow((1 - p_t), gamma) * bce
        return fl
    return loss

USE_FOCAL = False  # make True if needed for extreme imbalance

loss_fn = focal_loss() if USE_FOCAL else "binary_crossentropy"

# ============================================================
#             IMPROVED LSTM MODEL
# ============================================================

model = Sequential([
    LSTM(128, return_sequences=True, recurrent_dropout=0.2,
         input_shape=(X.shape[1], X.shape[2])),
    LayerNormalization(),
    Dropout(0.3),

    LSTM(64, return_sequences=False, recurrent_dropout=0.2),
    LayerNormalization(),

    Dense(64, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007),
    loss=loss_fn,
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

print("\n📌 Model Summary:")
print(model.summary())

# ============================================================
#                CALLBACKS
# ============================================================

callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"), save_best_only=True)
]

# ============================================================
#                   TRAINING
# ============================================================

print("\n🚀 Training Model...")

history = model.fit(
    X_train, y_train,
    epochs=80,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

model.save(os.path.join(MODEL_DIR, "final_model.h5"))

print("\n🎉 Training complete!")

# ============================================================
#              PREDICTIONS
# ============================================================

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

# ============================================================
#               THRESHOLD TUNING
# ============================================================

def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.02)
    best_f1, best_t = -1, 0.5
    for t in thresholds:
        pred = (y_prob > t).astype(int)
        f1 = f1_score(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

best_t, best_f1 = find_best_threshold(y_test, y_prob)
print(f"\n⭐ Best Threshold = {best_t:.3f} | F1 = {best_f1:.4f}")

# ============================================================
#               METRICS
# ============================================================

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n📌 Metrics:")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
f2 = f1_score(y_test, y_pred, beta=2)
auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("F2 Score:", f2)
print("ROC AUC:", auc)
print("PR AUC:", pr_auc)
print("Balanced Accuracy:", balanced_acc)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.close()

# ============================================================
# TRAINING CURVES
# ============================================================

# Loss
plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"))
plt.close()

# Accuracy
plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_curve.png"))
plt.close()

print("\n📌 All plots saved in:", PLOTS_DIR)
print("📌 Best model saved in:", MODEL_DIR)
print("\n🎯 Training & Evaluation Completed Successfully!")
