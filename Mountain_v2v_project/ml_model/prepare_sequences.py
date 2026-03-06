import pandas as pd
import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib

SEQUENCE_LENGTH = 10  # 10 timesteps window
DATA_DIR = "../ml_model/dataset"
SAVE_DIR = "../ml_model/sequences"

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------- Load dataset -----------------
features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
labels = pd.read_csv(os.path.join(DATA_DIR, "labels.csv"))

# Merge
data = pd.concat([features, labels], axis=1)

# ----------------- Normalize features -----------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

joblib.dump(scaler, "../ml_model/trained_model/scaler.pkl")

# ----------------- Build sequences -----------------
X, y = [], []
for i in range(len(data) - SEQUENCE_LENGTH):
    seq = scaled_features[i:i + SEQUENCE_LENGTH]
    label = labels.iloc[i + SEQUENCE_LENGTH]["collision_label"]
    X.append(seq)
    y.append(label)

X = np.array(X)
y = np.array(y)

print("Before balancing:")
print("Collision:", np.sum(y == 1), " | Non-Collision:", np.sum(y == 0))

# ----------------- Balance dataset -----------------
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]

# Oversample collision class
oversampled_pos = np.random.choice(pos_idx, size=len(neg_idx), replace=True)

balanced_idx = np.concatenate([neg_idx, oversampled_pos])
balanced_idx = shuffle(balanced_idx)

X_bal = X[balanced_idx]
y_bal = y[balanced_idx]

print("\nAfter balancing:")
print("Collision:", np.sum(y_bal == 1), " | Non-Collision:", np.sum(y_bal == 0))

# ----------------- Save sequences -----------------
np.save(os.path.join(SAVE_DIR, "X.npy"), X_bal)
np.save(os.path.join(SAVE_DIR, "y.npy"), y_bal)

print("\nSequences saved successfully!")
print("X shape:", X_bal.shape)
print("y shape:", y_bal.shape)
