#!/usr/bin/env python3
"""
analyze_and_prepare_sequences.py

FULL AUTO VERSION — NO ARGUMENTS NEEDED.

What this script does:

1) Load SUMO output: ../simulation/output/vehicle_data.csv
2) For each vehicle at each timestep, find its nearest neighbor (within DIST_THRESHOLD)
   using a spatial grid, and compute:
   - nearest distance
   - nearest relative speed
   - nearest TTC
   - AND also store the nearest vehicle's ID and basic features
3) Encode lane_id, edge_id, veh_type (for both ego and nearest vehicle)
4) Save:
   - features.csv   (one row per (ego, nearest) pair)
   - labels.csv     (collision label per row)
5) Create 10-step sequences per ego vehicle
6) Oversample collision class to balance data
7) Scale features (StandardScaler)
8) Save:
   - X.npy, y.npy
   - scaler.pkl
   - encoders
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle

# ==========================================================
#   FIXED RELATIVE PATHS (WORK ON ANY MACHINE)
# ==========================================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))   # ml_model folder

# vehicle_data.csv is inside: ../simulation/output/
DATA_FILE = os.path.abspath(os.path.join(BASE_DIR, "../simulation/output/vehicle_data.csv"))

# Dataset outputs inside: ml_model/dataset/
OUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "dataset"))

# Sequence outputs inside: ml_model/sequences/
SEQ_DIR = os.path.abspath(os.path.join(BASE_DIR, "sequences"))

# Save scaler in dataset folder
SCALER_PATH = os.path.join(OUT_DIR, "scaler.pkl")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SEQ_DIR, exist_ok=True)

# ---- Your chosen values ----
# Customization for Collision Prediction and analysis
GRID_SIZE = 30.0
DIST_THRESHOLD = 25.0       # only neighbors closer than this are considered
TTC_THRESHOLD = 2.0         # label = 1 if TTC <= this
WINDOW_SIZE = 10
STEP = 1
OVERSAMPLE = True
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ==========================================================
# Helper functions
# ==========================================================

def compute_heading(df):
    df = df.copy()
    df["prev_x"] = df.groupby("veh_id")["x"].shift(1).fillna(df["x"])
    df["prev_y"] = df.groupby("veh_id")["y"].shift(1).fillna(df["y"])
    dx = df["x"] - df["prev_x"]
    dy = df["y"] - df["prev_y"]
    df["heading"] = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    df = df.drop(columns=["prev_x", "prev_y"])
    return df

def nearest_neighbor_with_index(step_df):
    """
    Similar logic as your old nearest_neighbor_metrics, but now
    ALSO remembers which j index was the best neighbor for each i.
    """
    x = step_df["x"].values
    y = step_df["y"].values
    v = step_df["speed"].values
    N = len(step_df)

    nearest_dist = np.full(N, 1e6)
    nearest_rel_speed = np.zeros(N)
    nearest_ttc = np.full(N, 1e6)
    nearest_idx = np.full(N, -1, dtype=int)

    gx = (x // GRID_SIZE).astype(int)
    gy = (y // GRID_SIZE).astype(int)

    grid = defaultdict(list)
    for i in range(N):
        grid[(gx[i], gy[i])].append(i)

    EPS = 1e-6

    for i in range(N):
        ix, iy = gx[i], gy[i]
        for dxg in (-1, 0, 1):
            for dyg in (-1, 0, 1):
                cell = (ix + dxg, iy + dyg)
                for j in grid.get(cell, []):
                    if j == i:
                        continue

                    dxp = x[j] - x[i]
                    dyp = y[j] - y[i]
                    dist = math.hypot(dxp, dyp)

                    if dist > DIST_THRESHOLD:
                        continue

                    rel = abs(v[i] - v[j])
                    ttc = dist / rel if rel > EPS else 1e6

                    # pick closest (smallest TTC) neighbor
                    if ttc < nearest_ttc[i]:
                        nearest_ttc[i] = ttc
                        nearest_dist[i] = dist
                        nearest_rel_speed[i] = rel
                        nearest_idx[i] = j

    return nearest_idx, nearest_dist, nearest_rel_speed, nearest_ttc

# ==========================================================
# MAIN FUNCTION
# ==========================================================

def main():

    print("\n🚀 LOADING VEHICLE DATA:", DATA_FILE)
    df = pd.read_csv(DATA_FILE)
    df = df.sort_values(["time", "veh_id"]).reset_index(drop=True)

    # If you want heading later, this keeps it available (not mandatory in features)
    print("📌 Computing headings...")
    df = compute_heading(df)

    features_rows = []
    labels = []

    times = df["time"].unique()
    print(f"⏳ Total timesteps: {len(times)}")

    for t in times:
        step_df = df[df.time == t].reset_index(drop=True)
        if len(step_df) == 0:
            continue

        nearest_idx, nearest_dist, nearest_rel_speed, nearest_ttc = nearest_neighbor_with_index(step_df)

        for i, row in step_df.iterrows():
            j = nearest_idx[i]
            if j == -1:
                # No valid neighbor within DIST_THRESHOLD
                continue

            other = step_df.iloc[j]

            # collision label: same logic as before, based on TTC
            collision_label = 1 if nearest_ttc[i] <= TTC_THRESHOLD else 0

            # Store BOTH vehicles' data: ego (this row) and nearest neighbor (other)
            features_rows.append({
                # time + IDs
                "time": row["time"],
                "veh_id": row["veh_id"],                  # ego vehicle (A)
                "other_veh_id": other["veh_id"],          # nearest vehicle (B)

                # ego vehicle features (A)
                "x": row["x"],
                "y": row["y"],
                "speed": row["speed"],
                "accel": row["accel"],
                "angle": row["angle"],
                "curvature": row["curvature"],
                "lane_id": row["lane_id"],
                "edge_id": row["edge_id"],
                "veh_type": row["veh_type"],

                # nearest vehicle features (B)
                "x_other": other["x"],
                "y_other": other["y"],
                "speed_other": other["speed"],
                "accel_other": other["accel"],
                "angle_other": other["angle"],
                "curvature_other": other["curvature"],
                "lane_id_other": other["lane_id"],
                "edge_id_other": other["edge_id"],
                "veh_type_other": other["veh_type"],

                # pairwise metrics
                "nearest_dist": nearest_dist[i],
                "nearest_rel_speed": nearest_rel_speed[i],
                "nearest_ttc": nearest_ttc[i],

                # label
                "collision_label": collision_label,
            })

            labels.append(collision_label)

    features_df = pd.DataFrame(features_rows)
    labels_df = pd.DataFrame({"collision_label": labels})

    # ==========================================================
    # Encode categorical columns (for BOTH vehicles)
    # ==========================================================
    print("🔧 Encoding categorical columns...")

    features_df["lane_id"] = features_df["lane_id"].astype(str)
    features_df["edge_id"] = features_df["edge_id"].astype(str)
    features_df["veh_type"] = features_df["veh_type"].astype(str)

    features_df["lane_id_other"] = features_df["lane_id_other"].astype(str)
    features_df["edge_id_other"] = features_df["edge_id_other"].astype(str)
    features_df["veh_type_other"] = features_df["veh_type_other"].astype(str)

    le_lane = LabelEncoder()
    le_edge = LabelEncoder()
    le_veh = LabelEncoder()

    # Fit on combined values of ego + other, so no unseen category issues
    all_lanes = pd.concat([features_df["lane_id"], features_df["lane_id_other"]], ignore_index=True)
    all_edges = pd.concat([features_df["edge_id"], features_df["edge_id_other"]], ignore_index=True)
    all_veh   = pd.concat([features_df["veh_type"], features_df["veh_type_other"]], ignore_index=True)

    le_lane.fit(all_lanes)
    le_edge.fit(all_edges)
    le_veh.fit(all_veh)

    features_df["lane_enc"]        = le_lane.transform(features_df["lane_id"])
    features_df["lane_other_enc"]  = le_lane.transform(features_df["lane_id_other"])
    features_df["edge_enc"]        = le_edge.transform(features_df["edge_id"])
    features_df["edge_other_enc"]  = le_edge.transform(features_df["edge_id_other"])
    features_df["veh_enc"]         = le_veh.transform(features_df["veh_type"])
    features_df["veh_other_enc"]   = le_veh.transform(features_df["veh_type_other"])

    joblib.dump(le_lane, os.path.join(OUT_DIR, "le_lane.pkl"))
    joblib.dump(le_edge, os.path.join(OUT_DIR, "le_edge.pkl"))
    joblib.dump(le_veh, os.path.join(OUT_DIR, "le_veh.pkl"))

    # ==========================================================
    # Final selected columns (features)
    # ==========================================================
    # Keep old columns PLUS "other_*" so you know which vehicle pair raised the event
    final_cols = [
        "time",
        "veh_id", "other_veh_id",

        # ego (A)
        "x", "y", "speed", "accel", "angle", "curvature",

        # other (B)
        "x_other", "y_other", "speed_other", "accel_other",
        "angle_other", "curvature_other",

        # pair metrics
        "nearest_dist", "nearest_rel_speed", "nearest_ttc",

        # encoded categories
        "lane_enc", "edge_enc", "veh_enc",
        "lane_other_enc", "edge_other_enc", "veh_other_enc",
    ]

    # store label inside features_df too for easier sequence building
    # (but we won't write it into features.csv)
    # collision_label is already present in features_df
    final_features = features_df[final_cols + ["collision_label"]]

    # Save CSVs (features without label, labels separately)
    final_features[final_cols].to_csv(os.path.join(OUT_DIR, "features.csv"), index=False)
    labels_df.to_csv(os.path.join(OUT_DIR, "labels.csv"), index=False)

    print("✅ Saved features.csv and labels.csv")

    # ==========================================================
    # Build sequences per vehicle (ego = veh_id)
    # ==========================================================
    print("\n📌 Building sequences...")

    X_list, y_list = [], []

    df_sorted = final_features.sort_values(["veh_id", "time"]).reset_index(drop=True)

    feature_cols_for_lstm = [
        # ego A
        "x","y","speed","accel","angle","curvature",
        # other B
        "x_other","y_other","speed_other","accel_other","angle_other","curvature_other",
        # pairwise
        "nearest_dist","nearest_rel_speed","nearest_ttc",
        # encodings
        "lane_enc","edge_enc","veh_enc",
        "lane_other_enc","edge_other_enc","veh_other_enc",
    ]

    groups = df_sorted.groupby("veh_id").indices

    for veh, idx in groups.items():
        idx = sorted(idx)
        feats = df_sorted.loc[idx, feature_cols_for_lstm].values
        labs = df_sorted.loc[idx, "collision_label"].values

        n = len(feats)
        for i in range(n - WINDOW_SIZE):
            X_list.append(feats[i:i+WINDOW_SIZE])
            y_list.append(labs[i+WINDOW_SIZE])

    X = np.array(X_list)
    y = np.array(y_list)

    print("Raw sequences:", X.shape, "Labels:", Counter(y))

    # ==========================================================
    # Scaling
    # ==========================================================
    print("📌 Scaling features...")
    n_samples, win, n_feats = X.shape
    scaler = StandardScaler()
    X_flat = X.reshape(-1, n_feats)
    X_scaled = scaler.fit_transform(X_flat).reshape(n_samples, win, n_feats)
    joblib.dump(scaler, SCALER_PATH)

    # ==========================================================
    # Balancing dataset
    # ==========================================================
    if OVERSAMPLE:
        print("📌 Balancing minority class...")
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        if len(pos_idx) > 0 and len(neg_idx) > 0:
            oversampled_pos = np.random.choice(pos_idx, size=len(neg_idx), replace=True)
            X_bal = np.concatenate([X_scaled[neg_idx], X_scaled[oversampled_pos]])
            y_bal = np.concatenate([y[neg_idx], y[oversampled_pos]])

            perm = np.random.permutation(len(y_bal))
            X_final = X_bal[perm]
            y_final = y_bal[perm]
        else:
            print("⚠ Not enough positive/negative samples! Skipping oversampling.")
            X_final, y_final = X_scaled, y
    else:
        X_final, y_final = X_scaled, y

    print("Final dataset:", X_final.shape, Counter(y_final))

    # ==========================================================
    # Save sequences
    # ==========================================================
    np.save(os.path.join(SEQ_DIR, "X.npy"), X_final)
    np.save(os.path.join(SEQ_DIR, "y.npy"), y_final)

    print("\n🎉 Done! Saved X.npy, y.npy")
    print("Ready for model training.\n")


# ==========================================================
# Entry point
# ==========================================================
if __name__ == "__main__":
    main()
