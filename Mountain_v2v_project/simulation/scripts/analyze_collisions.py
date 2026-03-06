import pandas as pd
import numpy as np
import os
import math
from collections import defaultdict

# =====================================================
#                 PATH CONFIGURATION
# =====================================================

# Script is inside:  simulation/scripts/
# vehicle_data.csv is inside:  simulation/output/
DATA_FILE = os.path.abspath("../output/vehicle_data.csv")
OUTPUT_DIR = os.path.abspath("../ml_model/dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(OUTPUT_DIR, "features.csv")
LABELS_FILE = os.path.join(OUTPUT_DIR, "labels.csv")

# =====================================================
#                   CONSTANTS
# =====================================================
DIST_THRESHOLD = 25.0     # meters - blind-curve collision zone
TTC_THRESHOLD = 2.0       # seconds
EPS = 1e-6
GRID_SIZE = 30.0          # spatial indexing grid (meters)

# =====================================================
#                   LOAD DATA
# =====================================================
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"❌ vehicle_data.csv not found at:\n{DATA_FILE}")

print(f"📌 Loading: {DATA_FILE}")
df = pd.read_csv(DATA_FILE)
df = df.sort_values(["time", "veh_id"]).reset_index(drop=True)

times = df["time"].unique()
print("Total time steps:", len(times))

# =====================================================
#          PRECOMPUTE HEADINGS (FAST METHOD)
# =====================================================
print("⚙ Precomputing headings...")

df["prev_x"] = df.groupby("veh_id")["x"].shift(1).fillna(df["x"])
df["prev_y"] = df.groupby("veh_id")["y"].shift(1).fillna(df["y"])

dx = df["x"] - df["prev_x"]
dy = df["y"] - df["prev_y"]

# heading in degrees (0–360)
df["heading"] = np.degrees(np.arctan2(dy, dx))

# =====================================================
#             MAIN COLLISION DETECTION LOOP
# =====================================================
features = []
labels = []

print("🚀 Starting timestep processing...")

for t in times:

    step_df = df[df.time == t]
    N = len(step_df)

    if N <= 1:
        continue

    # convert to arrays for speed
    x = step_df["x"].values
    y = step_df["y"].values
    v = step_df["speed"].values
    a = step_df["accel"].values
    head = step_df["heading"].values

    # =================================================
    #      BUILD SPATIAL GRID (FAST NEIGHBOR SEARCH)
    # =================================================
    grid = defaultdict(list)

    gx = (x // GRID_SIZE).astype(int)
    gy = (y // GRID_SIZE).astype(int)

    for i in range(N):
        grid[(gx[i], gy[i])].append(i)

    # =================================================
    #        CHECK COLLISIONS FOR EACH VEHICLE
    # =================================================
    for i in range(N):

        nearest_ttc = 9999.0

        ix, iy = gx[i], gy[i]
        vx, ax, hx = v[i], a[i], head[i]

        # Check 3x3 neighboring cells
        for dxg in (-1, 0, 1):
            for dyg in (-1, 0, 1):

                neigh_idx = grid.get((ix + dxg, iy + dyg), [])
                for j in neigh_idx:

                    if j == i:
                        continue

                    # -------- distance ----------
                    dxp = x[j] - x[i]
                    dyp = y[j] - y[i]
                    dist = math.sqrt(dxp*dxp + dyp*dyp)

                    if dist > DIST_THRESHOLD:
                        continue

                    # -------- heading difference ----------
                    dh = abs(hx - head[j])
                    dh = min(dh, 360 - dh)

                    # -------- relative speed ----------
                    rel_speed = abs(v[i] - v[j])
                    if rel_speed < EPS:
                        continue

                    ttc = dist / rel_speed

                    if ttc < nearest_ttc:
                        nearest_ttc = ttc

        # -------- append features ----------
        features.append([vx, ax, nearest_ttc])

        # -------- label collision ----------
        labels.append(1 if nearest_ttc < TTC_THRESHOLD else 0)

# =====================================================
#                 SAVE OUTPUT DATA
# =====================================================
features_df = pd.DataFrame(features, columns=['speed', 'accel', 'ttc'])
labels_df = pd.DataFrame(labels, columns=['collision_label'])

features_df.to_csv(FEATURES_FILE, index=False)
labels_df.to_csv(LABELS_FILE, index=False)

print("=======================================")
print("✅ Dataset Generated Successfully")
print("📁 Features saved at:", FEATURES_FILE)
print("📁 Labels saved at:", LABELS_FILE)
print("🚨 Total Collision Labels:", labels_df['collision_label'].sum())
print("=======================================")
