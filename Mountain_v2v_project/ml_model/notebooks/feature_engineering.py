import pandas as pd
import numpy as np
import os

# ----------------- Paths -----------------
BASE_DIR = os.path.abspath("../../simulation")
VEHICLE_FILE = os.path.join(BASE_DIR, "output", "vehicle_data.csv")
COLLISION_FILE = os.path.join(BASE_DIR, "output", "collision_labels.csv")

FEATURES_FILE = os.path.abspath("../dataset/features.csv")
LABELS_FILE = os.path.abspath("../dataset/labels.csv")

# Make sure output folder exists
os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)

# ----------------- Load Data -----------------
vehicles_df = pd.read_csv(VEHICLE_FILE)
collisions_df = pd.read_csv(COLLISION_FILE)

# ----------------- Initialize Features -----------------
feature_list = []

for t in vehicles_df['time'].unique():
    timestep_data = vehicles_df[vehicles_df['time'] == t].to_dict('records')

    for v in timestep_data:
        vid = v['veh_id']
        x1, y1 = v['x'], v['y']
        speed1 = v['speed']
        accel1 = v['accel']

        # Initialize nearest neighbor features
        min_dist = float('inf')
        rel_speed = 0
        ttc = float('inf')

        for other in timestep_data:
            if other['veh_id'] == vid:
                continue
            dx = x1 - other['x']
            dy = y1 - other['y']
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                rel_speed = abs(speed1 - other['speed'])
                ttc = dist / rel_speed if rel_speed > 0 else np.inf

        feature_list.append({
            "time": t,
            "veh_id": vid,
            "speed": speed1,
            "acceleration": accel1,
            "nearest_dist": min_dist,
            "relative_speed": rel_speed,
            "ttc": ttc
        })

# ----------------- Convert to DataFrame -----------------
features_df = pd.DataFrame(feature_list)

# ----------------- Merge with collision labels -----------------
collision_map = {}
for _, row in collisions_df.iterrows():
    collision_map[(row['time'], row['veh1'])] = row['collision_risk']
    collision_map[(row['time'], row['veh2'])] = row['collision_risk']

labels = []
for _, row in features_df.iterrows():
    key = (row['time'], row['veh_id'])
    labels.append(collision_map.get(key, 0))  # default 0 if no collision

# ----------------- Save CSVs -----------------
features_df.to_csv(FEATURES_FILE, index=False)
pd.DataFrame({"collision": labels}).to_csv(LABELS_FILE, index=False)

print(f"Feature CSV saved at: {FEATURES_FILE}")
print(f"Labels CSV saved at: {LABELS_FILE}")
