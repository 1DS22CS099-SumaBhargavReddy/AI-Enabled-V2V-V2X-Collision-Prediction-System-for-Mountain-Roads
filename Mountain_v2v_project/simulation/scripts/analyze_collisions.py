import pandas as pd
import numpy as np
import os

# ----------------- Paths -----------------
DATA_FILE = os.path.abspath("simulation/output/vehicle_data.csv")
OUTPUT_DIR = os.path.abspath("ml_model/dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(OUTPUT_DIR, "features.csv")
LABELS_FILE = os.path.join(OUTPUT_DIR, "labels.csv")

# ----------------- Parameters -----------------
TTC_THRESHOLD = 2.0  # seconds, below this we consider a collision risk

# ----------------- Load vehicle data -----------------
df = pd.read_csv(DATA_FILE)

# Ensure timestep order
df = df.sort_values(by=['time', 'veh_id']).reset_index(drop=True)

# ----------------- Compute features -----------------
# Add lane id for each vehicle if needed
# For simplicity, we consider nearest vehicle ahead in x-direction

features = []
labels = []

for t in df['time'].unique():
    step_df = df[df['time'] == t]
    step_df = step_df.sort_values(by='x')  # sort by x position
    
    veh_ids = step_df['veh_id'].values
    positions = step_df[['x', 'y']].values
    speeds = step_df['speed'].values
    accels = step_df['accel'].values
    
    for i, vid in enumerate(veh_ids):
        x, y = positions[i]
        speed = speeds[i]
        accel = accels[i]
        
        # Find nearest vehicle ahead (in same lane approximation)
        ttc = np.inf
        for j in range(i+1, len(veh_ids)):
            x_ahead, y_ahead = positions[j]
            distance = x_ahead - x
            rel_speed = speed - speeds[j]
            if rel_speed > 0:
                ttc = distance / rel_speed
                break
        
        # Feature vector
        feature = [speed, accel, ttc if ttc != np.inf else 1000.0]
        features.append(feature)
        
        # Label: 1 if TTC below threshold, else 0
        label = 1 if ttc < TTC_THRESHOLD else 0
        labels.append(label)

# ----------------- Save features and labels -----------------
features_df = pd.DataFrame(features, columns=['speed', 'accel', 'ttc'])
labels_df = pd.DataFrame(labels, columns=['collision_label'])

features_df.to_csv(FEATURES_FILE, index=False)
labels_df.to_csv(LABELS_FILE, index=False)

print(f"Features saved at: {FEATURES_FILE}")
print(f"Labels saved at: {LABELS_FILE}")
print(f"Collision labels (1.0) count: {labels_df['collision_label'].sum()}")
