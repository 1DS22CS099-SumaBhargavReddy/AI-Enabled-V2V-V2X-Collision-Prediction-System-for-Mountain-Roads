import os
import time
import numpy as np
import joblib

# TraCI / SUMO imports
import sumolib
import traci

# ----------------- Config -----------------
SEQ_LEN = 10                     # sequence length used during training
FEATURES = ["speed", "accel", "ttc"]  # order must match prepare_sequences
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "trained_model"))
SEQUENCES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "sequences"))
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Choose which tflite to use (dynamic quant recommended for CPU)
TFLITE_PATH = os.path.join(MODEL_DIR, "model_dynamic.tflite")
if not os.path.exists(TFLITE_PATH):
    TFLITE_PATH = os.path.join(MODEL_DIR, "model_fp32.tflite")

# SUMO configuration (adjust to your paths)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMO_CFG = os.path.join(PROJECT_ROOT, "simulation", "map", "simulation.sumocfg")

# Prediction thresholds & safety policy
PROB_THRESHOLD = 0.6   # probability above which we warn
EMERGENCY_PROB = 0.85  # emergency threshold
TTC_EMERGENCY = 1.0    # seconds

# Behavior suggestion parameters (tweak later)
MIN_SAFE_SPEED = 3.0   # m/s minimum safe crawl speed
SPEED_REDUCTION_FACTOR = 0.6  # reduce to 60% of current speed on warning
EMERGENCY_REDUCTION_FACTOR = 0.3  # reduce to 30% on emergency

# ----------------- Helpers -----------------
def make_input_sequence(window):
    """ window: list of feature tuples length SEQ_LEN
        returns shaped numpy array: (1, SEQ_LEN, n_features)
    """
    arr = np.array(window, dtype=np.float32)
    arr = arr.reshape(1, arr.shape[0], arr.shape[1])
    return arr

def recommend_speed(curr_speed, prob, ttc):
    """Simple rule-based action generator (customize as needed)."""
    if prob >= EMERGENCY_PROB or ttc <= TTC_EMERGENCY:
        return max(MIN_SAFE_SPEED, curr_speed * EMERGENCY_REDUCTION_FACTOR), "EMERGENCY_BRAKE"
    elif prob >= PROB_THRESHOLD:
        return max(MIN_SAFE_SPEED, curr_speed * SPEED_REDUCTION_FACTOR), "SLOW_DOWN"
    else:
        return curr_speed, "NO_ACTION"

# ----------------- Load scaler & TFLite model -----------------
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Scaler not found at: " + SCALER_PATH)

scaler = joblib.load(SCALER_PATH)
print("Loaded scaler from:", SCALER_PATH)
print("Using TFLite model at:", TFLITE_PATH)

# load TFLite interpreter
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------- Start SUMO (TraCI) -----------------
sumo_binary = sumolib.checkBinary('sumo')  # or 'sumo-gui'
traci.start([sumo_binary, "-c", SUMO_CFG])
print("SUMO started with config:", SUMO_CFG)

# Keep rolling windows per vehicle id
rolling = {}   # veh_id -> list of last SEQ_LEN feature rows (speed, accel, ttc)

try:
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step += 1

        # Read all vehicle ids
        vlist = traci.vehicle.getIDList()

        # Precompute positions/speeds for TTC calc
        # We'll compute pairwise nearest distances within 25 m like earlier logic
        positions = {}
        speeds = {}
        accels = {}
        for vid in vlist:
            x, y = traci.vehicle.getPosition(vid)
            positions[vid] = (x, y)
            speeds[vid] = traci.vehicle.getSpeed(vid)
            try:
                accels[vid] = traci.vehicle.getAcceleration(vid)
            except Exception:
                accels[vid] = 0.0

        # For each vehicle, compute TTC to nearest vehicle within DIST_THRESHOLD (25m)
        DIST_THRESHOLD = 25.0
        for vid in vlist:
            x1, y1 = positions[vid]
            v1 = speeds[vid]
            a1 = accels.get(vid, 0.0)

            # find nearest
            nearest_ttc = 9999.0
            for vid2 in vlist:
                if vid2 == vid:
                    continue
                x2, y2 = positions[vid2]
                dist = np.hypot(x2 - x1, y2 - y1)
                if dist > DIST_THRESHOLD:
                    continue
                v2 = speeds[vid2]
                rel_speed = abs(v1 - v2)
                if rel_speed <= 0.0001:
                    continue
                ttc = dist / rel_speed
                if ttc < nearest_ttc:
                    nearest_ttc = ttc

            if nearest_ttc == 9999.0:
                nearest_ttc = 1000.0

            # Build raw feature row (must match scaler's expected order)
            raw_row = [v1, a1, nearest_ttc]

            # maintain rolling window
            win = rolling.get(vid, [])
            win.append(raw_row)
            if len(win) > SEQ_LEN:
                win.pop(0)
            rolling[vid] = win

            # if we have full window, predict
            if len(win) == SEQ_LEN:
                seq_arr = np.array(win, dtype=np.float32)  # shape (SEQ_LEN, n_feats)
                # flatten for scaler (scaler expects 2D (N, n_features))
                seq_2d = seq_arr.reshape(-1, seq_arr.shape[-1])
                # scale per feature
                seq_scaled_2d = scaler.transform(seq_2d)
                # reshape back to (1, SEQ_LEN, n_feats)
                seq_scaled = seq_scaled_2d.reshape(1, SEQ_LEN, seq_arr.shape[-1]).astype(np.float32)

                # set input tensor (supporting models with single input)
                interpreter.set_tensor(input_details[0]['index'], seq_scaled)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details[0]['index'])
                prob = float(out.ravel()[0])

                # get recommendation
                suggested_speed, action = recommend_speed(v1, prob, nearest_ttc)

                # print/log the event (replace with message send later)
                if prob >= PROB_THRESHOLD:
                    print(f"[t={step}] WARNING vid={vid} prob={prob:.3f} ttc={nearest_ttc:.2f}s action={action} suggest_speed={suggested_speed:.2f} m/s")
                    # Example: apply via traci (optional)
                    # traci.vehicle.setSpeed(vid, suggested_speed)
                # else only log rarely
                elif step % 100 == 0:
                    print(f"[t={step}] vid={vid} prob={prob:.3f} ttc={nearest_ttc:.2f}s (OK)")

        # small sleep optional to slow loop when using sumo-gui
        time.sleep(0.001)

finally:
    traci.close()
    print("SUMO closed, script exiting.")
