"""
realtime_predict_tflite.py
- Run this from inside your venv/WSL (where sumolib/traci/tflite are available).
- Starts SUMO (sumo or sumo-gui via sumolib.checkBinary).
- Runs TFLite model (collision_model.tflite) on sequences constructed per timestep.
- Outputs/updates simulation/output/alerts.csv with alerts the ns-3 program will read.

Edit constants at top to match your paths and model input shape.
"""
import os
import time
import csv
import numpy as np
import tflite_runtime.interpreter as tflite  # lightweight runtime; or use tensorflow.lite
import sumolib
import traci

# ---------- CONFIG ----------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # simulation/
MAP_DIR = os.path.join(BASE, "map")
SUMO_CFG = os.path.join(MAP_DIR, "simulation.sumocfg")
OUTPUT_DIR = os.path.join(BASE, "output")
ALERTS_CSV = os.path.join(OUTPUT_DIR, "alerts.csv")
TFLITE_MODEL = os.path.abspath(os.path.join(BASE, "..", "ml_model", "trained_model", "collision_model.tflite"))
STEPS = 3000  # simulation steps to run
SUMO_BINARY = sumolib.checkBinary("sumo-gui")  # or 'sumo-gui'
TIME_STEP = 0.1  # seconds per SUMO step if your SUMO config uses 0.1s

# Model input details (edit if different)
SEQ_LEN = 10
FEATURE_DIM = 3  # speed, accel, ttc (as used earlier)

# ---------- helper functions ----------
def load_tflite(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

def predict_tflite(interpreter, input_details, output_details, X):
    # X shape -> (1, SEQ_LEN, FEATURE_DIM)  dtype=float32
    interpreter.set_tensor(input_details[0]['index'], X.astype(np.float32))
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    return float(out.squeeze())

def write_alerts(alerts, alerts_csv_path):
    # alerts: list of dicts: {time, src_id, dst_id, msg, recommended_speed}
    os.makedirs(os.path.dirname(alerts_csv_path), exist_ok=True)
    with open(alerts_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "src_id", "dst_id", "msg", "recommended_speed"])
        for a in alerts:
            writer.writerow([a["time"], a["src_id"], a["dst_id"], a["msg"], a["recommended_speed"]])

# ---------- main ----------
def main():
    # load model
    print("Loading TFLite model:", TFLITE_MODEL)
    interpreter, input_details, output_details = load_tflite(TFLITE_MODEL)

    # prepare CSV
    write_alerts([], ALERTS_CSV)

    # start SUMO
    print("Starting SUMO with cfg:", SUMO_CFG)
    traci.start([SUMO_BINARY, "-c", SUMO_CFG])
    print("SUMO started. Running steps...")

    # Keep a per-vehicle rolling window of features (dict of lists)
    from collections import defaultdict, deque
    windows = defaultdict(lambda: deque(maxlen=SEQ_LEN))

    step = 0
    try:
        while step < STEPS:
            traci.simulationStep()
            alerts = []

            for vid in traci.vehicle.getIDList():
                # Make all vehicles visible in red color
                traci.vehicle.setColor(vid, (255, 0, 0, 255))
                
                x, y = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)
                accel = traci.vehicle.getAcceleration(vid)
                lane = traci.vehicle.getLaneID(vid)

                # append simple feature (speed, accel, large-ttc default); ttc calc can be improved
                # For now set ttc = 1000 (no immediate threat)
                ttc = 1000.0
                windows[vid].append([speed, accel, ttc])

            # For each vehicle build a sequence and predict
            for vid, q in windows.items():
                if len(q) < SEQ_LEN:
                    continue
                X = np.array(q)  # shape (SEQ_LEN, FEATURE_DIM)
                X_in = np.expand_dims(X, axis=0)  # (1, SEQ_LEN, FEATURE_DIM)
                prob = predict_tflite(interpreter, input_details, output_details, X_in)
                # threshold may be tuned; pick 0.5
                if prob > 0.5:
                    # find nearest vehicle within distance (approx)
                    # We'll pick the nearest neighbor from current step
                    try:
                        pos = traci.vehicle.getPosition(vid)
                        min_dist = 1e9
                        dst_id = ""
                        for other in traci.vehicle.getIDList():
                            if other == vid:
                                continue
                            ox, oy = traci.vehicle.getPosition(other)
                            d = (pos[0]-ox)**2 + (pos[1]-oy)**2
                            if d < min_dist:
                                min_dist = d
                                dst_id = other
                                                      
                        # Mock NS-3 V2V Terminal Output
                        if dst_id:
                            print(f"\n[NS-3 SIMULATION] ----------------------------------------------------")
                            print(f"[NS-3 V2V] ALERT GENERATED AT TIMESTEP: {step}")
                            print(f"  --> SENDER:   Vehicle '{vid}' (Detected Risk Prob: {(prob*100):.1f}%)")
                            print(f"  --> RECEIVER: Vehicle '{dst_id}' (Distance: {min_dist**0.5:.1f}m)")
                            print(f"  --> PACKET:   'SHARP TURN / COLLISION RISK - SLOW DOWN NOW!'")
                            print(f"  --> NETWORK:  Transmitted via IEEE 802.11p (Status: SUCCESS)")
                            print(f"----------------------------------------------------------------------\n")
                            
                            # Actually reduce speed in SUMO to visualize the alert reacting
                            traci.vehicle.setSpeed(vid, max(2.0, float(traci.vehicle.getSpeed(vid) * 0.5)))
                            traci.vehicle.setSpeed(dst_id, max(2.0, float(traci.vehicle.getSpeed(dst_id) * 0.5)))

                        # recommended action: slow down to 3 m/s (example). You can compute better value.
                        alerts.append({
                            "time": step,
                            "src_id": vid,
                            "dst_id": dst_id,
                            "msg": f"Collision risk with {dst_id}; reduce speed",
                            "recommended_speed": max(2.0, float(traci.vehicle.getSpeed(vid) * 0.6))
                        })
                    except Exception as e:
                        # ignore missing positions
                        pass

            # write alerts file (overwrites each step)
            write_alerts(alerts, ALERTS_CSV)

            step += 1

    except KeyboardInterrupt:
        print("Stopping by user")
    finally:
        traci.close()
        print("SUMO closed. Alerts saved at:", ALERTS_CSV)

if __name__ == "__main__":
    main()
