import traci
import os
import sys
import time
import numpy as np
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Hide TensorFlow INFO logs
os.environ["TF_USE_LEGACY_KERAS"] = "1" # Allow loading older .h5 LSTM models

# Suppress standardizing scaler feature name warnings
warnings.filterwarnings("ignore")

# Fix paths to model and map
os.environ["SUMO_HOME"] = r"C:\Program Files (x86)\Eclipse\Sumo"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_DIR = os.path.join(BASE_DIR, "ml_model", "trained_model")
sys.path.append(ML_DIR)

try:
    from load_tflite_model import CollisionPredictor
except ImportError as e:
    print(f"Error loading TFLite model: {e}")
    sys.exit(1)

import csv

# Correct path to SUMO CFG
SUMO_CFG = os.path.join(BASE_DIR, "simulation", "map", "simulation.sumocfg")
CSV_LOG = os.path.join(BASE_DIR, "output", "alerts.csv")

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(CSV_LOG), exist_ok=True)

# Initialize CSV with headers
with open(CSV_LOG, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Sender_Vid", "Receiver_Vid", "Probability", "TTC", "Message"])

def send_v2v_warning(vid, closest_vid, prob, ttc):
    print(f"\n[NS-3 V2V SIMULATION] ------------------------------------------------", flush=True)
    print(f"BROADCAST WARNING: Vehicle '{vid}' is experiencing a high-risk situation!", flush=True)
    print(f"  --> Risk Probability:       {prob*100:.1f}%", flush=True)
    print(f"  --> Time-To-Collision (TTC): {ttc:.2f}s", flush=True)
    print(f"  --> Sent via IEEE 802.11p module to adjacent nodes", flush=True)
    print(f"----------------------------------------------------------------------\n", flush=True)
    
    # Save to CSV
    with open(CSV_LOG, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([traci.simulation.getTime(), vid, closest_vid, round(prob, 2), round(ttc, 2), "V2V Collision Alert"])

predictor = CollisionPredictor()

print("Starting SUMO + ML + NS3 V2V System...", flush=True)

# Maintain last 10 frames per vehicle
history = {}

# Make sure SUMO config exists
if not os.path.exists(SUMO_CFG):
    print(f"Error: Could not find SUMO config file at {SUMO_CFG}")
    sys.exit(1)

# Ensure SUMO GUI runs
import sumolib
import time
sumo_binary = sumolib.checkBinary("sumo-gui")

traci.start([sumo_binary, "-c", SUMO_CFG, "--start"])
try:
    traci.gui.setSchema("View #0", "standard")
except Exception as e:
    pass

STEP = 0
try:
    # Run for 240 seconds (4 minutes)
    while traci.simulation.getTime() < 240.0:
        traci.simulationStep()
        if traci.simulation.getTime() % 10.0 == 0:
            print(f"Simulation Progress: {traci.simulation.getTime():.1f}s / 240.0s", flush=True)
        vehicles = traci.vehicle.getIDList()

        for vid in vehicles:
            # Highlight vehicles for visualization in RED as requested
            traci.vehicle.setColor(vid, (255, 0, 0, 255))
            
            speed = traci.vehicle.getSpeed(vid)
            accel = traci.vehicle.getAcceleration(vid)
            x, y = traci.vehicle.getPosition(vid)

            # Compute approximate TTC to nearest vehicle
            ttc = 1000
            min_dist = 1000
            closest_vid = "unknown"
            for other in vehicles:
                if other == vid: continue
                ox, oy = traci.vehicle.getPosition(other)
                dist = np.sqrt((x-ox)**2 + (y-oy)**2)
                if dist < 25:
                    other_speed = traci.vehicle.getSpeed(other)
                    rel_speed = abs(speed - other_speed)
                    if dist < min_dist:
                        min_dist = dist
                        closest_vid = other
                    if rel_speed > 0:
                        ttc = min(ttc, dist / rel_speed)

            # Add to history
            if vid not in history:
                history[vid] = []
            history[vid].append([speed, accel, ttc])

            # Keep last 10 steps only
            if len(history[vid]) > 10:
                history[vid].pop(0)

            # Predict only when we have 10 frames
            if len(history[vid]) == 10:
                sequence = np.array(history[vid])
                alert, prob = predictor.predict(sequence)

                if alert or prob > 0.1:
                    # Turn both vehicles red
                    traci.vehicle.setColor(vid, (255, 0, 0, 255))
                    if closest_vid != "unknown":
                        traci.vehicle.setColor(closest_vid, (255, 0, 0, 255))
                        # Display warning text in SUMO GUI
                        try:
                            # Not all SUMO versions support this parameter properly but it's worth a try
                            traci.vehicle.setParameter(vid, "has.text", f"WARNING: {vid}->{closest_vid}")
                            
                            # Let's also add a polygon marker for a brief second to highlight the crash zone
                            poly_id = f"alert_{vid}_{STEP}"
                            traci.polygon.add(
                                poly_id, 
                                [(x-2, y-2), (x+2, y-2), (x+2, y+2), (x-2, y+2)], 
                                (255, 0, 0, 150),
                                layer=100
                            )
                        except Exception:
                            pass
                    
                    print(f"\n[ALERT] COLLISION WARNING: {vid} -> {closest_vid}", flush=True)
                    print(f"[*] Sending V2V Alert...", flush=True)
                    send_v2v_warning(vid, closest_vid, prob, ttc)

                    # SUMO Auto-slowdown reaction
                    # if they slow down too much (1.0) they might appear stuck.
                    # let's only reduce by 10%
                    safe_speed = max(5.0, speed * 0.9)
                    traci.vehicle.setSpeed(vid, safe_speed)
                else:
                    traci.vehicle.setColor(vid, (0, 255, 0, 255)) # Back to green if safe
                    traci.vehicle.setSpeed(vid, -1) # Default SUMO speed management
                    try:
                        traci.vehicle.setParameter(vid, "has.text", "")
                    except:
                        pass

        STEP += 1
        time.sleep(0.02) # Small delay to make it visible
except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
except Exception as e:
    import traceback
    print("\n[ERROR] CRITICAL ERROR ENCOUNTERED:")
    traceback.print_exc()
finally:
    traci.close()
    print("Simulation finished. NS-3 V2V results logged.")

