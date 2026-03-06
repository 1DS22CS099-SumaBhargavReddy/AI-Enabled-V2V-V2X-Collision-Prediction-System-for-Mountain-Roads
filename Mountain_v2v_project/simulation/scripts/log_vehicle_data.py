import traci
import sumolib
import os
import csv
import math

# ----------------- PATH FIX (Windows Safe) -----------------
def fix(path):
    return os.path.abspath(path).replace("\\", "/")

BASE_DIR = fix("../map")
SUMO_CFG = fix("../map/simulation.sumocfg")
OUTPUT_FILE = fix("../output/vehicle_data.csv")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print("SUMO CFG:", SUMO_CFG)
print("Output will be saved to:", OUTPUT_FILE)

# ----------------- Instead of reading net.xml manually -----------------
# We use SUMO via TraCI to get edge shapes → 100% Windows safe

def get_edge_curvature_from_traci(edge_id):
    """Compute curvature from edge shape directly via TraCI (Windows compatible)."""
    try:
        shape = traci.lane.getShape(edge_id + "_0")
        if len(shape) < 3:
            return 0.0

        curvature = 0
        for i in range(1, len(shape) - 1):
            x1, y1 = shape[i - 1]
            x2, y2 = shape[i]
            x3, y3 = shape[i + 1]

            a = math.dist((x1, y1), (x2, y2))
            b = math.dist((x2, y2), (x3, y3))
            c = math.dist((x1, y1), (x3, y3))

            if a * b != 0:
                cos_val = (a*a + b*b - c*c) / (2*a*b)
                curvature += abs(math.acos(max(-1, min(1, cos_val))))

        return curvature
    except:
        return 0.0


# ----------------- START SUMO -----------------
sumoBinary = sumolib.checkBinary("sumo")
traci.start([sumoBinary, "-c", SUMO_CFG])

# ----------------- CSV FILE -----------------
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "time", "veh_id",
        "x", "y",
        "speed", "accel",
        "angle",
        "lane_id", "edge_id",
        "curvature",
        "veh_type"
    ])

    step = 0
    while step < 2000:
        traci.simulationStep()

        for vid in traci.vehicle.getIDList():

            x, y = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
            accel = traci.vehicle.getAcceleration(vid)
            angle = traci.vehicle.getAngle(vid)

            lane_id = traci.vehicle.getLaneID(vid)
            edge_id = "unknown"

            if lane_id and "_" in lane_id:
                edge_id = lane_id.rsplit("_", 1)[0]

            curvature = get_edge_curvature_from_traci(edge_id)

            veh_type = traci.vehicle.getTypeID(vid)

            writer.writerow([
                step, vid,
                x, y,
                speed, accel,
                angle,
                lane_id, edge_id,
                curvature,
                veh_type
            ])

        step += 1

traci.close()
print("Vehicle data logged at:", OUTPUT_FILE)
