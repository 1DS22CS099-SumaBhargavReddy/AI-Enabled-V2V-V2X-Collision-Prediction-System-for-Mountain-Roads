import traci
import sumolib
import os
import csv

# ----------------- Paths -----------------
BASE_DIR = os.path.abspath("simulation/map")
SUMO_CFG = os.path.join(BASE_DIR, "simulation.sumocfg")  # make sure this file exists
OUTPUT_FILE = os.path.abspath("simulation/output/vehicle_data.csv")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ----------------- Start SUMO -----------------
sumoBinary = sumolib.checkBinary('sumo')  # use 'sumo-gui' to see simulation
traci.start([sumoBinary, "-c", SUMO_CFG])

# ----------------- Curve lanes (adjust based on your network) -----------------
curve_lanes = ["edge5_0", "edge7_0", "edge10_0"]  # put lane IDs of sharp curves

# ----------------- Open CSV -----------------
with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "veh_id", "x", "y", "speed", "accel"])

    step = 0
    while step < 2000:  # simulate 2000 steps
        traci.simulationStep()
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            speed = traci.vehicle.getSpeed(vid)
            accel = traci.vehicle.getAcceleration(vid)

            # Slow down vehicles on curve lanes
            lane_id = traci.vehicle.getLaneID(vid)
            if lane_id in curve_lanes:
                traci.vehicle.setSpeed(vid, min(speed, 5))  # reduce speed to 5 m/s

            writer.writerow([step, vid, x, y, speed, accel])

        step += 1

traci.close()
print(f"Vehicle data logged at: {OUTPUT_FILE}")
