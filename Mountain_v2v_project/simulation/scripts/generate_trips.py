import os
import subprocess

# ----------------- Paths -----------------
SUMO_HOME = "C:/sumo-1.24.0"  # Update if required

BASE_DIR = os.path.abspath("../map")    # points to simulation/map (one level above scripts folder)
os.makedirs(BASE_DIR, exist_ok=True)    # ensure folder exists

NET_FILE = os.path.join(BASE_DIR, "mountain.net.xml").replace("\\", "/")
ROUTE_FILE = os.path.join(BASE_DIR, "routes.rou.xml").replace("\\", "/")
ADDITIONAL_FILE = os.path.join(BASE_DIR, "additional.xml").replace("\\", "/")

# ----------------- Create additional.xml with proper root -----------------
additional_content = """<additional>
    <vType id="car_slow" accel="2.0" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="8"/>
    <vType id="car_medium" accel="2.5" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="12"/>
    <vType id="car_fast" accel="3.0" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15"/>
</additional>
"""

with open(ADDITIONAL_FILE, "w") as f:
    f.write(additional_content)

print(f"Additional vehicle types saved at: {ADDITIONAL_FILE}")

# ----------------- Generate Trips -----------------
num_vehicles = 350
randomTrips_script = os.path.join(SUMO_HOME, "tools", "randomTrips.py").replace("\\", "/")

# Change working directory to BASE_DIR so SUMO reads files correctly
os.chdir(BASE_DIR)

cmd = [
    "python",
    randomTrips_script,
    "-n", NET_FILE,
    "-r", ROUTE_FILE,
    "-e", str(num_vehicles),
    "-p", "1.0",
    "--trip-attributes", 'type="car_slow" departSpeed="random"'
]

subprocess.run(cmd)
print(f"Trips generated successfully: {ROUTE_FILE}")
print("Vehicle types added to additional.xml")
