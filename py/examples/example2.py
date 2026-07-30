import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Define the environment
from py.modules.enviroments.environments import ClassicalEnvironment

env = ClassicalEnvironment() # NOTE: Need to review for proper implmentation of the environment class

# Then, define the propagators
from py.modules.propagators.native_propagator import NativeTranslationalPropagator

translational_propagator = NativeTranslationalPropagator(integration_method="NATIVE_RK45")

from py.modules.new.simulation import Simulation
sim = Simulation(max_sim_time=3*60*60,
                 dt_propagation= 1, # 100 Hz,
                 environment= env,
                 translational_propagator_engine= translational_propagator,
                 verbose = True
)

from py.modules.sensors.generic_sensor import AbsoluteSensor

sensors = [AbsoluteSensor(100, verbose=False)]  # Sample rate of 100 Hz

initial_position_SC1 = np.array([7000e3, 0, 3500e3])  # Initial position in meters
initial_velocity_SC1 = np.array([0, 7.5e3, 0])
initial_attitude = np.array([0, 0, 0, 1])  # Initial quaternion
initial_angular_velocity = np.array([0, 0, 0])  # Initial angular velocity in rad/s

sim.add_spacecraft(
    name="SC1",
    initial_position=initial_position_SC1,
    initial_velocity=initial_velocity_SC1,
    sensors=sensors
)
initial_position_SC2 = np.array([0, 7000e3, 0])  # Initial position in meters
initial_velocity_SC2 = np.array([7.5e3, 0, 0])

sim.add_spacecraft(
    name="SC2",
    initial_position=initial_position_SC2,
    initial_velocity=initial_velocity_SC2,    
    sensors=sensors
)

# We can now simulate
result = sim.simulate()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Obtener datos de ambos spacecrafts
sc1_data = result.spacecrafts_history["SC1"]
sc2_data = result.spacecrafts_history["SC2"]

t = result.time

# ===========================
# Position vs Time
# ===========================

plt.figure(figsize=(12, 8))

plt.subplot(2,1,1)

plt.plot(t, sc1_data.true_pos[:,0], label="SC1 X")
plt.plot(t, sc1_data.true_pos[:,1], label="SC1 Y")
plt.plot(t, sc1_data.true_pos[:,2], label="SC1 Z")

plt.plot(t, sc2_data.true_pos[:,0], "--", label="SC2 X")
plt.plot(t, sc2_data.true_pos[:,1], "--", label="SC2 Y")
plt.plot(t, sc2_data.true_pos[:,2], "--", label="SC2 Z")

plt.title("Spacecraft Position Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid(True)
plt.legend()


# ===========================
# Velocity vs Time
# ===========================

plt.subplot(2,1,2)

plt.plot(t, sc1_data.true_vel[:,0], label="SC1 Vx")
plt.plot(t, sc1_data.true_vel[:,1], label="SC1 Vy")
plt.plot(t, sc1_data.true_vel[:,2], label="SC1 Vz")

plt.plot(t, sc2_data.true_vel[:,0], "--", label="SC2 Vx")
plt.plot(t, sc2_data.true_vel[:,1], "--", label="SC2 Vy")
plt.plot(t, sc2_data.true_vel[:,2], "--", label="SC2 Vz")

plt.title("Spacecraft Velocity Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# ===========================
# 2D Orbit XY
# ===========================

plt.figure(figsize=(8,8))

r1 = sc1_data.true_pos
r2 = sc2_data.true_pos

plt.plot(r1[:,0], r1[:,1], label="SC1 Orbit")
plt.plot(r2[:,0], r2[:,1], "--", label="SC2 Orbit")


# Earth
earth = plt.Circle(
    (0,0),
    6378e3,
    color="blue",
    alpha=0.3,
    label="Earth"
)

plt.gca().add_patch(earth)


# Start/end points
plt.scatter(r1[0,0], r1[0,1], color="green", marker="o", label="SC1 Start")
plt.scatter(r1[-1,0], r1[-1,1], color="red", marker="o", label="SC1 End")

plt.scatter(r2[0,0], r2[0,1], color="green", marker="x", label="SC2 Start")
plt.scatter(r2[-1,0], r2[-1,1], color="red", marker="x", label="SC2 End")


plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Spacecraft Orbits - REL2BP")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()


# ===========================
# 3D Orbit
# ===========================

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection="3d")

ax.plot(
    r1[:,0],
    r1[:,1],
    r1[:,2],
    label="SC1"
)

ax.plot(
    r2[:,0],
    r2[:,1],
    r2[:,2],
    "--",
    label="SC2"
)


# Start points
ax.scatter(
    r1[0,0],
    r1[0,1],
    r1[0,2],
    color="green"
)

ax.scatter(
    r2[0,0],
    r2[0,1],
    r2[0,2],
    color="lime"
)


ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title("3D Spacecraft Orbits")
ax.legend()

plt.show()