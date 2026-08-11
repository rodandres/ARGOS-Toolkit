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

translational_propagator = NativeTranslationalPropagator(dynamics="REL2BP", integration_method="NATIVE_RK45")
translational_propagator2 = NativeTranslationalPropagator(dynamics="REL2BP", integration_method="NATIVE_RK45")

from py.modules.new.simulation import Simulation
sim = Simulation(max_sim_time=1*60*60,                 
                 environment= env,                 
                 verbose = True
)


dt1 = 20.0  # Propagation update every 10 seconds
dt2 = 10.0  # Propagation update every 10 seconds

from py.modules.sensors.generic_sensor import AbsoluteSensor

sensors = [AbsoluteSensor(1/dt1, verbose=False)]  # Sample rate of 100 Hz
sensors2 = [AbsoluteSensor(1/dt2, verbose=False)]  # Sample rate of 100 Hz

from py.modules.navigation.basic_laws import IdealNavigation
from py.general.dataclasses import MissionPhase
from py.modules.new.mission_manager import MissionManager

nav_law = IdealNavigation()
nav_law2 = IdealNavigation()

phase = MissionPhase(
    name="Phase 1",
    navigation=nav_law,
    dt_nav=dt1,  # Navigation update every 1 second
    translational_model=translational_propagator,
    dt_propagation= dt1  # Propagation update every 1 second
)
phase2 = MissionPhase(
    name="Phase 1",
    navigation=nav_law2,
    dt_nav=dt2,  # Navigation update every 1 second
    dt_propagation=dt2,  # Propagation update every 10 seconds
    translational_model=translational_propagator2
)

mission_manager = MissionManager(
    initial_phase=phase
)

mission_manager2 = MissionManager(
    initial_phase=phase2
) 

initial_position_SC2 = np.array([0, 7000e3, 0])  # Initial position in meters
initial_velocity_SC2 = np.array([7.5e3, 0, 0])

sim.add_spacecraft(
    name="SC2",
    initial_position=initial_position_SC2,
    initial_velocity=initial_velocity_SC2,
    sensors=sensors2,
    mission_manager=mission_manager2,
)

initial_position_SC1 = np.array([7000e3, 0, 3500e3])  # Initial position in meters
initial_velocity_SC1 = np.array([0, 7.5e3, 0])
initial_attitude = np.array([0, 0, 0, 1])  # Initial quaternion
initial_angular_velocity = np.array([0, 0, 0])  # Initial angular velocity in rad/s

sim.add_spacecraft(
    name="SC1",
    initial_position=initial_position_SC1,
    initial_velocity=initial_velocity_SC1,
    sensors=sensors,
    mission_manager=mission_manager,
    target_name="SC2"
)

# We can now simulate
result = sim.simulate()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# ==========================================================
# Get spacecraft history
# ==========================================================

sc1 = result.spacecrafts_history["SC1"]
sc2 = result.spacecrafts_history["SC2"]

t1 = sc1.t
t2 = sc2.t


# True positions
r1 = sc1.true_state["position"]
r2 = sc2.true_state["position"]

print(t1.shape)
print(t2.shape)
print(r1.shape)
print(np.unique(r1, axis=0).shape)
print(r2.shape)
print(np.unique(r2, axis=0).shape)


# ==========================================================
# 2D Orbit XY
# ==========================================================

plt.figure(figsize=(8, 8))

plt.plot(
    r1[:, 0],
    r1[:, 1],
    label="SC1 Orbit"
)

plt.plot(
    r2[:, 0],
    r2[:, 1],
    "--",
    label="SC2 Orbit"
)


# Earth
earth = plt.Circle(
    (0, 0),
    6378e3,
    color="blue",
    alpha=0.3,
    label="Earth"
)

plt.gca().add_patch(earth)


# Initial/final points

plt.scatter(
    r1[0,0],
    r1[0,1],
    color="green",
    marker="o",
    label="SC1 Start"
)

plt.scatter(
    r1[-1,0],
    r1[-1,1],
    color="red",
    marker="o",
    label="SC1 End"
)


plt.scatter(
    r2[0,0],
    r2[0,1],
    color="green",
    marker="x",
    label="SC2 Start"
)

plt.scatter(
    r2[-1,0],
    r2[-1,1],
    color="red",
    marker="x",
    label="SC2 End"
)


plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.title("Spacecraft Orbits - XY Plane")
plt.axis("equal")
plt.grid(True)
plt.legend()



# ==========================================================
# 3D Orbit
# ==========================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)


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


ax.scatter(
    r1[0,0],
    r1[0,1],
    r1[0,2],
    color="green",
    label="SC1 Start"
)


ax.scatter(
    r2[0,0],
    r2[0,1],
    r2[0,2],
    color="lime",
    label="SC2 Start"
)


ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

ax.set_title("3D Spacecraft Orbits")
ax.legend()



# ==========================================================
# Position history
# True / Estimated own / Estimated target
# ==========================================================


def plot_position(ax, t, data, title):

    labels = [
        "X",
        "Y",
        "Z"
    ]

    for i in range(3):
        ax.plot(
            t,
            data[:,i],
            label=labels[i]
        )

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True)
    ax.legend()



fig, ax = plt.subplots(
    3,
    2,
    figsize=(16,12),
    constrained_layout=True
)


# ==========================================================
# SC1
# ==========================================================

plot_position(
    ax[0,0],
    t1,
    sc1.true_state["position"],
    "SC1 - True Position"
)


plot_position(
    ax[1,0],
    t1,
    sc1.estimated_data["spacecraft_state"]["position"],
    "SC1 - Estimated Own Position"
)


plot_position(
    ax[2,0],
    t1,
    sc1.estimated_data["reference_state"]["position"],
    "SC1 - Estimated Target Position"
)



# ==========================================================
# SC2
# ==========================================================

plot_position(
    ax[0,1],
    t2,
    sc2.true_state["position"],
    "SC2 - True Position"
)


plot_position(
    ax[1,1],
    t2,
    sc2.estimated_data["spacecraft_state"]["position"],
    "SC2 - Estimated Own Position"
)


plot_position(
    ax[2,1],
    t2,
    sc2.estimated_data["reference_state"]["position"],
    "SC2 - Estimated Target Position"
)



# ==========================================================
# Quaternion comparison SC1
# ==========================================================

fig, ax = plt.subplots(
    3,
    1,
    figsize=(12,10),
    sharex=True
)


true_q = sc1.true_state["attitude"]

estimated_q = sc1.estimated_data["spacecraft_state"]["attitude"]

reference_q = sc1.estimated_data["reference_state"]["attitude"]


for i in range(4):

    ax[0].plot(
        t1,
        true_q[:,i],
        label=f"q{i}"
    )

    ax[1].plot(
        t1,
        estimated_q[:,i],
        label=f"q{i}"
    )

    ax[2].plot(
        t1,
        reference_q[:,i],
        label=f"q{i}"
    )


ax[0].set_title("SC1 True Quaternion")
ax[1].set_title("SC1 Estimated Quaternion")
ax[2].set_title("SC1 Reference Quaternion")


for a in ax:
    a.grid(True)
    a.legend()


ax[-1].set_xlabel("Time [s]")



# ==========================================================
# Control output
# ==========================================================

force = sc1.control_output_data["force"]
torque = sc1.control_output_data["torque"]

fig, ax = plt.subplots(
    2,
    1,
    figsize=(12,8),
    sharex=True
)


labels = [
    "X",
    "Y",
    "Z"
]


for i in range(3):

    ax[0].plot(
        t1,
        force[:,i],
        label=f"F{labels[i]}"
    )


    ax[1].plot(
        t1,
        torque[:,i],
        label=f"T{labels[i]}"
    )


ax[0].set_title("SC1 Control Force")
ax[1].set_title("SC1 Control Torque")


for a in ax:
    a.grid(True)
    a.legend()


ax[-1].set_xlabel("Time [s]")



plt.show()