import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MU = 1.215e-2
TIME_FACTOR_SEC = 27.321661 / (2.0 * np.pi) * 24 * 3600     # time unit unidad -> sideral lunar month (days)
LENGTH_FACTOR = 384400.0e3


secondsdt_propagation=5e-4 * TIME_FACTOR_SEC 


# Define the environment
from py.modules.enviroments.environments import ClassicalEnvironment

env = ClassicalEnvironment() # NOTE: Need to review for proper implmentation of the environment class

# Then, define the propagators
from py.modules.propagators.native_propagator import NativeTranslationalPropagator

translational_propagator = NativeTranslationalPropagator(dynamics="CR3BP", integration_method="NATIVE_RK45")

from py.modules.new.simulation import Simulation
sim = Simulation(max_sim_time=0.75952417*2 * TIME_FACTOR_SEC,
                 environment= env,                 
                 verbose = True
)

from py.modules.sensors.generic_sensor import AbsoluteSensor

sensors = [AbsoluteSensor(1 / secondsdt_propagation, verbose=False)]  # Sample rate of 100 Hz

from py.modules.navigation.basic_laws import IdealNavigation
from py.general.dataclasses import MissionPhase
from py.modules.new.mission_manager import MissionManager

nav_law = IdealNavigation()

phase = MissionPhase(
    name="Phase 1",
    navigation=nav_law,
    dt_nav=secondsdt_propagation,  # Navigation update every 1 second
    dt_propagation=secondsdt_propagation,  # Propagation update every 10 secondsdt_propagation=5e-4, # 100 Hz,
    translational_model=translational_propagator

)

mission_manager = MissionManager(
    initial_phase=phase
)

initial_position_SC1 = np.array([1.02262383, 0, -0.18250869]) * LENGTH_FACTOR # Initial position in meters
initial_velocity_SC1 = np.array([0, -0.10456466, 0]) * LENGTH_FACTOR / TIME_FACTOR_SEC # Initial velocity in meters per second

sim.add_spacecraft(
    name="SC1",
    initial_position=initial_position_SC1,
    initial_velocity=initial_velocity_SC1,
    sensors=sensors,
    mission_manager=mission_manager,
)

# We can now simulate
result = sim.simulate()

from py.modules.visualization.trajectories import *

plot_trajectory_xy("SC1", result, show=True, body="Moon",
                       body_position=np.array([(1-MU)*LENGTH_FACTOR, 0.0, 0.0]),)
plot_trajectory_xz("SC1", result, show=True, body="Moon",body_position=np.array([(1-MU)*LENGTH_FACTOR, 0.0, 0.0]),)
plot_trajectory_yz("SC1", result, show=True, body="Moon",body_position=np.array([(1-MU)*LENGTH_FACTOR, 0.0, 0.0]),)
plot_trajectory_3d("SC1", result, show=True, body="Moon",body_position=np.array([(1-MU)*LENGTH_FACTOR, 0.0, 0.0]),)
plot_trajectory("SC1", result, show=True, body="Moon",body_position=np.array([(1-MU)*LENGTH_FACTOR, 0.0, 0.0]),)


















import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# ==========================================================
# Get spacecraft history
# ==========================================================

sc1 = result.spacecrafts_history["SC1"]
sc2 = result.spacecrafts_history["SC1"]

t = sc1.t  # Assuming both spacecrafts have the same time history


# True positions
r1 = sc1.true_state["position"]*384400.0
r2 = sc2.true_state["position"]*384400.0


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
    (385000, 0),
    1.737,
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
    t,
    sc1.true_state["position"]*384400.0,
    "SC1 - True Position"
)


plot_position(
    ax[1,0],
    t,
    sc1.estimated_data["spacecraft_state"]["position"]*384400.0,
    "SC1 - Estimated Own Position"
)


plot_position(
    ax[2,0],
    t,
    sc1.estimated_data["reference_state"]["position"]*384400.0,
    "SC1 - Estimated Target Position"
)



# ==========================================================
# SC2
# ==========================================================

plot_position(
    ax[0,1],
    t,
    sc2.true_state["position"]*384400.0,
    "SC2 - True Position"
)


plot_position(
    ax[1,1],
    t,
    sc2.estimated_data["spacecraft_state"]["position"]*384400.0,
    "SC2 - Estimated Own Position"
)


plot_position(
    ax[2,1],
    t,
    sc2.estimated_data["reference_state"]["position"]*384400.0,
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
        t,
        true_q[:,i],
        label=f"q{i}"
    )

    ax[1].plot(
        t,
        estimated_q[:,i],
        label=f"q{i}"
    )

    ax[2].plot(
        t,
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
        t,
        force[:,i],
        label=f"F{labels[i]}"
    )


    ax[1].plot(
        t,
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