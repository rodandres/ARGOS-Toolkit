# GENERAL NOTE:
# - Need to standarize the way of defining dts/freqs
# See the proper implementation of environment class

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
from py.modules.propagators.native_propagator import NativeRotationalPropagator

rot_propagator = NativeRotationalPropagator(integration_method="NATIVE_RK45")

# Now create the simulation object

from py.modules.new.simulation import Simulation
sim = Simulation(max_sim_time=5*60, # 3 minutes,
                 dt_propagation= 0.01, # 100 Hz,
                 environment= env,
                 rotational_propagator_engine= rot_propagator,
                 verbose = True
)

# --- FOR SPACECRAFT 1 ---
# First define the actuators
from py.modules.actuators.RCS import RCSThruster

thruster_X_pos = RCSThruster(
    nominal_thrust= 1 ,# Newtons, placeholder value
    direction= np.array([1, 0, 0]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
)

thruster_X_neg = RCSThruster(
    nominal_thrust= 1 ,# Newtons, placeholder value
    direction= np.array([-1, 0, 0]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
    
)

thruster_Y_pos = RCSThruster(
    nominal_thrust= 1 ,# Newtons, placeholder value
    direction= np.array([0, 1, 0]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
)

thruster_Y_neg = RCSThruster(
    nominal_thrust= 1 ,# Newtons, placeholder value
    direction= np.array([0, -1, 0]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
)

thruster_Z_pos = RCSThruster(
    nominal_thrust= 1 ,# Newtons, placeholder value
    direction= np.array([0, 0, 1]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
)

thruster_Z_neg = RCSThruster(
    nominal_thrust= 1 ,# Newtons, placeholder value
    direction= np.array([0, 0, -1]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
)

thrusters = [thruster_X_pos, thruster_X_neg, thruster_Y_pos, thruster_Y_neg, thruster_Z_pos, thruster_Z_neg]

# Then, define the sensors
# Add sensors
from py.modules.sensors.generic_sensor import AbsoluteSensor

sensors = [AbsoluteSensor(100, verbose=False)]  # Sample rate of 100 Hz

# Next step is define the guidance law
from py.general.dataclasses import GuidanceReference
from py.modules.guidance.guidance_base import GuidanceBase
from py.modules.guidance.basic_laws import ConstantReferenceGuidance, CustomGuidanceLaw
from py.modules.math import quaternion_from_euler

objective_orientation = [0, 0, 0]  # Desired orientation in Euler angles (degrees)


def compute_reference(navigation_estimated_data, simulation_sata):

    ref = GuidanceReference(
        attitude = quaternion_from_euler(np.deg2rad(objective_orientation[0]), np.deg2rad(objective_orientation[1]), np.deg2rad(objective_orientation[2]))
    )

    return ref

guidance_law = ConstantReferenceGuidance(
    desired_quat = quaternion_from_euler(np.deg2rad(objective_orientation[0]), np.deg2rad(objective_orientation[1]), np.deg2rad(objective_orientation[2]))
)

guidance_law = CustomGuidanceLaw(
    custom_reference_function= compute_reference
)

# Now define the navigation law
from py.general.dataclasses import EstimationOutput
from py.modules.navigation.basic_laws import IdealNavigation, CustomNavigation

def estimate( sensors):

    for sensor in sensors:
        if sensor.type   == "AbsoluteSensor":
            sensor_data = sensor.get_measurement()                

            estimated_attitude = sensor_data[0][3]
            estimated_angular_velocity = sensor_data[0][4]            

            return EstimationOutput(
                spacecraft_attitude=estimated_attitude,
                spacecraft_angular_velocity=estimated_angular_velocity
            )                

navigation_law = CustomNavigation(
    custom_estimation_function=estimate
)
navigation_law = IdealNavigation()  # Using the ideal navigation law for this example

# Next define the control law
from py.modules.controllers.classic_controllers import PDAttitudeController

control_law = PDAttitudeController(
    proportional_gain= 1000,
    derivative_gain= 20000,
    maximum_torque= 1000,
    minimum_torque= -1000
)

# And we need a control allocator as we have multiple actuators
from py.modules.controllers.basic_laws import CustomControlAllocator, BasicRCSAllocator

def allocate(control_output, actuators):
    #print("="*10)
    torque_to_allocate = control_output.torque
    # Extract direction and magnitude of the torque to allocate
    torque_magnitude = np.linalg.norm(torque_to_allocate)
    torque_direction = torque_to_allocate / torque_magnitude if torque_magnitude != 0 else np.zeros(3)
    #print(f"Torque to allocate: {torque_to_allocate}, Magnitude: {torque_magnitude}, Direction: {torque_direction}")
    for actuator in actuators:
        # Calculate the dot product between the actuator's direction and the torque direction
        dot_product = np.dot(actuator.direction, torque_direction)
        # If the dot product is positive, the actuator can contribute to the torque
        if dot_product > 0:
            # Allocate a portion of the torque to this actuator based on its direction
            allocated_torque = dot_product * torque_magnitude
            #print(f"Allocating torque {allocated_torque} to actuator with direction {actuator.direction}")
            # Set the actuator's command based on the allocated torque                
            actuator.set_command(allocated_torque)
        else:
            # If the actuator cannot contribute, set its command to zero
            actuator.set_command(0)


allocator_law = BasicRCSAllocator()
allocator_law = CustomControlAllocator(
    allocation_function= allocate
)

# With that, we can define a mission phase and a mission manager
# Classes

from py.general.dataclasses import MissionPhase
from py.modules.new.mission_manager import MissionManager


phase = MissionPhase(
    name="SinglePhase",
    guidance= guidance_law,
    navigation= navigation_law,
    controller= control_law,

    allocator = allocator_law,

    dt_nav = 0.1,  # Navigation update rate (10 Hz)
    dt_guid = 1,  # Guidance update rate (1 Hz)
    dt_control = 0.1 # Control update rate (10 Hz)
)



mission_manager = MissionManager(
    initial_phase= phase
)

# With all this, we can now add an spacecraft to the sim
from py.modules.math import quaternion_from_euler
from py.general.general_data import Ix_total, Iy_total, Iz_total

inital_pos = np.array([0.0, 0.0, 0.0])  # Initial position in meters
initial_vel = np.array([0.0, 0.0, 0.0])  # Initial velocity in meters per second
initial_orientation = initial_orientation = [45, -30, -10]  # Initial orientation in Euler angles (degrees)
initial_orientation_quat = quaternion_from_euler(np.deg2rad(initial_orientation[0]), np.deg2rad(initial_orientation[1]), np.deg2rad(initial_orientation[2]))
initial_angular_velocity = np.array([-0.08, 0.05, 0.1])  # Initial angular velocity in rad/s

sim.add_spacecraft(
    mass= 1.0, # kg NOTE: This is a placeholder value. Replace with the actual mass of the spacecraft.
    initial_state= np.concatenate((inital_pos, initial_vel, initial_orientation_quat, initial_angular_velocity)),
    inertia_tensor= np.diag([Ix_total, Iy_total, Iz_total]),  # Inertia tensor in kg*m^2
    actuators = thrusters,
    sensors = sensors,
    mission_manager = mission_manager,
    verbose = True
)

# We can now simulate
result = sim.simulate()


#print(result.time)

#print(result.spacecrafts_history[spacecrafts_history].quaternion[0])

import matplotlib.pyplot as plt

# # plt.plot(
# #     result.time,
# #     result.spacecrafts_history["Spacecraft_1"].quaternion[:, 0],
# #     label="q0"
# # )

# plt.figure(figsize=(10, 6))
# plt.plot(
#     result.time,
#     result.spacecrafts_history["Spacecraft_1"].current_torque_exerted[:, 0],
#     label="Tau X",
#     linestyle='--'
# )
# plt.plot(
#     result.time,
#     result.spacecrafts_history["Spacecraft_1"].current_torque_exerted[:, 1],
#     label="Tau Y"    
# )
# plt.plot(
#     result.time,
#     result.spacecrafts_history["Spacecraft_1"].current_torque_exerted[:, 2],
#     label="Tau Z",
#     linestyle='-.'
# )
# plt.legend()

# plt.figure(figsize=(10, 6))
# plt.plot(
#     result.time,
#     result.spacecrafts_history["Spacecraft_1"].quaternion[:, 0],
#     label="q0"
# )

# plt.plot(
#     result.time,
#     result.spacecrafts_history["Spacecraft_1"].quaternion[:, 1],
#     label="q1"
# )

# plt.plot(
#     result.time,
#     result.spacecrafts_history["Spacecraft_1"].quaternion[:, 2],
#     label="q2"
# )

# plt.plot(
#     result.time,
#     result.spacecrafts_history["Spacecraft_1"].quaternion[:, 3],
#     label="q3"
# )

# plt.legend()
# plt.show()


import matplotlib.pyplot as plt

history = result.spacecrafts_history["Spacecraft_1"]

t = result.time

true_q = history.true_q
navigation_q = history.navigation_estimated_data["spacecraft_attitude"]
guidance_q = history.guidance_reference_data["attitude"]

forces = history.control_output_data["force"]
torques = history.control_output_data["torque"]

fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)

# =====================================================
# Control Output
# =====================================================
ax = axs[0, 0]

for i, lbl in enumerate(("Fx", "Fy", "Fz")):
    ax.plot(t, forces[:, i], label=lbl)

for i, lbl in enumerate(("Tx", "Ty", "Tz")):
    ax.plot(t, torques[:, i], "--", label=lbl)

ax.set_title("Control Output")
ax.set_ylabel("Force / Torque")
ax.grid(True)
ax.legend(ncol=2)

# =====================================================
# Navigation Quaternion
# =====================================================
ax = axs[0, 1]

for i, lbl in enumerate(("q0", "q1", "q2", "q3")):
    ax.plot(t, navigation_q[:, i], label=lbl)

ax.set_title("Navigation Quaternion")
ax.set_ylabel("Quaternion")
ax.grid(True)
ax.legend()

# =====================================================
# Guidance Quaternion
# =====================================================
ax = axs[1, 0]

for i, lbl in enumerate(("q0", "q1", "q2", "q3")):
    ax.plot(t, guidance_q[:, i], label=lbl)

ax.set_title("Guidance Quaternion")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Quaternion")
ax.grid(True)
ax.legend()

# =====================================================
# True Quaternion
# =====================================================
ax = axs[1, 1]

for i, lbl in enumerate(("q0", "q1", "q2", "q3")):
    ax.plot(t, true_q[:, i], label=lbl)

ax.set_title("True Quaternion")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Quaternion")
ax.grid(True)
ax.legend()

plt.tight_layout()

# Torque commanded by the controller
control_torque = history.control_output_data["torque"]

# Torque actually applied to the spacecraft
applied_torque = history.current_torque_exerted

labels = ["X", "Y", "Z"]

fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i in range(3):
    axs[i].plot(
        t,
        control_torque[:, i],
        label="Control Torque",
        linewidth=2,
    )

    axs[i].plot(
        t,
        applied_torque[:, i],
        "--",
        label="Applied Torque",
        linewidth=2,
    )

    axs[i].set_ylabel(f"$\\tau_{labels[i]}$ [N·m]")
    axs[i].set_title(f"Torque {labels[i]}")
    axs[i].grid(True)
    axs[i].legend()

axs[-1].set_xlabel("Time [s]")

plt.tight_layout()
plt.show()