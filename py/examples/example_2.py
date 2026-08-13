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

dt1 = 10.0  # Propagation update every 10 seconds
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

from py.modules.visualization.state_variables import plot_attitude_quaternions, plot_position

plot_attitude_quaternions(["SC1", "SC2"], result, show=True)
plot_position(["SC1", "SC2"], result, show=True)

from py.modules.visualization.trajectories import *

plot_trajectory_xy(["SC1", "SC2"], result, show=True, body="Earth",)
plot_trajectory_xz(["SC1", "SC2"], result, show=True, body="Earth",)
plot_trajectory_yz(["SC1", "SC2"], result, show=True, body="Earth",)
plot_trajectory_3d(["SC1", "SC2"], result, show=True, body="Earth",)
plot_trajectory(["SC1", "SC2"], result, show=True, body="Earth",)
