import numpy as np
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.general.data_classes_declaration import *
from py.modules.math import quaternion_from_euler
from py.modules.sensors.generic_sensor import GaussianCovarianceSensor
from py.modules.controllers.classic_controllers import PDAttitudeController
from py.modules.actuators.actuators_base import RCSThruster_old as RCSThruster
from py.modules.enviroments.perturbations import GravityGradientPerturbation, SolarPressurePerturbation

from py.sim_engines.rotational_dynamics_engine import sim
from py.modules.visualization.attitude_graphs import animate_attitude_NEW, plot_torques_NEW, plot_states_NEW
from py.general.general_data import *

I = np.diag([Ix_total, Iy_total, Iz_total]) # Imported from general_data.py
I_inv = np.linalg.inv(I)

objective_orientation = [0, 0, 0]  # Desired orientation in Euler angles (degrees)
initial_orientation = [45, -30, -10]  # Initial orientation in Euler angles (degrees)
initial_angular_velocity = np.array([-0.08, 0.05, 0.1])  # Initial angular velocity in rad/s
initial_quat = quaternion_from_euler(np.deg2rad(initial_orientation[0]), np.deg2rad(initial_orientation[1]), np.deg2rad(initial_orientation[2]))
desired_quat = quaternion_from_euler(np.deg2rad(objective_orientation[0]), np.deg2rad(objective_orientation[1]), np.deg2rad(objective_orientation[2]))


sensor_model = GaussianCovarianceSensor(
    noise_covariance=np.diag([
            1e-7, 1e-7, 1e-7, 1e-7,   # quaternion (muy bajo ruido)
            1e-5, 1e-5, 1e-5          # gyro (rad/s)^2
        ]),
    random_seed=42
)

controller = PDAttitudeController(    
    proportional_gain=1000,
    derivative_gain=20000,
    maximum_torque=1000,
    minimum_torque=-1000
)

thruster_X_pos = RCSThruster(
    controlled_axis=0,
    thrust_direction=1,
    maximum_torque=1000,
    command_mode="PWM",
    modulation_window=0.1,
    minimum_on_time=0.01
)

thruster_X_neg = RCSThruster(
    controlled_axis=0,
    thrust_direction=-1,
    maximum_torque=1000,
    command_mode="PWM",
    modulation_window=0.1,
    minimum_on_time=0.01
)

thruster_Y_pos = RCSThruster(
    controlled_axis=1,
    thrust_direction=1,
    maximum_torque=1000,
    command_mode="PWM",
    modulation_window=0.1,
    minimum_on_time=0.01
)

thruster_Y_neg = RCSThruster(
    controlled_axis=1,
    thrust_direction=-1,
    maximum_torque=1000,
    command_mode="PWM",
    modulation_window=0.1,
    minimum_on_time=0.01
)

thruster_Z_pos = RCSThruster(
    controlled_axis=2,
    thrust_direction=1,
    maximum_torque=1000,
    command_mode="PWM",
    modulation_window=0.1,
    minimum_on_time=0.01
)

thruster_Z_neg = RCSThruster(
    controlled_axis=2,
    thrust_direction=-1,
    maximum_torque=1000,
    command_mode="PWM",
    modulation_window=0.1,
    minimum_on_time=0.01
)

solar_pressure_perturbation = SolarPressurePerturbation(
    solar_pressure= 4.5e-6, # Solar radiation pressure at 1 AU [N/m^2]
    reflectivity_coefficient= 1.5, # Reflectivity coefficient
    exposed_area= 3, # Cross-sectional area (m^2)
    center_of_pressure_offset= np.array([0.05, 0, 0]), # Vector from center of mass to center of pressure (m)
    surface_normal_body= np.array([0, 0, 1]), # Normal vector of the body surface (in body frame)
    sun_direction_inertial= np.array([1, 0, 0]) # Sun direction in
)

gravity_gradient_perturbation = GravityGradientPerturbation(
    gravitational_parameter= 3.986e14, # Earth's gravitational parameter (m^3/s
    inertial_position= np.array([7000e3, 0, 0]) # Position vector of the satellite in inertial frame (m)
)
    
general_settings_data = GeneralSimulationSettings(    
    total_sim_time = 3*60,
    navigation_freq = 10.0, # Hz
    guidance_freq = 1.0, # Hz
    control_freq = 10.0, # Hz
    dynamics_freq = 100.0, # Hz
    
    q0 = initial_quat,
    w0 = initial_angular_velocity,
    
    initial_state = np.concatenate([initial_quat, initial_angular_velocity]), # Quat (x, y, z, w) + Angular velocity (rad/s)    
)

dynamics_settings = DynamicsSettings(
    I = I,
    I_inv = I_inv,
)
    
sim_settings = SimulationSettings(
    general_settings = general_settings_data,
    controller = controller,
    sensor = sensor_model,
    actuators = [thruster_X_pos, thruster_X_neg, thruster_Y_pos, thruster_Y_neg, thruster_Z_pos, thruster_Z_neg],
    perturbations = [solar_pressure_perturbation, gravity_gradient_perturbation],
    dynamics_settings = dynamics_settings
)

_, results = sim(sim_settings, desired_quat)


sim_args = {
    "Simulation Settings": {
        "Total sim time": 3*60, # in seconds
        "Control frequency": 10, # in Hz
        "Dynamics frequency": 100, # in Hz
        "Initial quat": initial_quat, # x,y,z,w
        "Initial angular rate": initial_angular_velocity, # x,y,z
    },
    'Sensors':{
        "Model": None,
        "noise_cov": np.diag([
            1e-7, 1e-7, 1e-7, 1e-7,   # quaternion (muy bajo ruido)
            1e-5, 1e-5, 1e-5          # gyro (rad/s)^2
        ])
    },
    "Controller":{
        "Desired quat": desired_quat, # x,y,z,w
        "Control type": "PD",
        'Kp': 1000, 
        'Kd': 20000,
        'Tau_max': 1000,
        'Tau_min': -1000,
    },

    "Actuators":{
        "X":{
            'Actuator type': 'RCS',
            'Command': {
                'Method': 'PWM',
                'Modulation window': 0.5,
                'Minimum on time': 0.01
            },
            'Physical model': {
                'Max val': 1000
            }            
        },
        "Y":{
            'Actuator type': 'RCS',
            'Command': {
                'Method': 'PWM',
                'Modulation window': 0.5,
                'Minimum on time': 0.01
            },
            'Physical model': {
                'Max val': 1000
            }            
        },
        "Z":{
            'Actuator type': 'RCS',
            'Command': {
                'Method': 'PWM',
                'Modulation window': 0.5,
                'Minimum on time': 0.01
            },
            'Physical model': {
                'Max val': 1000
            }            
        },
    },
    "Dynamics": {
        'I': I,
        'I_inv': I_inv,
        'Perturbations': { 
            'SRP_active': True,
            'SRP_params':{ 
                'sun_i': np.array([1, 0, 0]), # Sun direction in inertial frame (unit vector)
                'P': 4.5e-6, # Solar radiation pressure at 1
                'Cr': 1.5, # Reflectivity coefficient
                'A': 3, # Cross-sectional area (m^2)
                'r_cp_cm': np.array([0.05, 0, 0]), # Vector from center of mass to center of pressure (m)
                'n_body': np.array([0, 0, 1]) # Normal vector of the body surface (in body frame)
            },
            'Gravity_gradient_active': True,
            'Gravity_gradient_params': {
                'mu': 3.986e14, # Earth's gravitational parameter (m^3/s^2)
                'r_i': np.array([7000e3, 0, 0]) # Position vector of the satellite in inertial frame (m)
            },
        }
    }
} 

plot_torques_NEW(results)
plot_states_NEW(results, sim_args)
animate_attitude_NEW(results, sim_args, interval=1, frame_skip=100)