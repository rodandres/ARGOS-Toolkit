import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py.modules.enviroments.environments import ClassicalEnvironment

env = ClassicalEnvironment()

from py.modules.core.simulation import Simulation

sim = Simulation(max_sim_time=5*60, # 3 minutes,                 
                 environment= env,                 
                 verbose = True
)

from py.modules.actuators.RCS import RCSThruster

thruster_X_pos = RCSThruster(
    nominal_thrust= 0,
    direction= np.array([1, 0, 0]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
)

thruster_X_neg = RCSThruster(
    nominal_thrust= 0,
    direction= np.array([-1, 0, 0]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
    
)

thruster_Y_pos = RCSThruster(
    nominal_thrust= 0,
    direction= np.array([0, 1, 0]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
)

thruster_Y_neg = RCSThruster(
    nominal_thrust= 0,
    direction= np.array([0, -1, 0]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
)

thruster_Z_pos = RCSThruster(
    nominal_thrust= 0,
    direction= np.array([0, 0, 1]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0   
)

thruster_Z_neg = RCSThruster(
    nominal_thrust= 0,
    direction= np.array([0, 0, -1]),
    override_torque_value = 300,
    modulation_window= 0.1,
    minimum_on_time= 0.01,
    activation_threshold= 0.0
)

thrusters = [thruster_X_pos, thruster_X_neg, thruster_Y_pos, thruster_Y_neg, thruster_Z_pos, thruster_Z_neg]

print(thruster_Z_neg.is_available())
print(thruster_Z_neg.has_been_used())
print(thruster_Z_neg.is_controllable())

thruster_X_pos.print_information()

from py.modules.sensors.generic_sensor import AbsoluteSensor

sensors = [AbsoluteSensor(100, verbose=False)]  # Sample rate of 100 Hz

from py.modules.propagators.native_propagator import NativeRotationalPropagator

rot_propagator = NativeRotationalPropagator(integration_method="NATIVE_RK45")
from py.general.dataclasses import GuidanceReference, StateVariables
from py.modules.guidance.basic_laws import CustomGuidanceLaw
from py.modules.math import quaternion_from_euler

objective_orientation = [0, 0, 0]  # Desired orientation in Euler angles (degrees)


def compute_reference(navigation_estimated_data, simulation_data):

    ref = GuidanceReference(
        state=StateVariables(
            attitude=quaternion_from_euler(
                np.deg2rad(objective_orientation[0]),
                np.deg2rad(objective_orientation[1]),
                np.deg2rad(objective_orientation[2]),
            )
        )
    )
    return ref

guidance_law = CustomGuidanceLaw(
    custom_reference_function= compute_reference
)

from py.modules.guidance.basic_laws import ConstantReferenceGuidance

guidance_law = ConstantReferenceGuidance(
    desired_quat = quaternion_from_euler(np.deg2rad(objective_orientation[0]), np.deg2rad(objective_orientation[1]), np.deg2rad(objective_orientation[2]))
)
from py.general.dataclasses import EstimationOutput
from py.modules.navigation.basic_laws import CustomNavigation

def estimate(sensors):

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
from py.modules.navigation.basic_laws import IdealNavigation

navigation_law = IdealNavigation()  # Using the ideal navigation law for this example

from py.modules.controllers.classic_controllers import PDAttitudeController

control_law = PDAttitudeController(
    proportional_gain= 1000,
    derivative_gain= 20000,
    maximum_torque= 1000,
    minimum_torque= -1000
)

from py.modules.controllers.basic_laws import CustomControlAllocator

def allocate(control_output, actuators):
    
    torque_to_allocate = control_output.torque

    # Extract direction and magnitude of the torque to allocate
    torque_magnitude = np.linalg.norm(torque_to_allocate)
    torque_direction = torque_to_allocate / torque_magnitude if torque_magnitude != 0 else np.zeros(3)
    
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

allocator_law = CustomControlAllocator(
    allocation_function= allocate
)

from py.modules.controllers.basic_laws import BasicRCSAllocator

allocator_law = BasicRCSAllocator()

from py.general.dataclasses import MissionPhase

phase = MissionPhase(
    name="SinglePhase",
    guidance= guidance_law,
    navigation= navigation_law,
    controller= control_law,

    allocator = allocator_law,

    dt_nav = 0.1,  # Navigation update rate (10 Hz)
    dt_guid = 1,  # Guidance update rate (1 Hz)
    dt_control = 0.1, # Control update rate (10 Hz)

    rotational_model= rot_propagator,
    dt_propagation= 0.01 # Propagation time step (100 Hz)
)

from py.modules.core.mission_manager import MissionManager

mission_manager = MissionManager(
    initial_phase= phase
)

from py.modules.math import quaternion_from_euler
from py.general.general_data import Ix_total, Iy_total, Iz_total

initial_vel = np.array([0.0, 0.0, 0.0])  # Initial velocity in meters per second
initial_orientation = initial_orientation = [45, -30, -10]  # Initial orientation in Euler angles (degrees)
initial_orientation_quat = quaternion_from_euler(np.deg2rad(initial_orientation[0]), np.deg2rad(initial_orientation[1]), np.deg2rad(initial_orientation[2]))
initial_angular_velocity = np.array([-0.08, 0.05, 0.1])  # Initial angular velocity in rad/s

sim.add_spacecraft(    
    initial_attitude= initial_orientation_quat,
    initial_angular_velocity= initial_angular_velocity,
    inertia_tensor= np.diag([Ix_total, Iy_total, Iz_total]),  # Inertia tensor in kg*m^2
    actuators = thrusters,
    sensors = sensors,
    mission_manager = mission_manager,
    verbose = True
)

sc = sim.simulation_data.spacecrafts[0]
sc.show_sensors_info()
sc.update_sensor_name(current_name="Sensor_0", new_name="Absolute Sensor")
sc.show_sensors_info()

from py.modules.faults.sensors_fault_modes import SensorStuck
from py.modules.faults.fault_manager import FaultEvent
event_1 = FaultEvent(
    event_name= "Sensor Stuck Event 1",
    component_name= "Absolute Sensor",
    fault_mode= SensorStuck(),
    activation_condition= lambda sim_data: sim_data.current_time >= 60,  # Activate after 60 seconds
    deactivation_condition= lambda sim_data: sim_data.current_time >= 120  # Deactivate after 120 seconds
)

sc.fault_manager.add_fault_event(event_1)

event_2 = FaultEvent(
    event_name= "Sensor Stuck Event 2",
    component_name= "Absolute Sensor",
    fault_mode= SensorStuck(),
    activation_condition= lambda sim_data: sim_data.current_time >= 180,  # Activate after 180 seconds
    deactivation_condition= lambda sim_data: sim_data.current_time >= 240  # Deactivate after 240 seconds
)

event_3 = FaultEvent(
    event_name= "Sensor Stuck Event 3",
    component_name= "Absolute Sensor",
    fault_mode= SensorStuck(),
    activation_condition= lambda sim_data: sim_data.current_time >= 300,  # Activate after 300 seconds    
)

sc.fault_manager.add_fault_event([event_2, event_3])
sc.show_faults_info()

#result = sim.simulate()

from py.modules.visualization.state_variables import plot_attitude_quaternions, plot_angular_velocity, plot_position
from py.modules.visualization.gnc import plot_control_result

#plot_attitude_quaternions("Spacecraft_1", result)
#plot_angular_velocity("Spacecraft_1", result)
#plot_control_result("Spacecraft_1", result)
#plot_position("Spacecraft_1", result)



