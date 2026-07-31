#NOTE Add parameter condiguration on save, to save space on some data

# import numpy as np
# import copy

# class SpacecraftHistory:
#     def __init__(self):        

#         # True state (inertial frame)
#         self.true_pos = []
#         self.true_vel = []
#         self.true_accel = []
#         self.true_q = []
#         self.true_omega = []
#         self.true_alpha = []

#         # Body state
#         self.body_pos = []
#         self.body_vel = []
#         self.body_accel = []
#         self.body_q = []
#         self.body_omega = []
#         self.body_alpha = []

#         # Reference state (inertial frame)
#         self.true_ref_pos = []
#         self.true_ref_vel = []
#         self.true_ref_accel = []
#         self.true_ref_q = []
#         self.true_ref_omega = []
#         self.true_ref_alpha = []

#         # Navigation / Guidance / Control
#         self.navigation_estimated_data = []
#         self.guidance_reference_data = []
#         self.control_output_data = []

#         # Applied forces and torques
#         self.current_force_exerted = []
#         self.current_torque_exerted = []

#     def record(self, spacecraft_data):
#         # True state
#         self.true_pos.append(spacecraft_data.true_pos.copy())
#         self.true_vel.append(spacecraft_data.true_vel.copy())
#         self.true_accel.append(spacecraft_data.true_accel.copy())
#         self.true_q.append(spacecraft_data.true_q.copy())
#         self.true_omega.append(spacecraft_data.true_omega.copy())
#         self.true_alpha.append(spacecraft_data.true_alpha.copy())

#         # Body state
#         self.body_pos.append(spacecraft_data.body_pos.copy())
#         self.body_vel.append(spacecraft_data.body_vel.copy())
#         self.body_accel.append(spacecraft_data.body_accel.copy())
#         self.body_q.append(spacecraft_data.body_q.copy())
#         self.body_omega.append(spacecraft_data.body_omega.copy())
#         self.body_alpha.append(spacecraft_data.body_alpha.copy())

#         # Reference state
#         self.true_ref_pos.append(spacecraft_data.true_ref_pos.copy())
#         self.true_ref_vel.append(spacecraft_data.true_ref_vel.copy())
#         self.true_ref_accel.append(spacecraft_data.true_ref_accel.copy())
#         self.true_ref_q.append(spacecraft_data.true_ref_q.copy())
#         self.true_ref_omega.append(spacecraft_data.true_ref_omega.copy())
#         self.true_ref_alpha.append(spacecraft_data.true_ref_alpha.copy())

#         # Navigation / Guidance / Control
#         self.navigation_estimated_data.append(copy.deepcopy(spacecraft_data.navigation_estimated_data))
#         self.guidance_reference_data.append(copy.deepcopy(spacecraft_data.guidance_reference_data))
#         self.control_output_data.append(copy.deepcopy(spacecraft_data.control_output_data))

#         # Applied forces and torques
#         self.current_force_exerted.append(spacecraft_data.current_force_exerted.copy())
#         self.current_torque_exerted.append(spacecraft_data.current_torque_exerted.copy())

# # class SpacecraftHistory:
# #     def __init__(self):

# #         self.time = []
# #         self.position = []
# #         self.velocity = []
# #         self.acceleration = []
# #         self.quaternion = []
# #         self.angular_velocity = []
# #         self.angular_acceleration = []

# #         self.current_force_exerted = []
# #         self.current_torque_exerted = []


# #     def record(self, spacecraft_data):        
# #         self.position.append(spacecraft_data.true_pos.copy())
# #         self.velocity.append(spacecraft_data.true_vel.copy())
# #         self.acceleration.append(spacecraft_data.true_accel.copy())
# #         self.quaternion.append(spacecraft_data.true_q.copy())
# #         self.angular_velocity.append(spacecraft_data.true_omega.copy())
# #         self.angular_acceleration.append(spacecraft_data.true_alpha.copy())

# #         self.current_force_exerted.append(spacecraft_data.current_force_exerted.copy())
# #         self.current_torque_exerted.append(spacecraft_data.current_torque_exerted.copy())

# class SimulationHistory:
#     def __init__(self):
#         self.time = []
#         self.spacecrafts_history = {}

#     def record(self, simulation_data):
#         self.time.append(simulation_data.t)
#         for i, spacecraft in enumerate(simulation_data.spacecrafts):
#             name = spacecraft.name

#             if name not in self.spacecrafts_history:
#                 self.spacecrafts_history[name] = SpacecraftHistory()

#             self.spacecrafts_history[name].record(
#                 spacecraft.spacecraft_data
#             )

#     def finalize(self):
#         # Convert lists to numpy arrays for easier analysis
#         self.time = np.asarray(self.time)
        
#         for h in self.spacecrafts_history.values():

#             # ===========================
#             # True state
#             # ===========================
#             h.true_pos = np.asarray(h.true_pos)
#             h.true_vel = np.asarray(h.true_vel)
#             h.true_accel = np.asarray(h.true_accel)
#             h.true_q = np.asarray(h.true_q)
#             h.true_omega = np.asarray(h.true_omega)
#             h.true_alpha = np.asarray(h.true_alpha)

#             # ===========================
#             # Body state
#             # ===========================
#             h.body_pos = np.asarray(h.body_pos)
#             h.body_vel = np.asarray(h.body_vel)
#             h.body_accel = np.asarray(h.body_accel)
#             h.body_q = np.asarray(h.body_q)
#             h.body_omega = np.asarray(h.body_omega)
#             h.body_alpha = np.asarray(h.body_alpha)

#             # ===========================
#             # Reference state
#             # ===========================
#             h.true_ref_pos = np.asarray(h.true_ref_pos)
#             h.true_ref_vel = np.asarray(h.true_ref_vel)
#             h.true_ref_accel = np.asarray(h.true_ref_accel)
#             h.true_ref_q = np.asarray(h.true_ref_q)
#             h.true_ref_omega = np.asarray(h.true_ref_omega)
#             h.true_ref_alpha = np.asarray(h.true_ref_alpha)

#             # ===========================
#             # Forces and torques
#             # ===========================
#             h.current_force_exerted = np.asarray(h.current_force_exerted)
#             h.current_torque_exerted = np.asarray(h.current_torque_exerted)

#             # ===========================
#             # Navigation history
#             # ===========================
#             nav = h.navigation_estimated_data
#             h.navigation_estimated_data = {
#                 "spacecraft_position": np.asarray([x.spacecraft_position for x in nav]),
#                 "spacecraft_velocity": np.asarray([x.spacecraft_velocity for x in nav]),
#                 "spacecraft_acceleration": np.asarray([x.spacecraft_acceleration for x in nav]),
#                 "spacecraft_attitude": np.asarray([x.spacecraft_attitude for x in nav]),
#                 "spacecraft_angular_velocity": np.asarray([x.spacecraft_angular_velocity for x in nav]),
#                 "spacecraft_angular_acceleration": np.asarray([x.spacecraft_angular_acceleration for x in nav]),

#                 "reference_position": np.asarray([x.reference_position for x in nav]),
#                 "reference_velocity": np.asarray([x.reference_velocity for x in nav]),
#                 "reference_acceleration": np.asarray([x.reference_acceleration for x in nav]),
#                 "reference_attitude": np.asarray([x.reference_attitude for x in nav]),
#                 "reference_angular_velocity": np.asarray([x.reference_angular_velocity for x in nav]),
#                 "reference_angular_acceleration": np.asarray([x.reference_angular_acceleration for x in nav]),
#             }

#             # ===========================
#             # Guidance history
#             # ===========================
#             guide = h.guidance_reference_data
#             h.guidance_reference_data = {
#                 "position": np.asarray([x.position for x in guide]),
#                 "velocity": np.asarray([x.velocity for x in guide]),
#                 "acceleration": np.asarray([x.acceleration for x in guide]),
#                 "attitude": np.asarray([x.attitude for x in guide]),
#                 "angular_velocity": np.asarray([x.angular_velocity for x in guide]),
#                 "angular_acceleration": np.asarray([x.angular_acceleration for x in guide]),
#             }

#             # ===========================
#             # Control history
#             # ===========================
#             control = h.control_output_data
#             h.control_output_data = {
#                 "force": np.asarray([x.force for x in control]),
#                 "torque": np.asarray([x.torque for x in control]),
#             }

import copy
from dataclasses import is_dataclass
import numpy as np


def _stack(obj_list):
    """
    Convierte una lista de dataclasses en un diccionario de numpy arrays
    de forma recursiva.
    """

    first = obj_list[0]

    if is_dataclass(first):
        return {
            field: _stack([getattr(obj, field) for obj in obj_list])
            for field in first.__dataclass_fields__
        }

    if isinstance(first, np.ndarray):
        return np.asarray(obj_list)

    return obj_list


class SpacecraftHistory:

    def __init__(self):

        self.true_state = []

        self.estimated_data = []
        self.reference_data = []
        self.control_output_data = []

        self.current_force_exerted = []
        self.current_torque_exerted = []

    def record(self, data):

        self.true_state.append(copy.deepcopy(data.true_state))

        self.estimated_data.append(copy.deepcopy(data.estimated_data))
        self.reference_data.append(copy.deepcopy(data.reference_data))
        self.control_output_data.append(copy.deepcopy(data.control_output_data))

        self.current_force_exerted.append(data.current_force_exerted.copy())
        self.current_torque_exerted.append(data.current_torque_exerted.copy())

    def finalize(self):

        self.true_state = _stack(self.true_state)

        self.estimated_data = _stack(self.estimated_data)

        self.reference_data = _stack(self.reference_data)

        self.control_output_data = _stack(self.control_output_data)

        self.current_force_exerted = np.asarray(self.current_force_exerted)

        self.current_torque_exerted = np.asarray(self.current_torque_exerted)


class SimulationHistory:

    def __init__(self):

        self.time = []
        self.spacecrafts_history = {}

    def record(self, simulation_data):

        self.time.append(simulation_data.t)

        for spacecraft in simulation_data.spacecrafts:

            history = self.spacecrafts_history.setdefault(
                spacecraft.name,
                SpacecraftHistory()
            )

            history.record(spacecraft.spacecraft_data)

    def finalize(self):

        self.time = np.asarray(self.time)

        for history in self.spacecrafts_history.values():
            history.finalize()