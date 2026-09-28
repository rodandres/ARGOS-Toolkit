import numpy as np

print_summary = False

# ------------------------------------------------------------------------------
# From Orion capsule, the following parameters are estimated for the Crew module
# beeing consider as a truncated cone
CM_base_radius = 2.5 # m
CM_top_radius = 1.2 # m
CM_height = 3.35 # m
CM_mass = 10387 # kg

# Creating a model of the capsule in Fusion 360 with a mass x1000 lower,
# the following parameters are obtained:
Ix_scaled = 19.218
Iy_scaled = 19.218
Iz_scaled = 12.341

# Adjusting the inertia values to match the estimated weight and volume
Ixx_CM_local = Ix_scaled * 1000
Iyy_CM_local = Iy_scaled * 1000
Izz_CM_local = Iz_scaled * 1000

# The SM is estimated to be cylinder with the following parameters:
SM_radius = 2.5 # m
SM_height = 4.79 # m
SM_mass = 15600 # kg

# Then the inertia is calculated:
SM_Ix_local = (1/12) * SM_mass * (3 * SM_radius**2 + SM_height**2)
SM_Iy_local = (1/12) * SM_mass * (3 * SM_radius**2 + SM_height**2)
SM_Iz_local = 0.5 * SM_mass * SM_radius**2

# To calculate the total inertia matrix, first the c.m of the SM is calculated
SM_cm_x = 0
SM_cm_y = 0
SM_cm_z = CM_height + (SM_height / 2)

# Then the c.m of the CM is calculated and translated into a global reference frame
# with the CM on top of the SM
CM_cm_x = 0
CM_cm_y = 0

numerator = (
    CM_base_radius**2
    + 2 * CM_base_radius * CM_top_radius
    + 3 * CM_top_radius**2
)

denominator = (
    CM_base_radius**2
    + CM_base_radius * CM_top_radius
    + CM_top_radius**2
)

CM_cm_z = (CM_height / 4) * (numerator / denominator)

CM_cm_z_global = CM_height + CM_cm_z
SM_cm_z_global = SM_cm_z
# In a combine aproximation where the CM is on top of the SM
cm_total_x = 0
cm_total_y = 0
cm_total_z = (
    CM_mass * CM_cm_z_global +
    SM_mass * SM_cm_z_global
) / (CM_mass + SM_mass)

# Calculate the inertias of the CM and SM with respect to the total c.m using the parallel axis theorem
d_CM = CM_cm_z_global - cm_total_z
d_SM = SM_cm_z_global - cm_total_z

Ixx_CM = Ixx_CM_local + CM_mass * d_CM**2
Iyy_CM = Iyy_CM_local + CM_mass * d_CM**2
Izz_CM = Izz_CM_local

Ixx_SM = SM_Ix_local + SM_mass * d_SM**2
Iyy_SM = SM_Iy_local + SM_mass * d_SM**2
Izz_SM = SM_Iz_local

# Finally, the total inertia is calculated by summing the contributions of the CM and SM
Ix_total = Ixx_CM + Ixx_SM
Iy_total = Iyy_CM + Iyy_SM
Iz_total = Izz_CM + Izz_SM

#print("Total Inertia Matrix (kg*m^2):")
#print(f"Ix: {Ix_total:.2f}")
#print(f"Iy: {Iy_total:.2f}")
#print(f"Iz: {Iz_total:.2f}")

# -------------------------------------------------------------------------------

# From the orion information, it is known that there are RCS thrusters in SM and CM
# In the CM there are 12 thruster with 73kg of thrust each
# In the SM there are 24 thrusters with 23kg of thrust each

# It is supposed that the same amount of thrusters is used for each axis, 
# where 6 axis are defined:
# - +X, -X, +Y, -Y, +Z, -Z

# This means that for the CM, there are 2 thrusters for each axis,
# And for the SM there are 4 thrusters for each axis

# For translational control, the minimum thrust is where only one thruster is firing per axis
# while the maximum thrust is where all the thrusters are firing per axis.

# It will be supposed that the thrusters when performing translational maneuvers, will not
# induced rotational torques, and viceversa.

# It will also be supposed, that for rotational control, the distance between the torquers is
# simetrically and is located at the same distance from the total c.m, which is at 4m

# Therefore, the combination of possible torques for the whole vehicle is defined as:

# 1 thruster firing in each oposite axis: 1 *73kg * 2 * 9.81m/s^2 * 4m = 5728.52 Nm
# 2 thrusters firing in each oposite axis: 2 * 73kg * 4 * 9.81m/s^2 * 4m = 22914.08 Nm

# 1 thruster firing in each oposite axis: 1 * 23kg * 4 * 9.81m/s^2 * 4m = 3608.16 Nm
# 2 thrusters firing in each oposite axis: 2 * 23kg * 4 * 9.81m/s^2 * 4m = 14432.64 Nm
# 3 thrusters firing in each oposite axis: 3 * 23kg * 4 * 9.81m/s^2 * 4m = 21612.24 Nm
# 4 thrusters firing in each oposite axis: 4 * 23kg * 4 * 9.81m/s^2 * 4m = 28832.64 Nm

distance_to_cm = 4 # m
thrust_1 = 73 * 9.81 # N
thrust_2 = 23 * 9.81 # N

torque_case_1 = 1* 2 * thrust_1 * distance_to_cm
torque_case_2 = 2 * 2 * thrust_1 * distance_to_cm

torque_case_3 = 1 * 2 * thrust_2 * distance_to_cm
torque_case_4 = 2 * 2 * thrust_2 * distance_to_cm
torque_case_5 = 3 * 2 * thrust_2 * distance_to_cm
torque_case_6 = 4 * 2 * thrust_2 * distance_to_cm

tau_levels_max = np.array([torque_case_1, torque_case_2, torque_case_3, torque_case_4, torque_case_5, torque_case_6])
tau_levels_min = -tau_levels_max
tau_levels = np.sort(np.concatenate((tau_levels_min, [0], tau_levels_max)))


if print_summary:
    print("Inertia Matrix (kg*m^2):")
    print(f"Ix: {Ix_total:.2f}")
    print(f"Iy: {Iy_total:.2f}")
    print(f"Iz: {Iz_total:.2f}")
    print("\nTorque Levels (Nm):")
    print(tau_levels)
