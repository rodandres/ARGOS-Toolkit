#pragma once

#include "../solvers/rk45.hpp"

#include <array>
#include <string>
#include <vector>

namespace argos {

using Vector3 = std::array<double, 3>;
using Vector4 = std::array<double, 4>;
using Matrix3 = std::array<std::array<double, 3>, 3>;

struct TranslationalModelResult {
    Vector3 velocity;
    Vector3 acceleration;
};

/**
 * Compute translational dynamics using the relative two-body model.
 *
 * State ordering:
 * [x, y, z, vx, vy, vz]
 */
TranslationalModelResult rel2bp(
    double t,
    const State& state,
    double mu
);


/**
 * Compute translational dynamics using the circular restricted
 * three-body problem.
 *
 * State ordering:
 * [x, y, z, vx, vy, vz]
 */
TranslationalModelResult cr3bp(
    double t,
    const State& state,
    double mu,
    double length_factor,
    double time_factor
);


/**
 * Compute the complete translational equations of motion.
 */
void newton(
    double t,
    const State& state,
    double mass,
    const Vector3& applied_force,
    const Vector3& disturbance_force,
    const std::string& orbital_model,
    double mu,
    double length_factor,
    double time_factor,
    State& derivative
);


/**
 * Validate a rotational state.
 */
void validate_rotational_state(const State& state);


/**
 * Compute the complete rotational equations of motion.
 *
 * State ordering:
 * [qx, qy, qz, qw, wx, wy, wz]
 *
 * Quaternion convention:
 * [qx, qy, qz, qw]
 *
 * Rotational dynamics:
 *
 *     omega_dot =
 *         I^-1 * (
 *             tau
 *             - omega x (I * omega)
 *         )
 *
 * Quaternion kinematics:
 *
 *     q_dot = 0.5 * Omega(q) * omega
 */
void quaternion_dynamics(
    double t,
    const State& state,
    const Matrix3& inertia_matrix,
    const Matrix3& inverse_inertia_matrix,
    const Vector3& applied_torque,
    const Vector3& disturbance_torque,
    State& derivative
);

}  // namespace argos