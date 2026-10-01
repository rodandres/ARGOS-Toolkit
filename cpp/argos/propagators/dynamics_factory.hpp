#pragma once

#include "../solvers/rk45.hpp"

#include <array>
#include <string>

namespace argos {

using Vector3 = std::array<double, 3>;
using Matrix3 = std::array<std::array<double, 3>, 3>;

DynamicsFunction make_translational_dynamics(
    const std::string& dynamics_model,
    double mass,
    const Vector3& applied_force,
    const Vector3& disturbance_force,
    const std::string& orbital_model,
    double mu,
    double length_factor,
    double time_factor
);

DynamicsFunction make_rotational_dynamics(
    const std::string& dynamics_model,
    const Matrix3& inertia_matrix,
    const Matrix3& inverse_inertia_matrix,
    const Vector3& applied_torque,
    const Vector3& disturbance_torque
);

}  // namespace argos