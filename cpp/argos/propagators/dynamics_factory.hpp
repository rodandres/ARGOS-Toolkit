#pragma once

#include "../solvers/rk45.hpp"

#include <array>
#include <string>

namespace argos {

using Vector3 = std::array<double, 3>;

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

}  // namespace argos