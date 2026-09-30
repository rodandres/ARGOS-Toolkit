#pragma once

#include "../solvers/rk45.hpp"

#include <array>
#include <string>

namespace argos {

using Vector3 = std::array<double, 3>;


/**
 * Propagate translational spacecraft dynamics using the
 * native C++ RK45 solver.
 *
 * The propagator is responsible for connecting the physical
 * configuration to the generic RK45 solver.
 *
 * The RK45 solver itself remains agnostic to the physical model.
 */
RK45Result propagate_translational(
    const State& initial_state,

    double t0,
    double tf,

    double mass,

    const Vector3& applied_force,
    const Vector3& disturbance_force,

    const std::string& dynamics_model,
    const std::string& orbital_model,

    double mu,
    double length_factor,
    double time_factor,

    const RK45Options& options
);

State evaluate_translational_dynamics(
    const State& state,
    double t,
    double mass,
    const Vector3& applied_force,
    const Vector3& disturbance_force,
    const std::string& dynamics_model,
    const std::string& orbital_model,
    double mu,
    double length_factor,
    double time_factor
);

}  // namespace argos