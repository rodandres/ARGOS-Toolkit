#pragma once

#include "../solvers/rk45.hpp"

#include <array>
#include <string>
#include <vector>

namespace argos {

using Vector3 = std::array<double, 3>;

struct TranslationalModelResult {
    Vector3 velocity;
    Vector3 acceleration;
};

/**
 * Compute translational dynamics using the relative two-body model.
 *
 * State ordering:
 * [x, y, z, vx, vy, vz]
 *
 * Position is expressed in meters and velocity in meters per second.
 * The gravitational parameter mu is expressed in m^3/s^2.
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
 *
 * The input state is expressed in SI units and internally converted
 * to normalized CR3BP units.
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
 *
 * The total acceleration is the sum of:
 *
 *     a_total = a_orbital + a_external
 *
 * where a_orbital is obtained from the selected orbital model and
 * a_external is obtained from the applied and disturbance forces.
 *
 * Supported orbital models:
 *
 *     "REL2BP"
 *     "CR3BP"
 *     "NEWTON"
 *
 * For "NEWTON", no orbital acceleration is applied.
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

}  // namespace argos