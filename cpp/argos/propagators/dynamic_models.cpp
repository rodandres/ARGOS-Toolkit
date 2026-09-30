#include "dynamic_models.hpp"

#include <cmath>
#include <stdexcept>

namespace argos {
namespace {

constexpr std::size_t STATE_SIZE = 6;
constexpr std::size_t VECTOR_SIZE = 3;


/**
 * Validate a translational state.
 */
void validate_state(const State& state)
{
    if (state.size() != STATE_SIZE) {
        throw std::invalid_argument(
            "Translational dynamics requires a state of size 6."
        );
    }
}


/**
 * Validate a 3D vector.
 *
 * std::array already guarantees the size at compile time, so this
 * function is intentionally empty for now and exists only to keep
 * the structure explicit.
 */
void validate_vector(const Vector3& /*vector*/)
{
}


/**
 * Compute the norm of a 3D vector.
 */
double norm(const Vector3& vector)
{
    return std::sqrt(
        vector[0] * vector[0]
        + vector[1] * vector[1]
        + vector[2] * vector[2]
    );
}

}  // namespace


// ============================================================================
// REL2BP
// ============================================================================

TranslationalModelResult rel2bp(
    double /*t*/,
    const State& state,
    double mu
)
{
    validate_state(state);

    // ------------------------------------------------------------------------
    // Position
    // ------------------------------------------------------------------------

    const Vector3 position = {
        state[0],
        state[1],
        state[2]
    };

    // ------------------------------------------------------------------------
    // Velocity
    // ------------------------------------------------------------------------

    const Vector3 velocity = {
        state[3],
        state[4],
        state[5]
    };

    // ------------------------------------------------------------------------
    // Gravitational acceleration
    //
    //     a = -mu * r / |r|^3
    // ------------------------------------------------------------------------

    const double r = norm(position);

    if (r == 0.0) {
        throw std::invalid_argument(
            "REL2BP position norm cannot be zero."
        );
    }

    const double r_cubed = r * r * r;

    const double factor = -mu / r_cubed;

    const Vector3 acceleration = {
        factor * position[0],
        factor * position[1],
        factor * position[2]
    };

    return {
        velocity,
        acceleration
    };
}


// ============================================================================
// CR3BP
// ============================================================================

TranslationalModelResult cr3bp(
    double /*t*/,
    const State& state,
    double mu,
    double length_factor,
    double time_factor
)
{
    validate_state(state);

    // ------------------------------------------------------------------------
    // State in SI units
    // ------------------------------------------------------------------------

    double x = state[0];
    double y = state[1];
    double z = state[2];

    double x_dot = state[3];
    double y_dot = state[4];
    double z_dot = state[5];

    // ------------------------------------------------------------------------
    // Normalize position and velocity
    //
    //     r_normalized = r / L
    //
    //     v_normalized = v / (L / T)
    // ------------------------------------------------------------------------

    x /= length_factor;
    y /= length_factor;
    z /= length_factor;

    x_dot /= length_factor / time_factor;
    y_dot /= length_factor / time_factor;
    z_dot /= length_factor / time_factor;

    // ------------------------------------------------------------------------
    // Distances to the primary bodies
    // ------------------------------------------------------------------------

    const double r1_squared =
        (x + mu) * (x + mu)
        + y * y
        + z * z;

    const double r2_squared =
        (x - (1.0 - mu)) * (x - (1.0 - mu))
        + y * y
        + z * z;

    const double r1 = std::sqrt(r1_squared);
    const double r2 = std::sqrt(r2_squared);

    if (r1 == 0.0 || r2 == 0.0) {
        throw std::invalid_argument(
            "CR3BP position cannot coincide with a primary body."
        );
    }

    const double r1_cubed = r1 * r1 * r1;
    const double r2_cubed = r2 * r2 * r2;

    // ------------------------------------------------------------------------
    // CR3BP equations of motion
    // ------------------------------------------------------------------------

    const double omega_x =
        x
        - (1.0 - mu) * (x + mu) / r1_cubed
        - mu * (x - (1.0 - mu)) / r2_cubed;

    const double omega_y =
        y
        - (1.0 - mu) * y / r1_cubed
        - mu * y / r2_cubed;

    const double omega_z =
        -(1.0 - mu) * z / r1_cubed
        -mu * z / r2_cubed;

    double x_ddot =
        2.0 * y_dot + omega_x;

    double y_ddot =
        -2.0 * x_dot + omega_y;

    double z_ddot =
        omega_z;

    // ------------------------------------------------------------------------
    // Convert velocity and acceleration back to SI units
    // ------------------------------------------------------------------------

    x_dot *= length_factor / time_factor;
    y_dot *= length_factor / time_factor;
    z_dot *= length_factor / time_factor;

    x_ddot *= length_factor / (time_factor * time_factor);
    y_ddot *= length_factor / (time_factor * time_factor);
    z_ddot *= length_factor / (time_factor * time_factor);

    const Vector3 velocity = {
        x_dot,
        y_dot,
        z_dot
    };

    const Vector3 acceleration = {
        x_ddot,
        y_ddot,
        z_ddot
    };

    return {
        velocity,
        acceleration
    };
}


// ============================================================================
// NEWTON / TOTAL TRANSLATIONAL DYNAMICS
// ============================================================================

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
)
{
    validate_state(state);
    validate_vector(applied_force);
    validate_vector(disturbance_force);

    // ------------------------------------------------------------------------
    // Orbital dynamics
    // ------------------------------------------------------------------------

    Vector3 orbital_acceleration = {
        0.0,
        0.0,
        0.0
    };

    if (orbital_model == "REL2BP") {

        const TranslationalModelResult result =
            rel2bp(
                t,
                state,
                mu
            );

        orbital_acceleration = result.acceleration;
    }
    else if (orbital_model == "CR3BP") {

        const TranslationalModelResult result =
            cr3bp(
                t,
                state,
                mu,
                length_factor,
                time_factor
            );

        orbital_acceleration = result.acceleration;
    }
    else if (
        orbital_model == "NEWTON"
        || orbital_model.empty()
    ) {

        // No orbital model.
        //
        // This preserves the behavior of the Python implementation
        // when orbital_model is None.

        orbital_acceleration = {
            0.0,
            0.0,
            0.0
        };
    }
    else {

        throw std::invalid_argument(
            "Unsupported orbital model: "
            + orbital_model
            + ". Supported models are: REL2BP, CR3BP, NEWTON."
        );
    }

    // ------------------------------------------------------------------------
    // External acceleration
    //
    //     a_external = (F_applied + F_disturbance) / mass
    //
    // The Python implementation returns zero acceleration when
    // mass <= 0.
    // ------------------------------------------------------------------------

    Vector3 external_acceleration = {
        0.0,
        0.0,
        0.0
    };

    if (mass > 0.0) {

        external_acceleration = {
            (applied_force[0] + disturbance_force[0]) / mass,
            (applied_force[1] + disturbance_force[1]) / mass,
            (applied_force[2] + disturbance_force[2]) / mass
        };
    }

    // ------------------------------------------------------------------------
    // Total acceleration
    // ------------------------------------------------------------------------

    const Vector3 acceleration = {
        orbital_acceleration[0] + external_acceleration[0],
        orbital_acceleration[1] + external_acceleration[1],
        orbital_acceleration[2] + external_acceleration[2]
    };

    // ------------------------------------------------------------------------
    // State derivative
    //
    //     d/dt [r, v] = [v, a]
    // ------------------------------------------------------------------------

    derivative.resize(STATE_SIZE);

    derivative[0] = state[3];
    derivative[1] = state[4];
    derivative[2] = state[5];

    derivative[3] = acceleration[0];
    derivative[4] = acceleration[1];
    derivative[5] = acceleration[2];
}

}  // namespace argos