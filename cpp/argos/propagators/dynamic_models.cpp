#include "dynamic_models.hpp"

#include <cmath>
#include <stdexcept>

namespace argos {
namespace {

constexpr std::size_t TRANSLATIONAL_STATE_SIZE = 6;
constexpr std::size_t ROTATIONAL_STATE_SIZE = 7;
constexpr std::size_t VECTOR_SIZE = 3;
constexpr std::size_t QUATERNION_SIZE = 4;

/**
 * Validate a translational state.
 */
void validate_state(const State& state)
{
    if (state.size() != TRANSLATIONAL_STATE_SIZE) {
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

    derivative.resize(TRANSLATIONAL_STATE_SIZE);

    derivative[0] = state[3];
    derivative[1] = state[4];
    derivative[2] = state[5];

    derivative[3] = acceleration[0];
    derivative[4] = acceleration[1];
    derivative[5] = acceleration[2];

}

/**
 * Validate a rotational state.
 */
void validate_rotational_state(const State& state)
{
    if (state.size() != ROTATIONAL_STATE_SIZE) {
        throw std::invalid_argument(
            "Rotational dynamics requires a state of size 7."
        );
    }
}

// ============================================================================
// QUATERNION ROTATIONAL DYNAMICS
// ============================================================================

void quaternion_dynamics(
    double /*t*/,
    const State& state,
    const Matrix3& inertia_matrix,
    const Matrix3& inverse_inertia_matrix,
    const Vector3& applied_torque,
    const Vector3& disturbance_torque,
    State& derivative
)
{
    validate_rotational_state(state);
    validate_vector(applied_torque);
    validate_vector(disturbance_torque);

    // ------------------------------------------------------------------------
    // State
    //
    // q = [qx, qy, qz, qw]
    // omega = [wx, wy, wz]
    // ------------------------------------------------------------------------

    double qx = state[0];
    double qy = state[1];
    double qz = state[2];
    double qw = state[3];

    const double omega_x = state[4];
    const double omega_y = state[5];
    const double omega_z = state[6];

    // ------------------------------------------------------------------------
    // Normalize quaternion
    // ------------------------------------------------------------------------

    const double q_norm = std::sqrt(
        qx * qx
        + qy * qy
        + qz * qz
        + qw * qw
    );

    if (q_norm == 0.0) {
        throw std::invalid_argument(
            "Quaternion norm cannot be zero."
        );
    }

    qx /= q_norm;
    qy /= q_norm;
    qz /= q_norm;
    qw /= q_norm;

    // ------------------------------------------------------------------------
    // I * omega
    // ------------------------------------------------------------------------

    const Vector3 I_omega = {
        inertia_matrix[0][0] * omega_x
            + inertia_matrix[0][1] * omega_y
            + inertia_matrix[0][2] * omega_z,

        inertia_matrix[1][0] * omega_x
            + inertia_matrix[1][1] * omega_y
            + inertia_matrix[1][2] * omega_z,

        inertia_matrix[2][0] * omega_x
            + inertia_matrix[2][1] * omega_y
            + inertia_matrix[2][2] * omega_z
    };

    // ------------------------------------------------------------------------
    // omega x (I * omega)
    // ------------------------------------------------------------------------

    const Vector3 omega_cross_Iomega = {
        omega_y * I_omega[2]
            - omega_z * I_omega[1],

        omega_z * I_omega[0]
            - omega_x * I_omega[2],

        omega_x * I_omega[1]
            - omega_y * I_omega[0]
    };

    // ------------------------------------------------------------------------
    // Total applied torque
    // ------------------------------------------------------------------------

    const Vector3 total_torque = {
        applied_torque[0] + disturbance_torque[0],
        applied_torque[1] + disturbance_torque[1],
        applied_torque[2] + disturbance_torque[2]
    };

    // ------------------------------------------------------------------------
    // Euler rotational dynamics
    //
    // omega_dot =
    //     I^-1 * (
    //         tau
    //         - omega x (I * omega)
    //     )
    // ------------------------------------------------------------------------

    const Vector3 torque_term = {
        total_torque[0] - omega_cross_Iomega[0],
        total_torque[1] - omega_cross_Iomega[1],
        total_torque[2] - omega_cross_Iomega[2]
    };

    const Vector3 angular_acceleration = {
        inverse_inertia_matrix[0][0] * torque_term[0]
            + inverse_inertia_matrix[0][1] * torque_term[1]
            + inverse_inertia_matrix[0][2] * torque_term[2],

        inverse_inertia_matrix[1][0] * torque_term[0]
            + inverse_inertia_matrix[1][1] * torque_term[1]
            + inverse_inertia_matrix[1][2] * torque_term[2],

        inverse_inertia_matrix[2][0] * torque_term[0]
            + inverse_inertia_matrix[2][1] * torque_term[1]
            + inverse_inertia_matrix[2][2] * torque_term[2]
    };

    // ------------------------------------------------------------------------
    // Quaternion kinematics
    // ------------------------------------------------------------------------

    const double q_dot_x = 0.5 * (
        qw * omega_x
        + qy * omega_z
        - qz * omega_y
    );

    const double q_dot_y = 0.5 * (
        qw * omega_y
        + qz * omega_x
        - qx * omega_z
    );

    const double q_dot_z = 0.5 * (
        qw * omega_z
        + qx * omega_y
        - qy * omega_x
    );

    const double q_dot_w = 0.5 * (
        -qx * omega_x
        - qy * omega_y
        - qz * omega_z
    );

    // ------------------------------------------------------------------------
    // State derivative
    // ------------------------------------------------------------------------

    derivative.resize(ROTATIONAL_STATE_SIZE);

    derivative[0] = q_dot_x;
    derivative[1] = q_dot_y;
    derivative[2] = q_dot_z;
    derivative[3] = q_dot_w;

    derivative[4] = angular_acceleration[0];
    derivative[5] = angular_acceleration[1];
    derivative[6] = angular_acceleration[2];
}

}  // namespace argos