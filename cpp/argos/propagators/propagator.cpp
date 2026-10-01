#include "propagator.hpp"

#include "dynamics_factory.hpp"
#include "iostream"

namespace argos {


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
)
{
    // -------------------------------------------------------------------------
    // Resolve the physical dynamics.
    //
    // The factory creates a native C++ DynamicsFunction. After this point,
    // the RK45 solver never needs to cross the Python/C++ boundary.
    // -------------------------------------------------------------------------

    DynamicsFunction dynamics =
        make_translational_dynamics(
            dynamics_model,

            mass,
            applied_force,
            disturbance_force,

            orbital_model,

            mu,
            length_factor,
            time_factor
        );


    // -------------------------------------------------------------------------
    // Integrate using the generic C++ RK45 solver.
    // -------------------------------------------------------------------------

    return integrate_rk45(
        dynamics,

        t0,
        tf,

        initial_state,

        options
    );
}

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
)
{
    DynamicsFunction dynamics =
        make_translational_dynamics(
            dynamics_model,
            mass,
            applied_force,
            disturbance_force,
            orbital_model,
            mu,
            length_factor,
            time_factor
        );

    State derivative;

    dynamics(
        t,
        state,
        derivative
    );

    return derivative;
}

RK45Result propagate_rotational(
    const State& initial_state,

    double t0,
    double tf,

    const Matrix3& inertia_matrix,
    const Matrix3& inverse_inertia_matrix,

    const Vector3& applied_torque,
    const Vector3& disturbance_torque,

    const std::string& dynamics_model,

    const RK45Options& options
)
{
    // -------------------------------------------------------------------------
    // Resolve the physical dynamics.
    //
    // The factory creates a native C++ DynamicsFunction. After this point,
    // the RK45 solver never needs to cross the Python/C++ boundary.
    // -------------------------------------------------------------------------

    DynamicsFunction dynamics =
        make_rotational_dynamics(
            dynamics_model,

            inertia_matrix,
            inverse_inertia_matrix,

            applied_torque,
            disturbance_torque
        );


    // -------------------------------------------------------------------------
    // Integrate using the generic C++ RK45 solver.
    // -------------------------------------------------------------------------

    return integrate_rk45(
        dynamics,

        t0,
        tf,

        initial_state,

        options
    );
}


State evaluate_rotational_dynamics(
    const State& state,
    double t,

    const Matrix3& inertia_matrix,
    const Matrix3& inverse_inertia_matrix,

    const Vector3& applied_torque,
    const Vector3& disturbance_torque,

    const std::string& dynamics_model
)
{
    DynamicsFunction dynamics =
        make_rotational_dynamics(
            dynamics_model,

            inertia_matrix,
            inverse_inertia_matrix,

            applied_torque,
            disturbance_torque
        );

    State derivative;

    dynamics(
        t,
        state,
        derivative
    );

    return derivative;
}

}  // namespace argos