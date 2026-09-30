#include "propagator.hpp"

#include "dynamics_factory.hpp"

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

}  // namespace argos