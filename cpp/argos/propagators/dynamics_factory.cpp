#include "dynamics_factory.hpp"

#include "dynamic_models.hpp"

#include <stdexcept>
#include <utility>

namespace argos {

DynamicsFunction make_translational_dynamics(
    const std::string& dynamics_model,
    double mass,
    const Vector3& applied_force,
    const Vector3& disturbance_force,
    const std::string& orbital_model,
    double mu,
    double length_factor,
    double time_factor
) {
    if (dynamics_model == "NEWTON") {

        return [
            mass,
            applied_force,
            disturbance_force,
            orbital_model,
            mu,
            length_factor,
            time_factor
        ](
            double t,
            const State& state,
            State& derivative
        ) {
            newton(
                t,
                state,
                mass,
                applied_force,
                disturbance_force,
                orbital_model,
                mu,
                length_factor,
                time_factor,
                derivative
            );
        };
    }

    if (dynamics_model == "REL2BP") {

        return [
            mu
        ](
            double t,
            const State& state,
            State& derivative
        ) {
            const auto result = rel2bp(
                t,
                state,
                mu
            );

            derivative.resize(6);

            for (std::size_t i = 0; i < 3; ++i) {
                derivative[i] = result.velocity[i];
                derivative[i + 3] = result.acceleration[i];
            }
        };
    }

    if (dynamics_model == "CR3BP") {

        return [
            mu,
            length_factor,
            time_factor
        ](
            double t,
            const State& state,
            State& derivative
        ) {
            const auto result = cr3bp(
                t,
                state,
                mu,
                length_factor,
                time_factor
            );

            derivative.resize(6);

            for (std::size_t i = 0; i < 3; ++i) {
                derivative[i] = result.velocity[i];
                derivative[i + 3] = result.acceleration[i];
            }
        };
    }

    throw std::invalid_argument(
        "Unsupported translational dynamics model: " +
        dynamics_model
    );
}

}  // namespace argos