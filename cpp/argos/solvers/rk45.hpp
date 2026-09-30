#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace argos {

using State = std::vector<double>;

using DynamicsFunction =
    std::function<void(
        double t,
        const State& state,
        State& derivative
    )>;


/**
 * Configuration options for the RK45 solver.
 */
struct RK45Options {

    double h0 = 0.1;

    double h_min = 1e-6;

    double h_max = 1.0;

    double rtol = 1e-6;

    double atol = 1e-9;

    bool adaptive = true;

    std::size_t max_iter = 100000;

    bool verbose = false;
};


/**
 * Result returned by the RK45 solver.
 *
 * The structure contains the information required to construct
 * an OdeResult-compatible object on the Python side.
 */
struct RK45Result {

    // Integration history
    std::vector<double> t;

    std::vector<State> y;

    // Final state
    State state;

    // Solver status
    bool success = false;

    int status = 0;

    std::string message;

    // Integration statistics
    std::size_t nfev = 0;

    std::size_t njev = 0;

    std::size_t nlu = 0;

    // Event information
    std::vector<std::vector<double>> t_events;

    std::vector<std::vector<State>> y_events;

    // Step statistics
    std::size_t accepted_steps = 0;

    std::size_t rejected_steps = 0;
};


/**
 * Integrate an ordinary differential equation using the
 * Runge-Kutta-Fehlberg 4(5) method.
 *
 * The solver itself is completely agnostic to the physical model.
 * The DynamicsFunction provided by the caller is evaluated directly
 * in C++ during every RK stage.
 */
RK45Result integrate_rk45(
    const DynamicsFunction& dynamics,
    double t0,
    double tf,
    const State& initial_state,
    const RK45Options& options
);

}  // namespace argos