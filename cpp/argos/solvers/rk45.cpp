#include "rk45.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace argos {
namespace {

constexpr double SAFETY = 0.9;
constexpr double P = 5.0;

// -----------------------------------------------------------------------------
// Runge-Kutta-Fehlberg 4(5) coefficients
// -----------------------------------------------------------------------------

constexpr double C[] = {
    0.0,
    1.0 / 4.0,
    3.0 / 8.0,
    12.0 / 13.0,
    1.0,
    1.0 / 2.0
};

constexpr double A[6][5] = {
    {
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    },
    {
        1.0 / 4.0,
        0.0,
        0.0,
        0.0,
        0.0
    },
    {
        3.0 / 32.0,
        9.0 / 32.0,
        0.0,
        0.0,
        0.0
    },
    {
        1932.0 / 2197.0,
        -7200.0 / 2197.0,
        7296.0 / 2197.0,
        0.0,
        0.0
    },
    {
        439.0 / 216.0,
        -8.0,
        3680.0 / 513.0,
        -845.0 / 4104.0,
        0.0
    },
    {
        -8.0 / 27.0,
        2.0,
        -3544.0 / 2565.0,
        1859.0 / 4104.0,
        -11.0 / 40.0
    }
};

constexpr double B[] = {
    25.0 / 216.0,
    0.0,
    1408.0 / 2565.0,
    2197.0 / 4104.0,
    -1.0 / 5.0,
    0.0
};

constexpr double B_STAR[] = {
    16.0 / 135.0,
    0.0,
    6656.0 / 12825.0,
    28561.0 / 56430.0,
    -9.0 / 50.0,
    2.0 / 55.0
};


// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

double clamp(
    double value,
    double lower,
    double upper
)
{
    return std::max(
        lower,
        std::min(value, upper)
    );
}

}  // namespace


RK45Result integrate_rk45(
    const DynamicsFunction& dynamics,
    double t0,
    double tf,
    const State& initial_state,
    const RK45Options& options
)
{
    if (!dynamics) {
        throw std::invalid_argument(
            "RK45 requires a valid dynamics function."
        );
    }

    if (initial_state.empty()) {
        throw std::invalid_argument(
            "RK45 initial state cannot be empty."
        );
    }

    if (tf < t0) {
        throw std::invalid_argument(
            "RK45 currently requires tf >= t0."
        );
    }

    if (options.h0 <= 0.0) {
        throw std::invalid_argument(
            "RK45 h0 must be greater than zero."
        );
    }

    if (options.h_min <= 0.0) {
        throw std::invalid_argument(
            "RK45 h_min must be greater than zero."
        );
    }

    if (options.h_max < options.h_min) {
        throw std::invalid_argument(
            "RK45 requires h_max >= h_min."
        );
    }

    if (options.rtol < 0.0 || options.atol < 0.0) {
        throw std::invalid_argument(
            "RK45 tolerances must be non-negative."
        );
    }

    if (options.max_iter == 0) {
        throw std::invalid_argument(
            "RK45 max_iter must be greater than zero."
        );
    }

    // -------------------------------------------------------------------------
    // Initialize result
    // -------------------------------------------------------------------------

    RK45Result result;

    result.success = false;
    result.status = -1;
    result.message = "Integration failed.";

    result.njev = 0;
    result.nlu = 0;

    // -------------------------------------------------------------------------
    // Initial state
    // -------------------------------------------------------------------------

    const std::size_t n = initial_state.size();

    State state = initial_state;

    double t = t0;

    double h = clamp(
        options.h0,
        options.h_min,
        options.h_max
    );

    // -------------------------------------------------------------------------
    // Store initial condition
    // -------------------------------------------------------------------------

    result.t.push_back(t);
    result.y.push_back(state);

    // -------------------------------------------------------------------------
    // Special case: zero-length integration
    // -------------------------------------------------------------------------

    if (t0 == tf) {

        result.state = state;

        result.success = true;
        result.status = 0;
        result.message = "The solver successfully reached the end.";

        return result;
    }

    // -------------------------------------------------------------------------
    // Main integration loop
    // -------------------------------------------------------------------------

    std::size_t iter_count = 0;

    while (t < tf && iter_count < options.max_iter) {

        // ---------------------------------------------------------------------
        // Prevent the final step from exceeding tf
        // ---------------------------------------------------------------------

        if (t + h > tf) {
            h = tf - t;
        }

        // ---------------------------------------------------------------------
        // Runge-Kutta stages
        // ---------------------------------------------------------------------

        State k[6];

        for (auto& stage : k) {
            stage.resize(n);
        }

        // ---------------------------------------------------------------------
        // Stage 1
        // ---------------------------------------------------------------------

        dynamics(
            t,
            state,
            k[0]
        );

        result.nfev++;

        // ---------------------------------------------------------------------
        // Remaining stages
        // ---------------------------------------------------------------------

        for (int i = 1; i < 6; ++i) {

            State stage_state = state;

            for (int j = 0; j < i; ++j) {

                for (std::size_t index = 0; index < n; ++index) {

                    stage_state[index] +=
                        h
                        * A[i][j]
                        * k[j][index];
                }
            }

            dynamics(
                t + C[i] * h,
                stage_state,
                k[i]
            );

            result.nfev++;
        }

        // ---------------------------------------------------------------------
        // Compute fourth- and fifth-order solutions
        // ---------------------------------------------------------------------

        State y4 = state;
        State y5 = state;

        for (std::size_t index = 0; index < n; ++index) {

            double increment4 = 0.0;
            double increment5 = 0.0;

            for (int i = 0; i < 6; ++i) {

                increment4 +=
                    B[i] * k[i][index];

                increment5 +=
                    B_STAR[i] * k[i][index];
            }

            y4[index] += h * increment4;
            y5[index] += h * increment5;
        }

        // ---------------------------------------------------------------------
        // Fixed-step mode
        // ---------------------------------------------------------------------

        if (!options.adaptive) {

            state = y5;

            t += h;

            result.accepted_steps++;

            result.t.push_back(t);
            result.y.push_back(state);

            h = clamp(
                options.h0,
                options.h_min,
                options.h_max
            );

            ++iter_count;

            continue;
        }

        // ---------------------------------------------------------------------
        // Adaptive error estimation
        // ---------------------------------------------------------------------

        double error_sum = 0.0;

        for (std::size_t index = 0; index < n; ++index) {

            const double scale =
                options.atol
                + options.rtol
                * std::max(
                    std::abs(state[index]),
                    std::abs(y5[index])
                );

            if (scale == 0.0) {
                continue;
            }

            const double normalized_error =
                (y5[index] - y4[index])
                / scale;

            error_sum +=
                normalized_error
                * normalized_error;
        }

        const double error =
            std::sqrt(
                error_sum
                / static_cast<double>(n)
            );

        // ---------------------------------------------------------------------
        // Accept step
        // ---------------------------------------------------------------------

        if (error <= 1.0) {

            state = y5;

            t += h;

            result.accepted_steps++;

            result.t.push_back(t);
            result.y.push_back(state);

            double factor;

            if (error == 0.0) {

                factor = 2.0;
            }
            else {

                factor =
                    SAFETY
                    * std::pow(
                        error,
                        -1.0 / P
                    );
            }

            factor = clamp(
                factor,
                0.2,
                5.0
            );

            h *= factor;
        }

        // ---------------------------------------------------------------------
        // Reject step
        // ---------------------------------------------------------------------

        else {

            result.rejected_steps++;

            double factor =
                SAFETY
                * std::pow(
                    error,
                    -1.0 / P
                );

            factor = clamp(
                factor,
                0.1,
                0.5
            );

            h *= factor;

            h = std::max(
                h,
                options.h_min
            );

            if (
                h <= options.h_min
                && error > 1.0
            ) {

                result.state = state;

                result.success = false;
                result.status = -1;

                result.message =
                    "RK45 minimum step reached "
                    "without satisfying tolerance.";

                return result;
            }
        }

        // ---------------------------------------------------------------------
        // Clamp next step
        // ---------------------------------------------------------------------

        h = clamp(
            h,
            options.h_min,
            options.h_max
        );

        // ---------------------------------------------------------------------
        // Optional verbose output
        // ---------------------------------------------------------------------

        if (options.verbose) {
            // Intentionally left empty for now.
            //
            // Logging can be added later without affecting the
            // numerical implementation.
        }

        ++iter_count;
    }

    // -------------------------------------------------------------------------
    // Final result
    // -------------------------------------------------------------------------

    result.state = state;

    if (t >= tf) {

        result.success = true;
        result.status = 0;

        result.message =
            "The solver successfully reached the end.";
    }
    else if (iter_count >= options.max_iter) {

        result.success = false;
        result.status = -1;

        result.message =
            "Maximum number of iterations exceeded.";
    }

    return result;
}

}  // namespace argos