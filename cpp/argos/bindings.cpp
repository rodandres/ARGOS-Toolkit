#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "propagators/propagator.hpp"
#include "solvers/rk45.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;


namespace {


argos::State numpy_to_state(
    const py::array_t<double>& array
)
{
    auto buffer = array.request();

    if (buffer.ndim != 1) {
        throw std::invalid_argument(
            "State must be a one-dimensional NumPy array."
        );
    }

    const auto* data =
        static_cast<const double*>(buffer.ptr);

    return argos::State(
        data,
        data + buffer.shape[0]
    );
}


argos::Vector3 numpy_to_vector3(
    const py::array_t<double>& array
)
{
    auto buffer = array.request();

    if (buffer.ndim != 1 || buffer.shape[0] != 3) {
        throw std::invalid_argument(
            "Vector3 input must be a one-dimensional NumPy array "
            "with exactly 3 elements."
        );
    }

    const auto* data =
        static_cast<const double*>(buffer.ptr);

    return {
        data[0],
        data[1],
        data[2]
    };
}

argos::Matrix3 numpy_to_matrix3(
    const py::array_t<double>& array
)
{
    auto buffer = array.request();

    if (
        buffer.ndim != 2
        || buffer.shape[0] != 3
        || buffer.shape[1] != 3
    ) {
        throw std::invalid_argument(
            "Matrix3 input must be a two-dimensional NumPy array "
            "with shape (3, 3)."
        );
    }

    const auto* data =
        static_cast<const double*>(buffer.ptr);

    return {{
        {data[0], data[1], data[2]},
        {data[3], data[4], data[5]},
        {data[6], data[7], data[8]}
    }};
}

py::array_t<double> vector_to_numpy(
    const std::vector<double>& values
)
{
    py::array_t<double> result(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(values.size())
        }
    );

    auto mutable_buffer = result.mutable_unchecked<1>();

    for (py::ssize_t i = 0; i < mutable_buffer.shape(0); ++i) {
        mutable_buffer(i) = values[static_cast<std::size_t>(i)];
    }

    return result;
}


py::array_t<double> state_to_numpy(
    const argos::State& state
)
{
    return vector_to_numpy(state);
}


py::list events_to_python(
    const std::vector<std::vector<double>>& events
)
{
    py::list result;

    for (const auto& event : events) {
        result.append(vector_to_numpy(event));
    }

    return result;
}


py::list state_events_to_python(
    const std::vector<std::vector<std::vector<double>>>& events
)
{
    py::list result;

    for (const auto& event_group : events) {

        py::list group;

        for (const auto& state : event_group) {
            group.append(state_to_numpy(state));
        }

        result.append(group);
    }

    return result;
}


py::array_t<double> history_to_numpy(
    const std::vector<argos::State>& history
)
{
    if (history.empty()) {
        return py::array_t<double>(
            py::array::ShapeContainer{
                static_cast<py::ssize_t>(0),
                static_cast<py::ssize_t>(0)
            }
        );
    }

    const std::size_t n_times =
        history.size();

    const std::size_t n_states =
        history.front().size();

    py::array_t<double> result(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(n_states),
            static_cast<py::ssize_t>(n_times)
        }
    );

    auto mutable_buffer = result.mutable_unchecked<2>();

    for (std::size_t i = 0; i < n_times; ++i) {

        if (history[i].size() != n_states) {
            throw std::runtime_error(
                "Inconsistent state dimensions in RK45 history."
            );
        }

        for (std::size_t j = 0; j < n_states; ++j) {
            mutable_buffer(
                static_cast<py::ssize_t>(j),
                static_cast<py::ssize_t>(i)
            ) = history[i][j];
        }
    }

    return result;
}


py::dict rk45_result_to_python(
    const argos::RK45Result& result
)
{
    py::dict output;

    // -------------------------------------------------------------------------
    // Time history
    // -------------------------------------------------------------------------

    output["t"] =
        vector_to_numpy(result.t);


    // -------------------------------------------------------------------------
    // State history
    //
    // C++ stores:
    //
    //     y = [state(t0), state(t1), ...]
    //
    // Python scipy-style OdeResult expects:
    //
    //     y.shape == (n_states, n_times)
    // -------------------------------------------------------------------------

    output["y"] =
        history_to_numpy(result.y);


    // -------------------------------------------------------------------------
    // Final state
    // -------------------------------------------------------------------------

    output["state"] =
        state_to_numpy(result.state);


    // -------------------------------------------------------------------------
    // Solver status
    // -------------------------------------------------------------------------

    output["success"] =
        result.success;

    output["status"] =
        result.status;

    output["message"] =
        result.message;


    // -------------------------------------------------------------------------
    // Solver statistics
    // -------------------------------------------------------------------------

    output["nfev"] =
        result.nfev;

    output["njev"] =
        result.njev;

    output["nlu"] =
        result.nlu;


    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    output["t_events"] =
        events_to_python(result.t_events);

    output["y_events"] =
        state_events_to_python(result.y_events);


    // -------------------------------------------------------------------------
    // Native RK45 statistics
    // -------------------------------------------------------------------------

    output["accepted_steps"] =
        result.accepted_steps;

    output["rejected_steps"] =
        result.rejected_steps;


    return output;
}


}  // namespace


PYBIND11_MODULE(_cpp, m)
{
    m.doc() =
        "ARGOS native C++ simulation backend";

    m.def(
        "propagate_translational",
        [](
            py::array_t<double> state,
            double t0,
            double tf,
            double mass,
            py::array_t<double> applied_force,
            py::array_t<double> disturbance_force,
            const std::string& dynamics_model,
            const std::string& orbital_model,
            double mu,
            double length_factor,
            double time_factor,
            double h0,
            double h_min,
            double h_max,
            double rtol,
            double atol,
            bool adaptive,
            std::size_t max_iter,
            bool verbose
        )
        {        
            // ------------------------------------------------------------
            // Convert Python types to C++ types
            // ------------------------------------------------------------

            argos::State cpp_state =
                numpy_to_state(state);

            argos::Vector3 cpp_applied_force =
                numpy_to_vector3(applied_force);

            argos::Vector3 cpp_disturbance_force =
                numpy_to_vector3(disturbance_force);

            // ------------------------------------------------------------
            // Configure RK45
            // ------------------------------------------------------------

            argos::RK45Options options;

            options.h0 = h0;
            options.h_min = h_min;
            options.h_max = h_max;
            options.rtol = rtol;
            options.atol = atol;
            options.adaptive = adaptive;
            options.max_iter = max_iter;
            options.verbose = verbose;

            // ------------------------------------------------------------
            // Run C++ translational propagation
            // ------------------------------------------------------------

            const auto result =
                argos::propagate_translational(
                    cpp_state,
                    t0,
                    tf,
                    mass,
                    cpp_applied_force,
                    cpp_disturbance_force,
                    dynamics_model,
                    orbital_model,
                    mu,
                    length_factor,
                    time_factor,
                    options
                );        

            // ------------------------------------------------------------
            // Convert result back to Python
            // ------------------------------------------------------------

            return rk45_result_to_python(result);
        },

        // ------------------------------------------------------------
        // Python argument names
        // ------------------------------------------------------------

        py::arg("state"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("mass"),
        py::arg("applied_force"),
        py::arg("disturbance_force"),
        py::arg("dynamics_model"),
        py::arg("orbital_model"),
        py::arg("mu"),
        py::arg("length_factor"),
        py::arg("time_factor"),
        py::arg("h0"),
        py::arg("h_min"),
        py::arg("h_max"),
        py::arg("rtol"),
        py::arg("atol"),
        py::arg("adaptive"),
        py::arg("max_iter"),
        py::arg("verbose")
    );    

    m.def(
        "propagate_rotational",
        [](
            py::array_t<double> state,
            double t0,
            double tf,
            py::array_t<double> inertia_matrix,
            py::array_t<double> inverse_inertia_matrix,
            py::array_t<double> applied_torque,
            py::array_t<double> disturbance_torque,
            const std::string& dynamics_model,
            double h0,
            double h_min,
            double h_max,
            double rtol,
            double atol,
            bool adaptive,
            std::size_t max_iter,
            bool verbose
        )
        {
            // ------------------------------------------------------------
            // Convert Python types to C++ types
            // ------------------------------------------------------------

            argos::State cpp_state =
                numpy_to_state(state);

            argos::Matrix3 cpp_inertia_matrix =
                numpy_to_matrix3(inertia_matrix);

            argos::Matrix3 cpp_inverse_inertia_matrix =
                numpy_to_matrix3(inverse_inertia_matrix);

            argos::Vector3 cpp_applied_torque =
                numpy_to_vector3(applied_torque);

            argos::Vector3 cpp_disturbance_torque =
                numpy_to_vector3(disturbance_torque);

            // ------------------------------------------------------------
            // Configure RK45
            // ------------------------------------------------------------

            argos::RK45Options options;

            options.h0 = h0;
            options.h_min = h_min;
            options.h_max = h_max;
            options.rtol = rtol;
            options.atol = atol;
            options.adaptive = adaptive;
            options.max_iter = max_iter;
            options.verbose = verbose;

            // ------------------------------------------------------------
            // Run C++ rotational propagation
            // ------------------------------------------------------------

            const auto result =
                argos::propagate_rotational(
                    cpp_state,
                    t0,
                    tf,
                    cpp_inertia_matrix,
                    cpp_inverse_inertia_matrix,
                    cpp_applied_torque,
                    cpp_disturbance_torque,
                    dynamics_model,
                    options
                );

            // ------------------------------------------------------------
            // Convert result back to Python
            // ------------------------------------------------------------

            return rk45_result_to_python(result);
        },

        // ------------------------------------------------------------
        // Python argument names
        // ------------------------------------------------------------

        py::arg("state"),
        py::arg("t0"),
        py::arg("tf"),
        py::arg("inertia_matrix"),
        py::arg("inverse_inertia_matrix"),
        py::arg("applied_torque"),
        py::arg("disturbance_torque"),
        py::arg("dynamics_model"),
        py::arg("h0"),
        py::arg("h_min"),
        py::arg("h_max"),
        py::arg("rtol"),
        py::arg("atol"),
        py::arg("adaptive"),
        py::arg("max_iter"),
        py::arg("verbose")
    );

    m.def(
        "evaluate_translational_dynamics",
        [](
            py::array_t<double> state,
            double t,
            double mass,
            py::array_t<double> applied_force,
            py::array_t<double> disturbance_force,
            const std::string& dynamics_model,
            const std::string& orbital_model,
            double mu,
            double length_factor,
            double time_factor
        )
        {
            argos::State cpp_state = numpy_to_state(state);
            argos::Vector3 cpp_applied_force =
                numpy_to_vector3(applied_force);
            argos::Vector3 cpp_disturbance_force =
                numpy_to_vector3(disturbance_force);

            argos::State derivative =
                argos::evaluate_translational_dynamics(
                    cpp_state,
                    t,
                    mass,
                    cpp_applied_force,
                    cpp_disturbance_force,
                    dynamics_model,
                    orbital_model,
                    mu,
                    length_factor,
                    time_factor
                );

            return state_to_numpy(derivative);
        },
        py::arg("state"),
        py::arg("t"),
        py::arg("mass"),
        py::arg("applied_force"),
        py::arg("disturbance_force"),
        py::arg("dynamics_model"),
        py::arg("orbital_model"),
        py::arg("mu"),
        py::arg("length_factor"),
        py::arg("time_factor")
    );

    m.def(
        "evaluate_rotational_dynamics",
        [](
            py::array_t<double> state,
            double t,
            py::array_t<double> inertia_matrix,
            py::array_t<double> inverse_inertia_matrix,
            py::array_t<double> applied_torque,
            py::array_t<double> disturbance_torque,
            const std::string& dynamics_model
        )
        {
            argos::State cpp_state =
                numpy_to_state(state);

            argos::Matrix3 cpp_inertia_matrix =
                numpy_to_matrix3(inertia_matrix);

            argos::Matrix3 cpp_inverse_inertia_matrix =
                numpy_to_matrix3(inverse_inertia_matrix);

            argos::Vector3 cpp_applied_torque =
                numpy_to_vector3(applied_torque);

            argos::Vector3 cpp_disturbance_torque =
                numpy_to_vector3(disturbance_torque);

            argos::State derivative =
                argos::evaluate_rotational_dynamics(
                    cpp_state,
                    t,
                    cpp_inertia_matrix,
                    cpp_inverse_inertia_matrix,
                    cpp_applied_torque,
                    cpp_disturbance_torque,
                    dynamics_model
                );

            return state_to_numpy(derivative);
        },

        py::arg("state"),
        py::arg("t"),
        py::arg("inertia_matrix"),
        py::arg("inverse_inertia_matrix"),
        py::arg("applied_torque"),
        py::arg("disturbance_torque"),
        py::arg("dynamics_model")
    ); 
}