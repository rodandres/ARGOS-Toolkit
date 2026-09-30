cmake_minimum_required(VERSION 3.18)

project(argos_cpp LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND python -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_cpp
    cpp/argos/bindings.cpp
    cpp/argos/solvers/rk45.cpp
    cpp/argos/propagators/dynamic_models.cpp
    cpp/argos/propagators/dynamics_factory.cpp
    cpp/argos/propagators/propagator.cpp
)

target_compile_features(_cpp PRIVATE cxx_std_17)

install(
    TARGETS _cpp
    DESTINATION argos
)