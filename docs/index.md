# ARGOS Toolkit

**A mission-oriented simulation framework for spacecraft GNC and astrodynamics research.**

ARGOS provides a modular environment for building and executing spacecraft simulation scenarios. Rather than defining a simulation as a fixed collection of models, ARGOS allows spacecraft configuration, GNC algorithms, propagation models, and mission behavior to evolve throughout the simulation.

This documentation is organized around how a user works with ARGOS: understanding the simulation architecture, configuring spacecraft and their components, defining mission behavior, running simulations, and analyzing the resulting data.

---

## Getting Started

If you are new to ARGOS, the recommended path is:

1. **Understand the simulation architecture**
2. **Create or configure a spacecraft**
3. **Configure its sensors, navigation, guidance, control, and actuators**
4. **Define the spacecraft mission phases**
5. **Run the simulation**
6. **Inspect and analyze the simulation history**

The [`examples/`](https://github.com/rodandres/ARGOS-Toolkit/tree/dev/examples) directory contains complete executable examples that complement this documentation.

---

## How ARGOS Works

At the highest level, an ARGOS simulation is composed of an environment and one or more spacecraft.

```text
                        Simulation
                            │
             ┌──────────────┴──────────────┐
             │                             │
        Environment                  Spacecraft
                                           │
                  ┌────────────────────────┼────────────────────────┐
                  │                        │                        │
                State                  Mission                  GNC System
                                           │                        │
                                      MissionPhase                  │
                                           │                        │
                                      Transitions                   │
                                                                    │
                                  ┌─────────┼─────────┐
                                  │         │         │
                               Sensors  Navigation  Guidance
                                                       │
                                                       ▼
                                                   Control
                                                       │
                                                       ▼
                                              Control Allocation
                                                       │
                                                       ▼
                                                   Actuators
                                                       │
                                                       ▼
                                                 Propagation
```

The important concept is that these components are not required to be tightly coupled to one another. ARGOS provides interfaces and data structures that allow different implementations to be assembled into different simulation configurations.

---

## Core Concepts

The following concepts form the foundation of the ARGOS simulation architecture.

### Simulation

The `Simulation` object provides the global execution context.

It coordinates:

- Simulation time
- Environment
- Spacecraft
- Execution of the simulation loop
- Simulation history

A simulation may contain multiple spacecraft with independent configurations.

### Spacecraft

A `Spacecraft` represents an individual simulated vehicle.

A spacecraft contains the state and the components required to simulate its behavior, including:

- Sensors
- Navigation
- Guidance
- Controllers
- Control allocation
- Actuators
- Propagation models
- Mission management
- Fault management

Different spacecraft can therefore use different configurations within the same simulation.

### Mission Phases

A spacecraft does not necessarily have a single configuration throughout an entire mission.

ARGOS represents mission behavior using `MissionPhase` objects. A phase defines the configuration that is active during a particular portion of the mission.

A transition condition can then activate another phase:

```text
             Mission Phase A
                    │
                    │ Transition Condition
                    ▼
             Mission Phase B
                    │
                    │ Transition Condition
                    ▼
             Mission Phase C
```

This allows the same spacecraft to operate with different GNC and propagation configurations as the mission progresses.

### GNC Components

ARGOS separates the main GNC functions into independent components:

```text
Sensors
   │
   ▼
Navigation
   │
   ▼
Guidance
   │
   ▼
Controller
   │
   ▼
Control Allocation
   │
   ▼
Actuators
   │
   ▼
Spacecraft Dynamics
```

The interfaces allow users to implement or replace individual components without restructuring the entire simulation.

### Faults

Fault injection is integrated into the spacecraft architecture through the `FaultManager`.

Faults can be associated with spacecraft components and activated according to user-defined conditions. The affected components are responsible for implementing the corresponding change in behavior.

FDIR functionality is an ongoing area of development.

### Simulation History

ARGOS separates simulation execution from post-processing through the `SimulationHistory` system.

Simulation data is recorded independently for each spacecraft and progressively persisted to disk in fixed-size chunks. The resulting data can subsequently be used for:

- Visualization
- Numerical analysis
- Performance assessment
- Custom post-processing
- External data-processing workflows

See the API Reference for the `SimulationHistory` implementation and the examples for practical usage.

---

## A Typical ARGOS Workflow

A typical workflow can be understood as:

```text
Define Environment
        │
        ▼
Configure Spacecraft
        │
        ├── State
        ├── Sensors
        ├── Navigation
        ├── Guidance
        ├── Control
        ├── Control Allocation
        ├── Actuators
        └── Propagation
        │
        ▼
Define Mission Phases
        │
        ▼
Define Transition Conditions
        │
        ▼
Run Simulation
        │
        ▼
Record Simulation History
        │
        ▼
Analyze / Visualize Results
```

The individual components can then be replaced or extended depending on the experiment being investigated.

---

## Examples

The examples are the best starting point for seeing how the architecture is assembled into complete simulations.

### Python Examples

- [Example 1 — Attitude Control](https://github.com/rodandres/ARGOS-Toolkit/blob/dev/examples/py/example_1.py)
- [Example 2 — Multi-Spacecraft Translational Simulation](https://github.com/rodandres/ARGOS-Toolkit/blob/dev/examples/py/example_2.py)
- [Example 3 — CR3BP Trajectory Propagation](https://github.com/rodandres/ARGOS-Toolkit/blob/dev/examples/py/example_3.py)
- [Example 4 — Mission Phase Transition and Attitude Control](https://github.com/rodandres/ARGOS-Toolkit/blob/dev/examples/py/example_4.py)
- [Continuation of Lyapunov and Halo Orbit Families](https://github.com/rodandres/ARGOS-Toolkit/blob/dev/examples/py/Continuation%20of%20Lyapunov%20and%20Halo%20Orbit%20Families.py)

Jupyter notebooks are also available in [`examples/ipynb/`](https://github.com/rodandres/ARGOS-Toolkit/tree/dev/examples/ipynb).

---

## API Reference

The API reference is generated directly from the ARGOS source code and its NumPy-style docstrings.

It is organized by package:

- [Actuators](api/actuators.md)
- [Cislunar Astrodynamics](api/cislunar_astrodynamics.md)
- [Controllers](api/controllers.md)
- [Core](api/core.md)
- [Enviroments](api/enviroments.md)
- [Faults](api/faults.md)
- [General](api/general.md)
- [Guidance](api/guidance.md)
- [Navigation](api/navigation.md)
- [Propagators](api/propagators.md)
- [Sensors](api/sensors.md)
- [Solvers](api/solvers.md)

The API reference describes the available classes, functions, parameters, data structures, and interfaces.

---

## Cislunar Astrodynamics

ARGOS also provides a dedicated set of tools for studying trajectories and dynamics in the Circular Restricted Three-Body Problem (CR3BP).

These tools include:

- CR3BP dynamics
- Lagrange points
- Jacobi constant
- Effective potential
- Linearized dynamics
- State Transition Matrix propagation
- Eigenanalysis
- Lyapunov orbits
- Halo orbits
- Orbit-family continuation

These capabilities can be used independently for astrodynamics studies or as part of broader spacecraft simulation scenarios.

The [Lyapunov and Halo orbit continuation example](https://github.com/rodandres/ARGOS-Toolkit/blob/dev/examples/py/Continuation%20of%20Lyapunov%20and%20Halo%20Orbit%20Families.py) provides a complete example of this workflow.

---

## Development and Validation

ARGOS is an actively developed research framework.

The test suite is intended to validate the implementation at multiple levels, including:

- Unit-level behavior
- Integration between simulation components
- Package structure and imports
- Execution of representative examples

Run the tests locally with:

```bash
pytest -q
```

The API documentation can be generated and served locally with:

```bash
python script.py
mkdocs serve
```

---

## Where to Go Next

Depending on what you want to do with ARGOS:

| Goal | Start here |
|---|---|
| Understand the architecture | [Core](api/core.md) |
| Build a spacecraft | [Core](api/core.md) |
| Configure sensors | [Sensors](api/sensors.md) |
| Implement navigation | [Navigation](api/navigation.md) |
| Implement guidance | [Guidance](api/guidance.md) |
| Implement control | [Controllers](api/controllers.md) |
| Configure actuators | [Actuators](api/actuators.md) |
| Select propagation models | [Propagators](api/propagators.md) |
| Model faults | [Faults](api/faults.md) |
| Study CR3BP dynamics | [Cislunar Astrodynamics](api/cislunar_astrodynamics.md) |
| See complete simulations | [Examples](https://github.com/rodandres/ARGOS-Toolkit/tree/dev/examples) |