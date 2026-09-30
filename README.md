# ARGOS Toolkit

**ARGOS Toolkit** is a modular Python framework for **spacecraft simulation, Guidance Navigation and Control (GNC) research, and astrodynamics analysis**.

The toolkit is designed around a **mission-oriented simulation architecture**, where spacecraft, physical components, GNC algorithms, propagation models, mission phases, and operational transitions can be configured and executed within a common simulation environment.

ARGOS is being developed as a research-oriented platform for studying spacecraft behavior and GNC architectures, with a particular interest in **multi-spacecraft and cislunar applications**.

---
![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)
![Python](https://img.shields.io/badge/Python-3.14-blue)
![Project](https://img.shields.io/badge/project-Research-green)
![Domain](https://img.shields.io/badge/domain-RPOD-informational)
![Domain](https://img.shields.io/badge/domain-Spacecraft%20GNC-informational)
<!-- Add or remove badges depending on the project -->
---

## Overview

Spacecraft GNC simulations require the integration of several disciplines and computational models:

- Spacecraft dynamics
- Sensors and measurement models
- Navigation
- Guidance
- Control
- Control allocation
- Actuators
- Numerical propagation
- Mission logic
- Fault injection
- Astrodynamics
- Data analysis and visualization

ARGOS provides a common simulation architecture in which these elements can be developed independently and combined into executable spacecraft scenarios.

A simulation can contain one or more spacecraft, each with its own physical and computational configuration, while sharing a common simulation environment.

## Why ARGOS?

ARGOS is intended to reduce the amount of infrastructure required to investigate spacecraft GNC concepts.

Instead of implementing every simulation from scratch, a user can assemble a spacecraft from reusable components and focus on the problem being investigated. This is emphasize through its central concept of Mission-Oriented Simulation where the simulation configuration is not necessarily fixed for the entire mission.

Therefore, a spacecraft can be represented through a sequence of mission phases:

Mission Phase 1
    │
    ├── Navigation
    ├── Guidance
    ├── Control
    ├── Control Allocation
    └── Propagation
    │
    │ Transition Condition
    ▼
Mission Phase 2
    │
    ├── Navigation
    ├── Guidance
    ├── Control
    ├── Control Allocation
    └── Propagation
    │
    │ Transition Condition
    ▼
Mission Phase 3

## Architecture
The architecture emphasizes:

- **Modularity —** GNC and simulation components are separated into well-defined interfaces.
- **Composability —** Different sensors, navigation laws, guidance algorithms, controllers, actuators, and propagators can be combined.
- **Multi-spacecraft simulation —** Multiple spacecraft can coexist within the same simulation and exchange relevant state information.
- **Mission-level behavior —** Mission phases allow the active GNC and dynamics configuration to change throughout a simulation.
- **Research flexibility —** Custom algorithms can be introduced through extensible base interfaces.
- **Astrodynamics integration —** Spacecraft simulation capabilities are complemented by dedicated cislunar and CR3BP analysis tools.
- **Reproducible analysis —** Simulation results are stored in structured history objects that can be analyzed and visualized after execution.

This is organized around a small number of fundamental concepts.
### Simulation
`Simulation` provides the global simulation context.
It manages:
- Simulation time
- Environment
- Multiple spacecraft
- Global execution
- Simulation results

> A simulation can contain multiple independent spacecraft:

### Spacecraft
A `Spacecraft` represents an individual simulated vehicle.

It maintains its own:
- Translational state
- Rotational state
- Sensors
- Actuators
- Mission configuration
- GNC configuration
- Propagation models
- Fault configuration

> Multiple spacecraft can therefore participate in the same simulation while maintaining independent configurations.

### Mission Manager
`MissionManager` controls the active mission phase of a spacecraft.
This provides a mechanism for representing phase-dependent spacecraft behavior without coupling the mission logic to individual GNC implementations.

> A mission can be defined by creating MissionPhase objects and associating transition conditions with them.

### Modular GNC Interfaces
ARGOS exposes base interfaces for major simulation and GNC components.
These include:
- Sensors
- Navigation
- Guidance
- Controllers
- Actuators
- Propagators
- Environments

### Fault Injection
The `FaultManager` coordinates user-defined fault events, while affected components implement the mechanisms required to modify their behavior.

This architecture distinguishes between:
- Nominal sensor error models
- Injected component faults
- Fault activation conditions
- Fault deactivation conditions
  
> The architecture allows users to replace individual implementations while preserving the surrounding simulation configuration.

## Cislunar Astrodynamics

In addition to the spacecraft simulation framework, ARGOS contains dedicated tools for analysis in the Circular Restricted Three-Body Problem.

Current capabilities include:

- CR3BP equations of motion
- Jacobi constant
- Effective potential
- Lagrange points
- CR3BP Jacobian
- State Transition Matrix propagation
- Linearized dynamics
- Eigenanalysis
- Lyapunov orbit computation
- Halo orbit generation
- Orbit-family continuation

These capabilities provide a foundation for studying spacecraft trajectories in systems where two-body approximations are insufficient, particularly in cislunar environments.

A more detail example of usage can be seen in:

```text
examples/xx/Continuation of Lyapunov and Halo Orbit Families.xx
``` 

Where a demonstration of the cislunar astrodynamics capabilities of ARGOS is implemented through the computation and continuation of periodic orbit families in the Earth-Moon CR3BP.

## MBSE Direction
One of the long-term objectives of ARGOS is to investigate the connection between Model-Based Systems Engineering (MBSE) and executable spacecraft GNC simulation.

The mission-oriented architecture provides explicit abstractions for:
- Mission context
- Mission phases
- Spacecraft
- System functions
- Sensors
- Actuators
- GNC components
- Propagation models
- Operational configurations

These abstractions provide a semantic basis for connecting system-level descriptions with executable simulation configurations.

> The current research direction investigates the mapping between MBSE models and ARGOS configurations, particularly using Arcadia/Capella models.

## Current Capabilities

| Area          | Capability                            | Status      |
| ------------- | ------------------------------------- | ----------- |
| Simulation    | Modular simulation engine             | Implemented |
| Simulation    | Multiple spacecraft                   | Implemented |
| Simulation    | Independent spacecraft configurations | Implemented |
| Mission       | Mission phases                        | Implemented |
| Mission       | Conditional phase transitions         | Implemented |
| Dynamics      | Translational dynamics                | Implemented |
| Dynamics      | Rotational rigid-body dynamics        | Implemented |
| Propagation   | Relative two-body propagation         | Implemented |
| Propagation   | CR3BP propagation                     | Implemented |
| Propagation   | Native adaptive RK45 solver           | Implemented |
| Sensors       | Modular sensor architecture           | Implemented |
| Sensors       | Absolute state measurements           | Implemented |
| Sensors       | Measurement error models              | Implemented |
| Navigation    | Ideal navigation                      | Implemented |
| Navigation    | Custom navigation interfaces          | Implemented |
| Guidance      | Constant-reference guidance           | Implemented |
| Guidance      | Custom guidance interfaces            | Implemented |
| Control       | PD attitude control                   | Implemented |
| Control       | Custom controller interfaces          | Implemented |
| Control       | Control allocation                    | Implemented |
| Actuation     | RCS thruster model                    | Implemented |
| Actuation     | PWM / bang-bang behavior              | Implemented |
| Environment   | Environment abstraction               | Implemented |
| Faults        | Fault manager and fault events        | Implemented |
| Faults        | Sensor fault injection                | Implemented |
| Visualization | State visualization                   | Implemented |
| Visualization | GNC visualization                     | Implemented |
| Visualization | Trajectory visualization              | Implemented |
| Astrodynamics | CR3BP analysis                        | Implemented |
| Astrodynamics | Lagrange point computation            | Implemented |
| Astrodynamics | Linearized CR3BP analysis             | Implemented |
| Astrodynamics | Eigenanalysis                         | Implemented |
| Astrodynamics | Lyapunov orbit computation            | Implemented |
| Astrodynamics | Halo orbit generation                 | Implemented |
| Astrodynamics | Orbit-family continuation             | Implemented |


## Examples
ARGOS includes Python examples and Jupyter notebooks demonstrating different aspects of the toolkit.

### Example 1 — Attitude GNC
Introduces the complete GNC execution chain:
```text
Sensor
  ↓
Navigation
  ↓
Guidance
  ↓
Controller
  ↓
Control Allocation
  ↓
RCS
  ↓
Rotational Dynamics
```

### Example 2 — Multi-Spacecraft Simulation
Demonstrates multiple spacecraft within the same simulation environment, including spacecraft-to-spacecraft state relationships.

This provides the basis for scenarios where one spacecraft needs information about another spacecraft.

### Example 3 — CR3BP Propagation
Demonstrates translational propagation using the Circular Restricted Three-Body Problem (CR3BP), including the Earth-Moon system.

This example introduces the connection between the spacecraft simulation framework and the cislunar astrodynamics tools.

### Example 4 — Mission Phases
Demonstrates how a spacecraft can change its active simulation and GNC configuration during a mission.

The example illustrates:
``` text
Mission Phase
      ↓
Transition Condition
      ↓
New Mission Phase
      ↓
New GNC / Dynamics Configuration
```

### Simulation Results
ARGOS records simulation results through the `SimulationHistory` system. History is maintained independently for each spacecraft and is progressively written to disk in fixed-size chunks, allowing long simulations to be recorded without keeping the complete history in memory.

For each spacecraft, the recorded history can include:

- Simulation time and simulation tick
- True translational state:
  - Position
  - Velocity
  - Acceleration
- True rotational state:
  - Attitude quaternion
  - Angular velocity
  - Angular acceleration
- Estimated spacecraft state produced by the navigation system
- Estimated target state, when applicable
- Guidance references:
  - Position
  - Velocity
  - Acceleration
  - Attitude
  - Angular velocity
  - Angular acceleration
- Control outputs:
  - Force
  - Torque
- Forces and torques actually exerted on the spacecraft
- Mission phase transition events
- Target spacecraft information

This allows the simulation to be separated from the subsequent analysis and visualization workflow.

Conceptually:
```text
Simulation
     ↓
SimulationHistory
     ↓
 ┌──────────────────────┐
 ↓          ↓           ↓
Plots  Analysis  Custom Processing
```

## Installation

> ARGOS currently requires Python 3.14 or newer.

Clone the repository:

```bash
git clone https://github.com/rodandres/ARGOS-Toolkit.git
cd ARGOS-Toolkit
```

Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install ARGOS in editable mode:
```bash
python -m pip install -e .
```

For development and testing:
```bash
python -m pip install -e ".[dev]"
```

Verify the installation:
```bash
python -c "import argos; print(argos.__file__)"
```

## Quick Start

The simplest way to become familiar with ARGOS is to run the provided examples.

The examples are located under:
```text
examples/
```

and are provided both as Python scripts and Jupyter notebooks.

## Project Structure
ARGOS-Toolkit/
│
├── src/
│   └── argos/
│       ├── actuators/
│       ├── cislunar_astrodynamics/
│       ├── controllers/
│       ├── core/
│       ├── enviroments/
│       ├── faults/
│       ├── frames/
│       ├── general/
│       ├── guidance/
│       ├── navigation/
│       ├── propagators/
│       ├── sensors/
│       ├── solvers/
│       └── visualization/
│
├── examples/
│   ├── ipynb/
│   └── py/
│
├── tests/
│
├── docs/
│   ├── api/
│   └── index.md
│
├── mkdocs.yml
├── script.py
├── pyproject.toml
├── README.md
└── LICENSE

### Main Modules
| Package                  | Purpose                                         |
| ------------------------ | ----------------------------------------------- |
| `core`                   | Simulation, spacecraft, and mission management  |
| `actuators`              | Spacecraft actuator models                      |
| `controllers`            | Control laws and control allocation             |
| `enviroments`            | Simulation environment models                   |
| `faults`                 | Fault events, fault management, and fault modes |
| `general`                | Shared data structures and simulation utilities |
| `guidance`               | Guidance interfaces and algorithms              |
| `navigation`             | Navigation interfaces and algorithms            |
| `propagators`            | Translational and rotational propagation        |
| `sensors`                | Sensor models and measurement errors            |
| `solvers`                | Numerical integration methods                   |
| `cislunar_astrodynamics` | CR3BP and cislunar trajectory analysis          |
| `frames`                 | Reference-frame utilities                       |
| `visualization`          | Simulation and GNC visualization                |


## Development Status

ARGOS is under active development. The current architecture should be considered a research and development framework rather than flight-ready software.

### Implemented
- Modular spacecraft simulation
- Multi-spacecraft scenarios
- Mission phases and conditional transitions
- Translational dynamics
- Rotational rigid-body dynamics
- Modular sensors
- Navigation interfaces
- Guidance interfaces
- Controller interfaces
- Control allocation
- RCS actuator modeling
- Native numerical propagation
- RK45 integration
- Fault injection infrastructure
- Simulation history
- Visualization tools
- CR3BP analysis
- Lagrange point analysis
- Lyapunov orbit tools
- Halo orbit generation and continuation
- Automatic API documentation
  
### Under Development
- Higher-fidelity environmental models
- Expanded environmental perturbations
- Higher-fidelity sensor models
- Expanded actuator models
- Additional navigation algorithms
- Additional dynamics models
- FDIR capabilities
- MBSE model-to-simulation mapping
- Simulation-based requirements verification
- External simulation-tool integration


## Research Direction
The long-term objective with ARGOS is focused on increasing the fidelity and autonomy of spacecraft GNC simulations, particularly for multi-spacecraft and cislunar applications.

- RPOD guidance and control
- Advanced estimation
- Computer-vision-based navigation
- Additional spacecraft and environment models
- Hardware/software interfaces
- Higher-fidelity simulation environments
- MBSE-driven simulation configuration
- Simulation-based requirements verification
- Monte Carlo and experiment-generation capabilities
- Coupling with specialized simulation environments


## Development

The dev branch contains the active development architecture.

When adding new functionality, the preferred approach is to integrate it into the existing modular architecture rather than coupling it directly to the simulation engine.

For example, a new guidance algorithm should ideally implement the corresponding guidance interface and remain independent of the specific spacecraft, actuator, or propagator implementation.

This approach helps preserve the main design goal of ARGOS:

```text
Reusable component
        ↓
Standard interface
        ↓
Multiple simulation configurations
```

## Testing

ARGOS includes an automated test suite designed to verify both the structural integrity and the functional correctness of the toolkit.

The testing strategy combines:

- **Unit tests** to verify individual classes, functions, algorithms, and components in isolation.
- **Integration tests** to verify the interaction between multiple ARGOS components and ensure that complete simulation workflows behave as expected.
- **Structural tests** to verify that the package and its modules can be discovered and imported successfully.
- **Example execution tests** to ensure that the provided Python examples remain executable as the codebase evolves.

Run the complete test suite with:

```bash
pytest -q
```

## Dependencies

ARGOS is primarily built on the following Python libraries:

- **NumPy** — Numerical computing.
- **SciPy** — Scientific computing and numerical methods.
- **Matplotlib** — Data visualization and simulation plotting.
- **Pandas** — Data analysis and structured simulation results.
- **psutil** — System and process information used by the simulation infrastructure.

Development and testing additionally use:
- pytest
- MkDocs
- Material for MkDocs
- mkdocstrings

## Contributing

ARGOS is currently developed as part of an ongoing academic and research project. To maintain a consistent architecture, development methodology, and validation process, **direct code contributions (Pull Requests)** are not being accepted at this stage.

However, community involvement is highly encouraged through indirect contributions, including:

- Reporting bugs or unexpected behavior.
- Suggesting new features or improvements.
- Proposing new algorithms or research ideas.
- Sharing references, publications, or technical resources.
- Participating in technical discussions through GitHub Issues.

If you have an idea or would like to discuss a potential enhancement, please open an issue. Community feedback plays an important role in shaping the future direction of the project.

Once ARGOS reaches a more mature and stable architecture, direct code contributions will be welcomed under a formal contribution workflow.

## License

Copyright 2026 Autonomous RPOD & GNC for On-Orbit Servicing (ARGOS) Toolkit

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## AI-Assisted Development

ARGOS is developed with the assistance of Large Language Models (LLMs) as engineering productivity tools.

AI assistance is primarily used for:

- Improving code readability and formatting.
- Refining documentation and docstrings.
- Reviewing software architecture and design decisions.
- Identifying potential optimizations and implementation alternatives.
- Assisting with repository organization and project documentation.

All engineering decisions, algorithms, mathematical derivations, implementations, and validation are performed and reviewed by the author. AI-generated suggestions are treated as recommendations rather than authoritative solutions, and every contribution generated with AI is critically evaluated before being incorporated into the project.

The scientific accuracy, engineering validity, and overall quality of the software remain the sole responsibility of the author.

## Author

**Andres Rodriguez**

Aerospace Engineering Student

Aerospace Engineering student focused on spacecraft GNC, cislunar astrodynamics, RPOD, electronics, prototyping and rocketry.

- LinkedIn: [Andrés Felipe Rodríguez Acosta](https://www.linkedin.com/in/andr%C3%A9s-felipe-rodr%C3%ADguez-acosta-8a2b761a7/)
- Website: [Andrés Personal Website](https://rodandres.github.io/)