# ARGOS Toolkit

**ARGOS Toolkit** is a modular Python framework for **spacecraft simulation, Guidance Navigation and Control (GNC) research, and astrodynamics analysis**.

The toolkit is designed to make spacecraft simulations **composable, extensible, and configurable**, allowing different spacecraft subsystems, dynamics models, GNC algorithms, mission phases, and multi-spacecraft scenarios to be combined within a common simulation environment.

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

Spacecraft simulations often require the integration of several independent disciplines:

- Orbital and attitude dynamics
- Sensors and measurement models
- Navigation and state estimation
- Guidance algorithms
- Control laws
- Actuator models
- Mission logic
- Numerical propagation
- Data analysis and visualization

ARGOS approaches this problem through a **modular spacecraft architecture**, where these elements can be developed and combined independently.

A typical simulation can be structured as:

```text
Simulation
    │
    ├── Spacecraft
    │     │
    │     ├── Sensors
    │     ├── Navigation
    │     ├── Guidance
    │     ├── Controller
    │     ├── Control Allocation
    │     ├── Actuators
    │     └── Dynamics
    │
    └── Mission Manager
            │
            └── Mission Phases
```

During execution, the simulation coordinates the spacecraft subsystems through a common data flow:

```text
Sensors
   ↓
Navigation
   ↓
Guidance
   ↓
Control
   ↓
Control Allocation
   ↓
Actuators
   ↓
Spacecraft Dynamics
   ↓
Simulation State
```

This architecture makes it possible to replace or extend individual components without redesigning the entire simulation.

## Why ARGOS?

ARGOS is intended to reduce the amount of infrastructure required to investigate spacecraft GNC concepts.

Instead of implementing every simulation from scratch, a user can assemble a spacecraft from reusable components and focus on the problem being investigated.

The architecture emphasizes:

- **Modularity —** GNC and simulation components are separated into well-defined interfaces.
- **Composability —** Different sensors, navigation laws, guidance algorithms, controllers, actuators, and propagators can be combined.
- **Multi-spacecraft simulation —** Multiple spacecraft can coexist within the same simulation and exchange relevant state information.
- **Mission-level behavior —** Mission phases allow the active GNC and dynamics configuration to change throughout a simulation.
- **Research flexibility —** Custom algorithms can be introduced through extensible base interfaces.
- **Astrodynamics integration —** Spacecraft simulation capabilities are complemented by dedicated cislunar and CR3BP analysis tools.
- **Reproducible analysis —** Simulation results are stored in structured history objects that can be analyzed and visualized after execution.

The objective is not simply to provide a collection of equations or isolated algorithms, but to provide a common simulation framework in which spacecraft systems and GNC architectures can be investigated as integrated systems.

## Current Capabilities

| Area          | Capability                             | Status                |
| ------------- | -------------------------------------- | --------------------- |
| Simulation    | Modular simulation engine              | Implemented           |
| Simulation    | Multiple spacecraft                    | Implemented           |
| Simulation    | Individual spacecraft configurations   | Implemented           |
| Mission       | Mission phases                         | Implemented           |
| Mission       | Conditional phase transitions          | Implemented           |
| Dynamics      | Translational dynamics                 | Implemented           |
| Dynamics      | Rotational rigid-body dynamics         | Implemented           |
| Propagation   | Two-body relative propagation          | Implemented           |
| Propagation   | CR3BP propagation                      | Implemented           |
| Propagation   | Native adaptive RK45 solver            | Implemented           |
| Sensors       | Modular sensor architecture            | Implemented           |
| Sensors       | Absolute state measurements            | Implemented           |
| Sensors       | Sensor error/noise models              | Partially implemented |
| Navigation    | Ideal navigation                       | Implemented           |
| Navigation    | Custom navigation laws                 | Implemented           |
| Guidance      | Constant-reference guidance            | Implemented           |
| Guidance      | Custom guidance laws                   | Implemented           |
| Control       | PD attitude control                    | Implemented           |
| Control       | Custom controllers                     | Implemented           |
| Control       | Control allocation                     | Implemented           |
| Actuation     | RCS thruster model                     | Implemented           |
| Actuation     | PWM / bang-bang behavior               | Implemented           |
| Environment   | Environment abstraction                | Implemented           |
| Environment   | Integrated environmental perturbations | Under development     |
| Visualization | Trajectory visualization               | Implemented           |
| Visualization | State and GNC plots                    | Implemented           |
| Astrodynamics | CR3BP analysis                         | Implemented           |
| Astrodynamics | Lagrange point computation             | Implemented           |
| Astrodynamics | State Transition Matrix                | Implemented           |
| Astrodynamics | Lyapunov orbit tools                   | Implemented           |
| Astrodynamics | Halo orbit generation / continuation   | Implemented           |

## Architecture
The core architecture is organized around a small number of fundamental concepts.
### Simulation
`Simulation` is the top-level execution environment.

It manages:

- Simulation time
- Environment
- Multiple spacecraft
- Global execution timing
- Simulation history
- The overall simulation loop

### Spacecraft
A `Spacecraft` represents an individual simulated vehicle and contains its physical state, GNC configuration, sensors, actuators, and dynamics models.

```text
Conceptually:
Spacecraft
│
├── State
│   ├── Position
│   ├── Velocity
│   ├── Attitude
│   └── Angular velocity
│
├── Sensors
├── Navigation
├── Guidance
├── Controller
├── Control Allocator
├── Actuators
│
└── Dynamics
    ├── Translational
    └── Rotational
```

### Mission Manager
The MissionManager controls the sequence of mission phases assigned to a spacecraft.
A mission can therefore be represented as:
```text
Phase 1
   │
   ├── Navigation Law 1
   ├── Guidance Law 1
   ├── Control Law 1
   └── Translational Dynamics Only
   │
   └── Transition condition
            ↓
Phase 2
   │
   ├── Navigation Law 2
   ├── Guidance Law 1
   ├── Control Law 2
   └── Translational + Rotational Dynamics
```

This allows different GNC configurations to become active depending on the state of the mission.

### Modular GNC Components
Most major GNC subsystems follow a common base-class architecture:

```text
Base Interface
      │
      ├── Default implementation
      └── Custom implementation
```

This applies to areas such as:

- Sensors
- Navigation
- Guidance
- Controllers
- Actuators
- Propagators
- Environments

The intent is to allow users to replace individual algorithms without modifying the simulation infrastructure around them.

## Installation

Clone the repository and install the required Python dependencies.

```bash
git clone https://github.com/<your-username>/ARGOS-Toolkit.git
cd ARGOS-Toolkit

pip install -r requirements.txt
```

ARGOS is currently intended to be executed directly from the repository. Packaging and installation through `pip` are planned for a future.

## Quick Start

The simplest way to become familiar with ARGOS is to run the provided examples.

The examples are located under:
```text
py/examples/
```

and are provided both as Python scripts and Jupyter notebooks.

### Examples
The examples are organized to progressively introduce the framework.

#### Example 1 — Attitude GNC
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

#### Example 2 — Multi-Spacecraft Simulation
Demonstrates multiple spacecraft within the same simulation environment, including spacecraft-to-spacecraft state relationships.

This provides the basis for scenarios where one spacecraft needs information about another spacecraft.

#### Example 3 — CR3BP Propagation
Demonstrates translational propagation using the Circular Restricted Three-Body Problem (CR3BP), including the Earth-Moon system.

This example introduces the connection between the spacecraft simulation framework and the cislunar astrodynamics tools.

#### Example 4 — Mission Phases
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
ARGOS stores simulation results in structured history objects.

The recorded information can include:

- True spacecraft state
- Estimated state
- Guidance references
- Control outputs
- Applied forces
- Applied torques
- Simulation time
- Spacecraft-specific data

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

## Project Structure
```text
ARGOS-Toolkit/
│
├── py/
│   │
│   ├── general/
│   │   ├── dataclasses.py
│   │   ├── data_classes_declaration.py
│   │   ├── data_save.py
│   │   └── general_data.py
│   │
│   ├── modules/
│   │   │
│   │   ├── actuators/
│   │   ├── controllers/
│   │   ├── enviroments/
│   │   ├── frames/
│   │   ├── guidance/
│   │   ├── navigation/
│   │   ├── propagators/
│   │   ├── sensors/
│   │   ├── solvers/
│   │   ├── visualization/
│   │   │
│   │   ├── cislunar_astrodynamics/
│   │   │
│   │   └── core/
│   │       ├── simulation.py
│   │       ├── spacecraft.py
│   │       └── mission_manager.py
│   │
│   │
│   └── examples/
│       ├── py/
│       └── ipynb/
│
├── requirements.txt
├── LICENSE
└── README.md
``` 

### Main Modules
| Module                   | Purpose                                           |
| ------------------------ | ------------------------------------------------- |
| `general`                | Shared data structures and simulation history     |
| `actuators`              | Spacecraft actuator models                        |
| `controllers`            | Control laws and control allocation               |
| `enviroments`            | Simulation environment and perturbation framework |
| `frames`                 | Reference-frame utilities                         |
| `guidance`               | Guidance laws                                     |
| `navigation`             | Navigation and state estimation interfaces        |
| `propagators`            | Translational and rotational state propagation    |
| `sensors`                | Sensor models and measurement errors              |
| `solvers`                | Numerical integration                             |
| `visualization`          | Simulation and GNC visualization                  |
| `cislunar_astrodynamics` | CR3BP and cislunar trajectory analysis            |
| `new`                    | Current simulation architecture                   |
| `examples`               | Demonstrations and usage examples                 |

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
py/examples/xx/Continuation of Lyapunov and Halo Orbit Families.xx
``` 

## Development Status

ARGOS is under active development. The current architecture should be considered a research and development framework rather than flight-ready software.

### Implemented
- Modular spacecraft simulation
- Multi-spacecraft scenarios
- Mission phases and transitions
- Translational and rotational dynamics
- Basic GNC pipeline
- RCS actuator modeling
- Custom GNC interfaces
- Native numerical integration
- Simulation history
- Visualization tools
- CR3BP analysis
- Lagrange point analysis
- Lyapunov and Halo orbit tools
- Under Development
- More complete environmental modeling
- Integration of environmental perturbations
- Higher-fidelity sensor models
- More complete actuator modeling
- Expanded navigation capabilities
- Additional dynamics models
- Improved framework consistency and API stability
- Planned Direction
  
### The long-term development
The objective with ARGOS is focused on increasing the fidelity and autonomy of spacecraft GNC simulations, particularly for multi-spacecraft and cislunar applications.

### Potential development areas include:

- Relative orbital dynamics
- Relative navigation
- RPOD-specific guidance and control
- Advanced estimation
- Additional spacecraft and environment models
- Computer-vision-based navigation
- Hardware/software interfaces
- Higher-fidelity simulation environments
- Expanded autonomous spacecraft operations

These items represent development direction and should not be interpreted as currently implemented capabilities.

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

## Dependencies

ARGOS is primarily built on the following Python libraries:

- **NumPy** — Numerical computing.
- **SciPy** — Scientific computing and numerical methods.
- **Matplotlib** — Data visualization and simulation plotting.

Additional dependencies are listed in `requirements.txt`.

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