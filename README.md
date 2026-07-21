# Autonomous RPOD & GNC for On-Orbit Servicing (ARGOS) Toolkit


> A modular research toolkit for developing and validating autonomous spacecraft Guidance, Navigation & Control (GNC) systems, enabling the transition from Earth-orbit  to cislunar and deep space Rendezvous, Proximity Operations and Docking (RPOD) for future On-Orbit Servicing missions.

![Status](https://img.shields.io/badge/status-active-success)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)
![Python](https://img.shields.io/badge/Python-3.14-blue)
![Project](https://img.shields.io/badge/project-Research-green)
![Domain](https://img.shields.io/badge/domain-RPOD-informational)
![Domain](https://img.shields.io/badge/domain-Spacecraft%20GNC-informational)
<!-- Add or remove badges depending on the project -->

## Overview
The Autonomous RPOD & GNC for On-Orbit Servicing (ARGOS) Toolkit is an open-source, modular research platform for developing and validating autonomous spacecraft Guidance, Navigation and Control (GNC) technologies. Its primary objective is to support the transition from conventional Earth-orbit missions to autonomous cislunar and deep-space Rendezvous, Proximity Operations and Docking (RPOD), laying the technological foundation for future On-Orbit Servicing (OOS) missions.

To achieve this objective, ARGOS provides a unified engineering framework that integrates astrodynamics, spacecraft dynamics, Guidance, Navigation and Control, state estimation, sensor and actuator modeling, environmental perturbations, and mission simulation within a common modular architecture. Each subsystem is designed to be independently developed, validated, and exchanged, enabling the rapid evaluation and comparison of algorithms across different mission scenarios and fidelity levels.

Unlike single-purpose simulators, ARGOS is intended to evolve as a research ecosystem, progressively expanding from classical attitude and translational dynamics in Earth orbit toward high-fidelity cislunar mission environments. The roadmap includes support for relative motion dynamics, nonlinear control, advanced state estimation, autonomous guidance, computer vision-based navigation, and realistic visualization of spacecraft operations, providing an end-to-end platform for autonomous RPOD research.

Ultimately, ARGOS aims to bridge the gap between fundamental spacecraft GNC research and the autonomous capabilities required for the next generation of deep-space exploration, orbital logistics, and On-Orbit Servicing missions.


### Engineering Objectives

ARGOS is designed to support the development and evaluation of:

- Modular spacecraft GNC architectures.
- Attitude and translational dynamics simulation.
- Relative motion, rendezvous, proximity operations, and docking (RPOD).
- Classical, optimal, adaptive, and nonlinear control techniques.
- State estimation, filtering, and sensor fusion.
- Astrodynamics for Earth-orbit, cislunar, and deep-space missions.
- High-fidelity simulation with configurable subsystem models.
- Autonomous guidance, navigation, and mission planning.
- Computer vision and perception for autonomous spacecraft operations.
- Technologies enabling future On-Orbit Servicing missions.


## Roadmap
ARGOS follows a progressive validation strategy in which every new capability is introduced and verified in a lower-complexity environment before being extended to higher-fidelity mission scenarios. This approach ensures that algorithms are validated incrementally, from classical Earth-orbit dynamics to autonomous cislunar Rendezvous, Proximity Operations, Docking, and On-Orbit Servicing.

### Near Term
- [x] Establish the ARGOS objective and roadmap.
- [x] Develop the initial modular attitude GNC simulation framework.
- [x] Implement configurable models for controllers, actuators, sensors, perturbations, and numerical integration.
- [x] Develop cislunar astrodynamics utilities and NRHO trajectory generation tools.

### Short-Term 
- [ ] Extend the simulation framework to six-degree-of-freedom (6-DOF) translational and attitude dynamics.
- [ ] Integrate modular state estimation and sensor fusion architectures.
- [ ] Implement relative motion dynamics and translational GNC modules.
- [ ] Validate the framework using Hill-Clohessy-Wiltshire (HCW) relative motion in Low Earth Orbit.

### Mid-Term
- [ ] Transition the RPOD framework from LEO validation to cislunar mission scenarios.
- [ ] Develop relative dynamics models in Near-Rectilinear Halo Orbits (NRHOs).
- [ ] Implement autonomous guidance, navigation, and control algorithms for cislunar RPOD.
- [ ] Expand support for realistic spacecraft sensors and mission environments.
### Long-Term
- [ ] Migrate the computational core from Python to modern C++ while maintaining a Python interface for research and rapid prototyping.
- [ ] Integrate a high-fidelity visualization environment using Unreal Engine.
- [ ] Develop computer vision and perception modules for autonomous navigation and docking.
- [ ] Implement advanced estimation, nonlinear control, optimization, and autonomous decision-making algorithms.
- [ ] Expand the framework toward complete On-Orbit Servicing mission scenarios.

## Capabilities

### Current

#### Simulation Core
- Modular GNC execution pipeline.
- Configurable multi-rate simulation scheduler.
- Adaptive and fixed-step numerical integration (RK45).

#### Dynamics
- Spacecraft attitude dynamics (quaternion formulation).
- Environmental perturbation framework.
    - Solar Radiation Pressure (SRP).
    - Gravity Gradient Torque.

#### Guidance, Navigation & Control
- PD attitude controller.

#### Sensors
- Gaussian covariance sensor model.

#### Actuators
- RCS thruster model.
    - PWM command modulation.
    - Bang-Bang command logic.
    - Independent thruster allocation.

#### Astrodynamics
- Circular Restricted Three-Body Problem (CR3BP).
- Lagrange point computation.
- NRHO family generation.
- Pseudo-arclength continuation.

#### Visualization
- Attitude animation.
- State history plots.

### In Development / Planned

ARGOS is under active development.

Upcoming features, planned enhancements, and ongoing work are tracked through the project's **GitHub Issues**. Community feedback and feature requests are welcome, and new ideas can be proposed by opening an issue.


## Project Structure

```text
ARGOS/
├── py/
│   ├── general/                  # Shared data structures, constants and configuration
│   ├── modules/                  # Modular implementations of spacecraft subsystems
│   │   ├── actuators.py
│   │   ├── controllers.py
│   │   ├── sensors.py
│   │   ├── perturbations.py
│   │   ├── attitude_dynamics.py
│   │   ├── cislunar_astrodynamics/
│   │   ├── solvers/
│   │   └── graphical/
│   ├── sim_engines/              # High-level simulation engines
│   └── usage_examples/           # Example applications and tutorials
├── outputs/                      # Simulation outputs
├── requirements.txt
└── README.md
```

| Directory         | Description                                                                                                                              |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `py/general/`        | Shared data structures, simulation settings, constants and common utilities.                                                             |
| `py/modules/`        | Reusable implementations of spacecraft dynamics, GNC algorithms, sensors, actuators, perturbations, numerical solvers and astrodynamics. |
| `py/sim_engines/`    | Core simulation engines orchestrating Guidance, Navigation, Control and dynamics execution.                                              |
| `py/examples/` | Complete examples demonstrating framework capabilities and module usage.                                                                 |
| `outputs/`        | Generated figures, animations and simulation results.                                                                                    |

## Installation

Clone the repository and install the required Python dependencies.

```bash
git clone https://github.com/<your-username>/ARGOS-Toolkit.git
cd ARGOS-Toolkit

pip install -r requirements.txt
```

ARGOS is currently intended to be executed directly from the repository. Packaging and installation through `pip` are planned for a future.


## Quick Start

The repository includes several standalone examples demonstrating the available modules and simulation engines.

For example, to run the attitude GNC simulation:

```bash
python py/examples/sim_v2.py
```

Additional examples are available in:

- `py/examples/` — GNC simulation examples.
- `py/examples/orbit_determination/` — Astrodynamics and orbit determination examples.

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