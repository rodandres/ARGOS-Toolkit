from dataclasses import dataclass, field
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from py.modules.new.simulation import Simulation


sim = Simulation(max_sim_time=100, dt_master=0.1, verbose=True)

sim.add_spacecraft(
    spacecraft_mass=1000.0,
    spacecraft_inertia_tensor=np.diag([500, 600, 700]),
    spacecraft_actuators=[None, None],
    spacecraft_sensors=[None],
    spacecraft_controllers=[None],
    spacecraft_guidance=[None],
    spacecraft_dt_nav=0.1,
    spacecraft_dt_guid=0.2,
    spacecraft_dt_control=0.3,
    initial_state=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),  # pos, vel, quat, omega in inertial frame
    name="Orion",
    verbose=True
)

