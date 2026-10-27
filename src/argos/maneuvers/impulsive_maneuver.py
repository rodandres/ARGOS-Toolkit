from dataclasses import dataclass

import numpy as np

class ImpulsiveManeuver:
    """
    Define an ideal translational impulsive maneuver.

    Parameters
    ----------
    name : str
        Name of the maneuver.
    time : float
        Simulation time at which the maneuver is executed [s].
    delta_v : np.ndarray
        Velocity increment [m/s], expressed in the current propagation frame.    
    """

    def __init__(self, name: str, time: float, delta_v: np.ndarray):
        self.name = name
        self.time = time
        self.delta_v = np.asarray(delta_v, dtype=float)
        self.executed = False  # Flag to indicate if the maneuver has been executed

        if self.delta_v.shape != (3,):
            raise ValueError(
                "delta_v must be a vector with shape (3,)."
            )

        if not np.all(np.isfinite(self.delta_v)):
            raise ValueError(
                "delta_v must contain finite values."
            )

        if not np.isfinite(self.time) or self.time < 0:
            raise ValueError(
                "Maneuver time must be finite and non-negative."
            )        

        # Make delta_v immutable
        self.delta_v.setflags(write=False)