import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class Frame:
    """
    Identify a reference frame.

    Parameters
    ----------
    name : str
        Frame identifier.
    inertial : bool
        Whether the frame is inertial.
    origin : str
        Physical location defining the frame origin.
    parent : str or None
        Identifier of the parent frame, if applicable.
    epoch : str or None
        Epoch associated with the frame definition, if applicable.
    """
    name: str
    inertial: bool
    origin: str
    parent: str | None = None
    epoch: str | None = None