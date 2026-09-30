import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass(frozen=True)
class Frame:
    name: str
    inertial: bool
    origin: str
    parent: str | None = None
    epoch: str | None = None

