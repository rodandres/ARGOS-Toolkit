from abc import ABC, abstractmethod

import numpy as np

class SensorErrorModel(ABC):

    @abstractmethod
    def apply(self, measurement: np.ndarray, dt=None) -> np.ndarray:
        """
        Apply the electronic error model to the given measurement.

        Parameters
        ----------
        measurement : np.ndarray
            The input measurement to which the electronic error model will be applied.

        Returns
        -------
        np.ndarray
            The measurement after applying the electronic error model.
        """
        pass

class Bias(SensorErrorModel):

    def __init__(self, bias):
        self.bias = bias

    def apply(self, measurement, dt=None):
        return measurement + self.bias
    
class WhiteNoise(SensorErrorModel):

    def __init__(self, std):
        self.std = std

    def apply(self, measurement, dt=None):
        noise = np.random.normal(0, self.std, measurement.shape)
        return measurement + noise
    
class RandomWalk(SensorErrorModel): #NOTE: Possible BUG as it is not updating the bias time to time, only when it is called

    def __init__(self, std):
        self.std = std
        self.bias = np.zeros_like(std)

    def apply(self, measurement, dt=None):

        if dt is None:
            raise ValueError("RandomWalkError requires dt.")

        self.bias += np.random.normal(
            0,
            self.std*np.sqrt(dt),
            self.bias.shape
        )

        return measurement + self.bias
    
class Saturation(SensorErrorModel):

    def __init__(self, limit):
        self.limit = limit

    def apply(self, measurement, dt=None):
        return np.clip(
            measurement,
            -self.limit,
            self.limit
        )
    
class Quantization(SensorErrorModel):

    def __init__(self, limit, bits):

        self.limit = limit
        self.bits = bits

        print(f"Quantization Error Model: limit={self.limit}, bits={self.bits}")

    def apply(self, measurement, dt=None):

        if self.bits is None:
            return measurement
        
        if np.any(np.isinf(self.limit)):            
            return measurement

        step = (
            2*self.limit /
            (2**self.bits - 1)
        )

        return np.round(measurement/step)*step
