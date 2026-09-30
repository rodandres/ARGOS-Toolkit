from abc import ABC, abstractmethod

import numpy as np

class SensorErrorModel(ABC):
    """
    Abstract base class for sensor measurement error models.

    Error models transform an ideal sensor measurement to account for
    effects such as bias, noise, saturation, or quantization.
    """
    @abstractmethod
    def apply(self, measurement: np.ndarray, dt=None) -> np.ndarray:
        """
        Apply the error model to a sensor measurement.

        Parameters
        ----------
        measurement : np.ndarray
            Ideal or previously modified sensor measurement.
        dt : float, optional
            Time interval associated with the measurement update [s].

        Returns
        -------
        np.ndarray
            Measurement after applying the error model.
        """
        pass

class Bias(SensorErrorModel):
    """
    Add a constant bias to a sensor measurement.

    Parameters
    ----------
    bias : array-like
        Constant bias added element-wise to the measurement.
    """
    def __init__(self, bias):
        self.bias = bias

    def apply(self, measurement, dt=None):
        """
        Apply the constant bias to a measurement.

        Parameters
        ----------
        measurement : np.ndarray
            Sensor measurement.
        dt : float, optional
            Time interval associated with the measurement update [s].
            Not used by this error model.

        Returns
        -------
        np.ndarray
            Biased measurement.
        """
        return measurement + self.bias
    
class WhiteNoise(SensorErrorModel):
    """
    Add independent zero-mean Gaussian white noise to a measurement.

    Parameters
    ----------
    std : array-like or float
        Standard deviation of the Gaussian noise applied to each
        measurement component.
    """
    def __init__(self, std):
        self.std = std

    def apply(self, measurement, dt=None):
        """
        Apply zero-mean Gaussian white noise to a measurement.

        Parameters
        ----------
        measurement : np.ndarray
            Sensor measurement.
        dt : float, optional
            Time interval associated with the measurement update [s].
            Not used by this error model.

        Returns
        -------
        np.ndarray
            Measurement with additive Gaussian noise.
        """
        noise = np.random.normal(0, self.std, measurement.shape)
        return measurement + noise
    
class RandomWalk(SensorErrorModel):
    """
    Add a time-dependent random-walk bias to a sensor measurement.

    The internal bias is updated at each call using Gaussian increments
    whose standard deviation scales with the square root of the supplied
    time interval.

    Parameters
    ----------
    std : array-like or float
        Random-walk standard deviation parameter.
    """ #NOTE: Possible BUG as it is not updating the bias time to time, only when it is called

    def __init__(self, std):
        self.std = std
        self.bias = np.zeros_like(std)

    def apply(self, measurement, dt=None):
        """
        Update the random-walk bias and apply it to a measurement.

        Parameters
        ----------
        measurement : np.ndarray
            Sensor measurement.
        dt : float, optional
            Time interval associated with the measurement update [s].

        Returns
        -------
        np.ndarray
            Measurement with the current random-walk bias applied.

        Raises
        ------
        ValueError
            If ``dt`` is not provided.
        """
        if dt is None:
            raise ValueError("RandomWalkError requires dt.")

        self.bias += np.random.normal(
            0,
            self.std*np.sqrt(dt),
            self.bias.shape
        )

        return measurement + self.bias
    
class Saturation(SensorErrorModel):
    """
    Limit sensor measurements to a symmetric saturation range.

    Parameters
    ----------
    limit : array-like or float
        Positive absolute measurement limit. Values are clipped to the
        interval ``[-limit, limit]``.
    """
    def __init__(self, limit):
        self.limit = limit

    def apply(self, measurement, dt=None):
        """
        Apply symmetric saturation limits to a measurement.

        Parameters
        ----------
        measurement : np.ndarray
            Sensor measurement.
        dt : float, optional
            Time interval associated with the measurement update [s].
            Not used by this error model.

        Returns
        -------
        np.ndarray
            Saturated measurement.
        """
        return np.clip(
            measurement,
            -self.limit,
            self.limit
        )
    
class Quantization(SensorErrorModel):
    """
    Quantize sensor measurements using a finite number of bits.

    Parameters
    ----------
    limit : array-like or float
        Positive measurement range used to determine the quantization step.
    bits : int or None
        Number of quantization bits. If ``None``, quantization is disabled.
    """
    def __init__(self, limit, bits):

        self.limit = limit
        self.bits = bits        

    def apply(self, measurement, dt=None):
        """
        Quantize a sensor measurement.

        Parameters
        ----------
        measurement : np.ndarray
            Sensor measurement.
        dt : float, optional
            Time interval associated with the measurement update [s].
            Not used by this error model.

        Returns
        -------
        np.ndarray
            Quantized measurement.
        """
        if self.bits is None:
            return measurement
        
        if np.any(np.isinf(self.limit)):            
            return measurement

        step = (
            2*self.limit /
            (2**self.bits - 1)
        )

        return np.round(measurement/step)*step
