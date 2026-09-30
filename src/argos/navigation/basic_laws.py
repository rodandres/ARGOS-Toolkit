import numpy as np
from argos.navigation.navigation_base import NavigationBase
from argos.general.dataclasses import NavigationOutput, StateVariables

class IdealNavigation(NavigationBase):
    """
    Ideal navigation model using an absolute sensor measurement.

    The navigation estimate is obtained directly from the selected
    absolute sensor without introducing estimation errors.

    Parameters
    ----------
    sensor_to_be_use : int, optional
        One-based index of the absolute sensor to use from the provided
        sensor list. Defaults to 1.
    """
    def __init__(self, sensor_to_be_use=1):
        super().__init__()        
        self.sensor_to_be_use = sensor_to_be_use

    def _check_initialization(self):
        # No specific initialization checks for IdealNavigation
        pass

    def estimate(self, sensors: list) -> NavigationOutput:
        """
        Estimate the spacecraft and target states from an absolute sensor.

        The selected absolute sensor measurement is used directly to construct
        the navigation output without applying any estimation algorithm.

        Parameters
        ----------
        sensors : list of SensorBase
            Sensors available to the navigation system.

        Returns
        -------
        NavigationOutput
            Navigation output containing the spacecraft state and the reference
            or target state provided by the selected absolute sensor.

        Raises
        ------
        ValueError
            If the requested absolute sensor is not present in ``sensors``.
        """      
        absolute_sensor = None
        absolute_sensor_count = 0

        for sensor in sensors:
            if sensor.type == "AbsoluteSensor":
                absolute_sensor_count += 1

                if absolute_sensor_count == self.sensor_to_be_use:
                    absolute_sensor = sensor
                    break

        if absolute_sensor is None:
            raise ValueError("No AbsoluteSensor found in the provided sensors.")

        sensor_data = absolute_sensor.get_measurement()
        spacecraft_sensor_data = sensor_data[0]
        reference_sensor_data = sensor_data[1]

        spacecraft_state = StateVariables(
            position=spacecraft_sensor_data[0],
            velocity=spacecraft_sensor_data[1],
            acceleration=spacecraft_sensor_data[2],
            attitude=spacecraft_sensor_data[3],
            angular_velocity=spacecraft_sensor_data[4],
            angular_acceleration=spacecraft_sensor_data[5]
        )

        target_state = StateVariables(
            position=reference_sensor_data[0],
            velocity=reference_sensor_data[1],
            acceleration=reference_sensor_data[2],
            attitude=reference_sensor_data[3],
            angular_velocity=reference_sensor_data[4],
            angular_acceleration=reference_sensor_data[5]
        )

        return NavigationOutput(
            spacecraft_state=spacecraft_state,
            target_state=target_state
        )
       

class CustomNavigation(NavigationBase):
    """
    Navigation model that delegates state estimation to a user-defined function.

    Parameters
    ----------
    custom_estimation_function : callable
        Function used to estimate the spacecraft state. It receives the
        available sensors and must return a ``NavigationOutput`` instance.

    Raises
    ------
    ValueError
        If ``custom_estimation_function`` is not callable.
    """
    def __init__(self, custom_estimation_function):
        self.custom_estimation_function = custom_estimation_function
        super().__init__()

    def _check_initialization(self):
        if not callable(self.custom_estimation_function):
            raise ValueError("custom_estimation_function must be callable.")

    def estimate(self, sensors: list) -> NavigationOutput:
        """
        Estimate the spacecraft state using the user-defined function.

        Parameters
        ----------
        sensors : list of SensorBase
            Sensors available to the navigation system.

        Returns
        -------
        NavigationOutput
            Navigation output returned by the custom estimation function.

        Raises
        ------
        TypeError
            If the custom estimation function does not return a
            ``NavigationOutput`` instance.
        """
        output = self.custom_estimation_function(sensors)

        if not isinstance(output, NavigationOutput):
            raise TypeError("custom_estimation_function must return an NavigationOutput object.")

        return output

class NavigationPlaceholder(NavigationBase):
    """
    Placeholder navigation model for unsupported or unimplemented
    navigation logic.
    """
    def __init__(self):
        super().__init__()

    def _check_initialization(self):
        # No specific initialization checks for NavigationPlaceholder
        pass

    def estimate(self, sensors: list) -> NavigationOutput:
        """
        Placeholder method for spacecraft state estimation.

        Parameters
        ----------
        sensors : list of SensorBase
            Sensors available to the navigation system.

        Raises
        ------
        NotImplementedError
            Always raised because the navigation estimation is not implemented.
        """
        raise NotImplementedError("NavigationPlaceholder does not implement the estimate method.")