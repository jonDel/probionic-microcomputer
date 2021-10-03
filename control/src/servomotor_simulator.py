"Servomotor rotor and an object proximity control algorithm simulator."
from bisect import bisect_left
from pandas import DataFrame
import numpy as np
import plotly.graph_objects as go
from numpy import ndarray
from scipy import optimize


MATERIALS_PROFILES = {
    "metal blade": {
        "name": "metal blade",
        "max_deformation_mean": 0,
        "max_deformation_std": 0,
        "deformation_duration_mean": 0,
        "deformation_duration_std": 0,
        "peak_current_mean": 0,
        "peak_current_std": 0,
        "valley_current_mean": 0,
        "valley_current_std": 0,
        "n_samples_mean": 0,
        "n_samples_std": 0,
        "time_step_delay_std": 0
    },
    "rubber ball": {
        "name": "rubber ball",
        "max_deformation_mean": -0.5623423444444444,
        "max_deformation_std": 0.03395287190018319,
        "deformation_duration_mean": 0.8065369,
        "deformation_duration_std": 0.1354742445503574,
        "peak_current_mean": 735.43,
        "peak_current_std": 22.333206218543705,
        "valley_current_mean": 4.9799999999999995,
        "valley_current_std": 2.8634943687739285,
        "n_samples_mean": 252.5,
        "n_samples_std": 15.545095689637938,
        "current_noise_std": 8.777813898712585,
        "time_step_delay_std": 0.000801660182630725
    },
    "cardboard": {
        "name": "cardboard",
        "max_deformation_mean": 0,
        "max_deformation_std": 0,
        "deformation_duration_mean": 0,
        "deformation_duration_std": 0,
        "peak_current_mean": 0,
        "peak_current_std": 0,
        "valley_current_mean": 0,
        "valley_current_std": 0,
        "n_samples_mean": 0,
        "n_samples_std": 0,
        "time_step_delay_std": 0
    },
    "polystyrene": {
        "name": "polystyrene",
        "max_deformation_mean": 0,
        "max_deformation_std": 0,
        "deformation_duration_mean": 0,
        "deformation_duration_std": 0,
        "peak_current_mean": 0,
        "peak_current_std": 0,
        "valley_current_mean": 0,
        "valley_current_std": 0,
        "n_samples_mean": 0,
        "n_samples_std": 0,
        "time_step_delay_std": 0
    }
}


def limiter(value: float, minimum: float, maximum: float) -> float:
    """Clip the value to the range minimum - maximum.

    Args:
        value (float): given value to be clipped
        minimum (float): minimum accepted value
        maximum (float): maximum accepted value

    Returns:
        float: the input value clipped.
    """
    if value < minimum:
        value = minimum
    elif value > maximum:
        value = maximum
    return value


class PlantSimulator:
    """Simulate a plant with a servomotor and a current sensor."""

    def __init__(
        self,
        material_profile: dict,
        object_position: float = 0,
        n_samples: int = 0
    ) -> None:
        """Simulate the contact of an object with a servomotor rotor.

        Args:
            material_profile (dict): characteristics of a particular material.
            object_position (float, optional): Position of the object, in the
                same coordinates as the rotor, between -1 and 1. Defaults to 0.
            n_samples (int, optional): Number of samples(current readings),
                used to simulate the ramp of contact of the object with the
                rotor, from no contact to full contact(stall current). If not
                given, the default value from material profile dict will be
                used. Defaults to 0.
        """
        if object_position < -1 or object_position > 1:
            raise ValueError(
                "The object position must be a value between -1 and 1, "
                f"not {object_position}!"
            )
        self.object_position = object_position
        self.material_profile = material_profile
        self.original_n_samples = int(np.random.normal(
            loc=self.material_profile["n_samples_mean"],
            scale=self.material_profile["n_samples_std"]
        ))
        self.n_samples = n_samples if n_samples else self.original_n_samples
        self.peak_current = np.random.normal(
            loc=self.material_profile["peak_current_mean"],
            scale=self.material_profile["peak_current_std"]
        )
        self.deformation_duration = np.random.normal(
            loc=self.material_profile["deformation_duration_mean"],
            scale=self.material_profile["deformation_duration_std"],
        )
        self.ramp_df = DataFrame()
        self.ramp_df["position"] = self.__get_positions()
        self.ramp_df["proximity"] = self.__get_proximity_ramp()
        self.ramp_df["current"] = self.__get_currents()

    def __get_timesteps(self) -> ndarray:
        """Return the time steps array of the plant simulation.

        Returns:
            ndarray: the time steps array of the simulation.
        """
        ts_delay_std = self.material_profile["time_step_delay_std"]
        scaled_ts_delay_std = ts_delay_std*(
            self.original_n_samples/self.n_samples
        )
        ts_noise = np.random.normal(
            loc=0,
            scale=scaled_ts_delay_std,
            size=self.n_samples
        )
        t_spaced = np.linspace(
            0,
            self.deformation_duration,
            self.n_samples
        )
        return t_spaced + ts_noise

    def __get_positions(self) -> ndarray:
        """Return the deformation steps array of the object.

        Returns:
            ndarray: the deformation steps array of the simulation.
        """
        max_deformation = np.random.normal(
            loc=self.material_profile["max_deformation_mean"],
            scale=self.material_profile["max_deformation_std"]
        )
        final_position = abs(max_deformation) + self.object_position
        if final_position > 1:
            raise ValueError(
                "You must choose a lesser value for the object position"
                f" since, with the {self.material_profile['name']}, the "
                f"maximum deformation {final_position} would be greater "
                "than the maximum possible position for the rotor, 1"
            )
        return np.linspace(
            self.object_position,
            final_position,
            self.n_samples
        )

    def __get_proximity_ramp(self) -> ndarray:
        """Return the proximity ramp array of the object with the rotor.

        Returns:
            ndarray: the proximity ramp array.
        """
        time_steps = self.__get_timesteps()
        valley_current = np.random.normal(
            loc=self.material_profile["valley_current_mean"],
            scale=self.material_profile["valley_current_std"]
        )
        prox_first_step = abs(valley_current/self.peak_current)
        ramp = np.exp(
            np.linspace(
                np.log(prox_first_step),
                np.log(1),
                len(time_steps)
            )
        )
        (a, b), _ = optimize.curve_fit(
            lambda ts, a, b: a*np.e**(ts*b),
            time_steps,
            ramp)
        return a*(np.e**(b*time_steps))

    def __get_currents(self) -> ndarray:
        """Return the current ramp array from the contact with the object.

        Returns:
            ndarray: the current ramp array.
        """
        return self.ramp_df["proximity"].to_numpy()*self.peak_current

    def find_nearest_current(self, position: float) -> float:
        """Return the nearest servomotor current given a rotor position.

        Args:
            position (float): rotor position.

        Returns:
            float: servomotor current in mili Amperes.
        """
        curr_noise = np.random.normal(
            0,
            self.material_profile["current_noise_std"]
        )
        if position < self.ramp_df["position"].iloc[0]:
            current = self.ramp_df["current"].iloc[0]
        elif position >= self.ramp_df["position"].iloc[0] and \
                position < self.ramp_df["position"].iloc[-1]:
            index = bisect_left(self.ramp_df["position"].values, position)
            current = self.ramp_df["current"].iloc[index]
        else:
            current = self.ramp_df["current"].iloc[-1]
        current = current + curr_noise
        current = current if current else 0
        return current


class ProsthesisSimulation:
    """Simulate the use of a control algorithm on a prosthesis plant."""

    def __init__(
        self,
        plant_simulator: PlantSimulator,
        proximity_reference: float,
        rotor_position_reference: float,
        simulation_duration: float = 30,
        controller_params: dict = {}
    ) -> None:
        """Simulate a control algorithm on a prosthesis.

        Args:
            plant_simulator (PlantSimulator): prosthesis simulator object.
            proximity_reference (float): proximity reference for the control
                algorithm to follow.
            rotor_position_reference (float): initial position reference for
                the rotor to follow.
            simulation_duration (float, optional): simulation duration in
                seconds. Defaults to 30.
            controller_params (dict, optional): custom parameters to be used
                in the control methods. Defaults to {}.
        """
        self.plant_simulator = plant_simulator
        if rotor_position_reference < -1 or rotor_position_reference > 1:
            raise ValueError(
                "The rotor position reference must be a value between -1 and 1"
                f", not {rotor_position_reference}!"
            )
        self.rotor_position_reference = rotor_position_reference
        self.proximity_reference = proximity_reference
        self.simulation_duration = simulation_duration
        self.controller_params = controller_params
        self.rotor_positions = []
        self.errors = []
        self.proximity_estimates = []
        self.position_references = []
        self.controller_signals = []
        self.currents = []
        self.time_steps = self.__get_simulation_time_steps()

    def __get_simulation_time_steps(self) -> ndarray:
        """Return the time steps array of the control simulation.

        Returns:
            ndarray: the time steps array of the simulation.
        """
        def_dur = self.plant_simulator.deformation_duration
        duration_mean = def_dur/(self.plant_simulator.n_samples - 1)
        n_samples = int(self.simulation_duration/duration_mean) + 1
        return np.linspace(0, self.simulation_duration, n_samples)

    def controller(self, error: float, t_index: int, t_step: float) -> float:
        """Return the controller output.

        To be overridden with your own control algorithm.

        Args:
            error (float): the closed loop error.
            t_index (int): the index of the current iteration
            t_step (float): the time in seconds of the current iteration

        Raises:
            NotImplementedError: method not implemented in the child class.

        Returns:
            float: the controller output.
        """
        raise NotImplementedError("You must override this method!")

    def conditioned_plant_input(
        self,
        controller_signal: float,
        position_reference: float
    ) -> float:
        """Return the input to the plant given the controller output.

        To be overridden with your own conditioning.

        Args:
            controller_signal (float): controller output
            position_reference (float): current position reference

        Raises:
            NotImplementedError: method not implemented in the child class.

        Returns:
            float: the rotor position
        """
        raise NotImplementedError("You must override this method!")

    def recalibrate_position_reference(
            self, position_reference: float, plant_input: float) -> float:
        """Recalibrate the position reference to match the proximity reference.

        To be overridden with your own recalibration.

        Args:
            position_reference (float): current rotor position reference.
            plant_input (float): the input to the plant.

        Raises:
            NotImplementedError: method not implemented in the child class.

        Returns:
            float: the rotor position reference recalibrated.
        """
        raise NotImplementedError("You must override this method!")

    def proximity_estimation(self, current: float) -> float:
        """Return an estimative of the rotor proximity to the object.

        Args:
            current (float): the rotor current in mili Amperes.

        Returns:
            float: the proximity estimative.
        """
        proximity = (3.07e-06*(current)**3 - 4.53e-03*(current)**2
                     + 2.14*(current)+5.05)/(367.3*(850/750))
        return limiter(proximity, 0, 1)

    def error(
        self,
        proximity_reference: float,
        proximity_estimate: float
    ) -> float:
        """Return the closed loop controller error.

        Args:
            proximity_reference (float): proximity reference
            proximity_estimate (float): estimated proximity

        Returns:
            float: the controller error
        """
        return proximity_reference - proximity_estimate

    def run(self, show_plots: bool = True) -> DataFrame:
        """Run the simulation.

        Args:
            show_plots (bool, optional): If True, shows a detailed
                plot of the results the end of the simulation.
                Defaults to True.

        Returns:
            DataFrame: the results of the simulation as the dataframe
                columns (timestamp, rotor_position, error, proximity_estimate,
                position_reference, controller_signal, current,
                normalized_current)
        """
        position_reference = self.rotor_position_reference
        for t_index, t_step in enumerate(self.time_steps):
            current = self.plant_simulator.find_nearest_current(
                position_reference
            )
            proximity_estimate = self.proximity_estimation(current)
            error = self.error(self.proximity_reference, proximity_estimate)
            controller_signal = self.controller(error, t_index, t_step)
            plant_input = self.conditioned_plant_input(
                controller_signal, position_reference)
            self.errors.append(error)
            self.proximity_estimates.append(proximity_estimate)
            self.position_references.append(position_reference)
            self.controller_signals.append(controller_signal)
            self.currents.append(current)
            self.rotor_positions.append(plant_input)
            position_reference = self.recalibrate_position_reference(
                position_reference, plant_input
            )
        sim_df = DataFrame(
            {
                "timestamp": self.time_steps,
                "rotor_position": self.rotor_positions,
                "error": self.errors,
                "proximity_estimate": self.proximity_estimates,
                "position_reference": self.position_references,
                "controller_signal": self.controller_signals,
                "current": self.currents
            }
        )
        sim_df["normalized_current"] = sim_df.current/sim_df.current.max()
        if show_plots:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sim_df.timestamp,
                y=sim_df.controller_signal,
                name="Controller output")
            )
            fig.add_trace(go.Scatter(
                x=sim_df.timestamp,
                y=sim_df.normalized_current,
                name="Normalized current (between 0 and 1)")
            )
            fig.add_trace(go.Scatter(
                x=sim_df.timestamp,
                y=sim_df.error,
                name="Closed loop error")
            )
            fig.add_trace(go.Scatter(
                x=sim_df.timestamp,
                y=sim_df.rotor_position,
                name="Servomotor rotor position (between -1 and 1)")
            )
            fig.add_trace(go.Scatter(
                x=sim_df.timestamp,
                y=sim_df.proximity_estimate,
                name="Object proximity estimation (between 0 and 1)")
            )
            fig.update_layout(
                legend_title_text='Simulation variables',
                title="Control of a servomotor rotor against an object "
                      + f" ({self.plant_simulator.material_profile['name']})"
            )
            fig.update_xaxes(title_text="Time (seconds)")
            fig.update_yaxes(title_text="Variables values")
            fig.show()
        return sim_df
