"""Implementation of standart control algorithms."""
from servomotor_simulator import ProsthesisSimulation, limiter


class ProsthesisSimulationKControl(ProsthesisSimulation):
    """Simulate the use of a proportional controller on a prosthesis plant."""

    def controller(self, error: float, t_index: int, t_step: float) -> float:
        """Return the proportional controller output.

        Args:
            error (float): the closed loop error.
            t_index (int): the index of the current iteration
            t_step (float): the time in seconds of the current iteration

        Returns:
            float: the controller output.
        """
        return self.controller_params["kp"]*error

    def conditioned_plant_input(
        self,
        controller_signal: float,
        position_reference: float
    ) -> float:
        """Return the input to the plant given the controller output.

        Args:
            controller_signal (float): controller output
            position_reference (float): current position reference

        Returns:
            float: the rotor position
        """
        plant_input = controller_signal + position_reference
        return limiter(plant_input, -1, 1)

    def recalibrate_position_reference(
            self, position_reference: float, plant_input: float) -> float:
        """Recalibrate the position reference to match the proximity reference.

        Args:
            position_reference (float): current rotor position reference.
            plant_input (float): the input to the plant.

        Returns:
            float: the rotor position reference recalibrated.
        """
        return plant_input


class ProsthesisSimulationIControl(ProsthesisSimulation):
    """Simulate the use of an integral controller on a prosthesis plant."""

    def controller(self, error: float, t_index: int, t_step: float) -> float:
        """Return the integral controller output.

        Args:
            error (float): the closed loop error.
            t_index (int): the index of the current iteration
            t_step (float): the time in seconds of the current iteration

        Returns:
            float: the controller output.
        """
        kp_output = self.controller_params["kp"]*error
        ki = self.controller_params["ki"]
        if t_index > 0:
            past_io = self.controller_params["integral_output"][t_index - 1]
        elif t_index == 0:
            past_io = 0
        else:
            raise ValueError(
                "The time step index must be a non negative integer, "
                f"not {t_index}!"
            )
        ki_output = ki*(t_step*error + past_io)
        self.controller_params["integral_output"].append(ki_output)
        return kp_output + ki_output

    def recalibrate_position_reference(
            self, position_reference: float, plant_input: float) -> float:
        """Recalibrate the position reference to match the proximity reference.

        Args:
            position_reference (float): current rotor position reference.
            plant_input (float): the input to the plant.
        Returns:
            float: the rotor position reference recalibrated.
        """
        return plant_input

    def conditioned_plant_input(
        self,
        controller_signal: float,
        position_reference: float
    ) -> float:
        """Return the input to the plant given the controller output.

        Args:
            controller_signal (float): controller output
            position_reference (float): current position reference
        Returns:
            float: the rotor position
        """
        plant_input = controller_signal + position_reference
        return limiter(plant_input, -1, 1)


class ProsthesisSimulationPIDControl(ProsthesisSimulation):
    """Simulate the use of an PID controller on a prosthesis plant."""

    def controller(self, error: float, t_index: int, t_step: float) -> float:
        """Return the PID controller output.

        Args:
            error (float): the closed loop error.
            t_index (int): the index of the current iteration
            t_step (float): the time in seconds of the current iteration

        Returns:
            float: the controller output.
        """
        kp_output = self.controller_params["kp"]*error
        ki = self.controller_params["ki"]
        kd = self.controller_params["kd"]
        if t_index > 0:
            past_io = self.controller_params["integral_output"][t_index - 1]
            time_delta = t_step - self.time_steps[t_index - 1]
            kd_output = kd*((error - self.errors[t_index - 1])/time_delta)
        elif t_index == 0:
            past_io = 0
            kd_output = 0
        else:
            raise ValueError(
                "The time step index must be a non negative integer, "
                f"not {t_index}!"
            )
        ki_output = ki*(t_step*error + past_io)
        self.controller_params["integral_output"].append(ki_output)
        return kp_output + ki_output + kd_output

    def recalibrate_position_reference(
            self, position_reference: float, plant_input: float) -> float:
        """Recalibrate the position reference to match the proximity reference.
        Args:
            position_reference (float): current rotor position reference.
            plant_input (float): the input to the plant.
        Returns:
            float: the rotor position reference recalibrated.
        """
        return plant_input

    def conditioned_plant_input(
        self,
        controller_signal: float,
        position_reference: float
    ) -> float:
        """Return the input to the plant given the controller output.

        Args:
            controller_signal (float): controller output
            position_reference (float): current position reference
        Returns:
            float: the rotor position
        """
        plant_input = controller_signal + position_reference
        return limiter(plant_input, -1, 1)
