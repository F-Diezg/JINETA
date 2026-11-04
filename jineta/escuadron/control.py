"""Multirotor flight control for many drones at once: `jinete.control` on stacks.

The same discrete loops as the reference, run once per step from the pre_step of
the batch loop, with one row per drone: commands, states and the memory of every
loop (its integrals) are arrays (n, ...), and each drone's memory is its own. The
gains are those of the reference (`RateGains`, ...), shared by every drone of the
batch (one vehicle profile per batch).

`reset` takes the drones to reset: all of them, or only some (a boolean mask (n,)
or their indices), because in training each drone starts a new lap when its own
lap ends.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.core.vector import cross, matvec
from jineta.jinete.control import RateGains
from jineta.jinete.multirotor import MultirotorParams

# Which drones: None for all, a boolean mask (n,) or an array of indices.
Drones = npt.ArrayLike | None


def _rows(drones: Drones) -> slice | np.ndarray:
    """Turn a `Drones` argument into something that indexes the rows of an array."""
    return slice(None) if drones is None else np.asarray(drones)


def rotor_speeds_for_wrench(params: MultirotorParams, wrench: np.ndarray) -> np.ndarray:
    """Rotor speed commands (n, rotors) [rad/s] for the wrenches (n, 4)
    [T, Mx, My, Mz]: the mixer's inverse (thrusts = A+ @ wrench, the terms added in
    order), each thrust clipped to [0, max_rotor_thrust], then Omega = sqrt(T / k_T)
    at the reference density. `MultirotorParams.rotor_thrusts` followed by
    `rotor_speeds_for`, for every drone."""
    inverse = params.allocation_inv  # (rotors, 4)
    thrusts = wrench[:, 0:1] * inverse[:, 0]
    for j in range(1, 4):
        thrusts = thrusts + wrench[:, j : j + 1] * inverse[:, j]
    thrusts = np.clip(thrusts, 0.0, params.max_rotor_thrust)
    return np.sqrt(np.clip(thrusts, 0.0, None) / params.thrust_coefficient)


@dataclass
class BatchRateController:
    """Body-rate loop for n drones: [p, q, r] commands + total thrusts -> rotor
    speed commands, as `jinete.control.RateController`.

    alpha = kp e + ki integral(e) (integral clamped, anti-windup), e = omega_cmd -
    omega; moment = I alpha + omega x (I omega); through the mixer (with its
    clipping) to rotor speeds at the reference density.
    """

    params: MultirotorParams
    gains: RateGains
    dt: float  # control period [s]: the simulation step
    n: int  # number of drones
    integral: np.ndarray = field(init=False)  # (n, 3) [rad/s^2]

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.n < 1:
            raise ValueError("n (number of drones) must be >= 1")
        self.integral = np.zeros((self.n, 3))

    def reset(self, drones: Drones = None) -> None:
        """Forget the accumulated integral of these drones (all by default)."""
        self.integral[_rows(drones)] = 0.0

    def update(
        self,
        rate_command: np.ndarray,
        thrust_command: np.ndarray,
        states: np.ndarray,
    ) -> np.ndarray:
        """Rotor speed commands (n, rotors) [rad/s] for the commanded body rates
        (n, 3) [rad/s] and total thrusts (n,) [N], given the states (n, 13) at the
        start of the step."""
        omega = states[:, 10:13]
        error = rate_command - omega
        limit = self.gains.integral_limit
        self.integral = np.clip(
            self.integral + self.gains.ki * error * self.dt, -limit, limit
        )
        angular_acceleration = self.gains.kp * error + self.integral

        inertia = self.params.inertia
        moment = matvec(inertia, angular_acceleration) + cross(
            omega, matvec(inertia, omega)
        )
        wrench = np.column_stack([thrust_command, moment])
        return rotor_speeds_for_wrench(self.params, wrench)
