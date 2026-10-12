"""Multirotor flight control: discrete loops run from the simulation pre_step.

Cascade, outer to inner: velocity -> attitude -> body rate -> mixer -> rotor speeds.
Each loop runs once per simulation step with the state at the start of the step,
and its output is held for the whole step (zero-order hold, see core.simulation).
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.jinete.multirotor import MultirotorParams


def _axis_gains(value: npt.ArrayLike, name: str) -> np.ndarray:
    """Validate a per-axis gain [roll, pitch, yaw]: 3 values >= 0, read-only copy."""
    gains = np.array(value, dtype=float)
    if gains.shape != (3,) or np.any(gains < 0.0):
        raise ValueError(f"{name} must be 3 values >= 0 [roll, pitch, yaw]")
    gains.setflags(write=False)
    return gains


@dataclass(frozen=True)
class RateGains:
    """Gains of the body-rate loop, one value per axis [roll, pitch, yaw].

    The loop asks for an angular acceleration alpha = kp * e + ki * integral(e), with
    e = omega_cmd - omega, so the gains do not depend on the vehicle inertia.
    - kp [1/s]: with P only, each axis is a first-order system, time constant 1/kp.
    - ki [1/s^2]: with PI the axis follows s^2 + kp s + ki = 0; ki = kp^2 / 4 gives
      critical damping (fastest response without overshoot).
    - integral_limit [rad/s^2]: the integral term is clamped to +-this (anti-windup).
    """

    kp: np.ndarray
    ki: np.ndarray
    integral_limit: np.ndarray

    def __post_init__(self) -> None:
        for name in ("kp", "ki", "integral_limit"):
            object.__setattr__(self, name, _axis_gains(getattr(self, name), name))


@dataclass
class RateController:
    """Body-rate loop: [p, q, r] command + total thrust -> rotor speed commands.

    moment = I @ alpha + omega x (I @ omega). The second term cancels the gyroscopic
    coupling of Euler's equations, so each axis behaves as omega_dot = alpha.
    The moments and the thrust go through the mixer (with its clipping) and become
    rotor speeds at the reference density; the real density effect is left for the
    integral term to absorb, as on a real vehicle.
    """

    params: MultirotorParams
    gains: RateGains
    dt: float  # control period [s]: the simulation step, the loop runs every step
    integral: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.reset()

    def reset(self) -> None:
        """Forget the accumulated integral (for example before a new run)."""
        self.integral = np.zeros(3)

    def update(
        self, rate_command: npt.ArrayLike, thrust_command: float, state: np.ndarray
    ) -> np.ndarray:
        """Rotor speed commands [rad/s] for the commanded body rates [rad/s] and
        total thrust [N], given the 13-element state at the start of the step."""
        omega = state[10:13]
        error = np.asarray(rate_command, dtype=float) - omega
        limit = self.gains.integral_limit
        self.integral = np.clip(
            self.integral + self.gains.ki * error * self.dt, -limit, limit
        )
        angular_acceleration = self.gains.kp * error + self.integral

        inertia = self.params.inertia
        moment = inertia @ angular_acceleration + np.cross(omega, inertia @ omega)
        thrusts = self.params.rotor_thrusts([thrust_command, *moment])
        return self.params.rotor_speeds_for(thrusts)
