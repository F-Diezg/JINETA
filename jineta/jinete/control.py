"""Multirotor flight control: discrete loops run from the simulation pre_step.

Cascade, outer to inner: velocity -> attitude -> body rate -> mixer -> rotor speeds.
Each loop runs once per simulation step with the state at the start of the step,
and its output is held for the whole step (zero-order hold, see core.simulation).
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.core import quaternion
from jineta.core.environment import GRAVITY
from jineta.jinete.multirotor import Multirotor, MultirotorParams


def _axis_gains(value: npt.ArrayLike, name: str) -> np.ndarray:
    """Validate a per-axis gain ([roll, pitch, yaw] or [north, east, down]): 3 values
    >= 0, read-only copy."""
    gains = np.array(value, dtype=float)
    if gains.shape != (3,) or np.any(gains < 0.0):
        raise ValueError(f"{name} must be 3 values >= 0")
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


@dataclass(frozen=True)
class AttitudeGains:
    """Gains of the attitude loop, one value per axis [roll, pitch, yaw].

    - kp [1/s]: rate command = -kp * attitude error (rotation vector). Over a rate
      loop with P gain kp_rate, each axis follows s^2 + kp_rate s + kp_rate kp = 0;
      kp = kp_rate / 4 gives critical damping.
    - rate_limit [rad/s]: the rate command is clipped to +-this.
    """

    kp: np.ndarray
    rate_limit: np.ndarray

    def __post_init__(self) -> None:
        for name in ("kp", "rate_limit"):
            object.__setattr__(self, name, _axis_gains(getattr(self, name), name))


@dataclass
class AttitudeController:
    """Attitude loop: commanded attitude quaternion -> body-rate command.

    The error is q_err = conj(q_cmd) ⊗ q: the rotation that takes the commanded
    attitude to the actual one, expressed in body axes. As a rotation vector theta
    (the short way, < 180 degrees), the body rates change it at d(theta)/dt ~ omega,
    so omega_cmd = -kp * theta turns the body back towards the command.
    """

    gains: AttitudeGains

    def update(self, attitude_command: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Body-rate command [rad/s] for the commanded attitude (quaternion
        BODY -> NED), given the 13-element state."""
        error = quaternion.multiply(quaternion.conjugate(attitude_command), state[6:10])
        rate_command = -self.gains.kp * quaternion.to_rotation_vector(error)
        limit = self.gains.rate_limit
        return np.clip(rate_command, -limit, limit)



@dataclass(frozen=True)
class VelocityGains:
    """Gains and limits of the velocity loop. Per-axis values are [north, east, down].

    - kp [1/s] and ki [1/s^2]: acceleration command a = kp * e + ki * integral(e),
      with e = v_cmd - v in NED (the same units as the PX4 MPC_*_VEL_*_ACC gains).
    - integral_limit [m/s^2]: clamp on the integral term (anti-windup).
    - max_tilt [rad]: largest angle between the thrust axis and the vertical.
    - thrust_range: (min, max) total thrust as fractions of the vehicle maximum. The
      minimum keeps the thrust axis defined; the margin below 1 leaves rotor thrust
      for the attitude control.
    """

    kp: np.ndarray
    ki: np.ndarray
    integral_limit: np.ndarray
    max_tilt: float
    thrust_range: tuple[float, float] = (0.1, 0.9)

    def __post_init__(self) -> None:
        for name in ("kp", "ki", "integral_limit"):
            object.__setattr__(self, name, _axis_gains(getattr(self, name), name))
        if not 0.0 < self.max_tilt < np.pi / 2:
            raise ValueError("max_tilt must be between 0 and pi/2 rad")
        low, high = self.thrust_range
        if not 0.0 < low < high <= 1.0:
            raise ValueError("thrust_range must satisfy 0 < min < max <= 1")


@dataclass
class VelocityController:
    """Velocity loop: NED velocity command + heading -> attitude + total thrust.

    1. Acceleration command a from the velocity error (PI, NED).
    2. Thrust vector the rotors must produce: F = m (a - g), with g = [0, 0, g] in
       NED (hovering needs F = [0, 0, -m g], straight up).
    3. Limits, vertical first: the upward part is kept within the thrust range and
       the horizontal part is shortened so that the tilt stays below max_tilt and
       the total thrust below its maximum.
    4. Attitude: the body z axis points against F (thrust acts along -z_body), the
       nose as close as possible to the commanded heading.
    5. Total thrust = F projected on the current thrust axis, so that the vehicle
       does not push in a wrong direction while its attitude is still turning.
    """

    params: MultirotorParams
    gains: VelocityGains
    dt: float  # control period [s]
    integral: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        self.reset()

    def reset(self) -> None:
        """Forget the accumulated integral."""
        self.integral = np.zeros(3)

    @property
    def max_thrust(self) -> float:
        """Largest total thrust the loop will ask for [N]."""
        total = len(self.params.rotors) * self.params.max_rotor_thrust
        return self.gains.thrust_range[1] * total

    def update(
        self, velocity_command: npt.ArrayLike, yaw_command: float, state: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Attitude command (quaternion BODY -> NED) and total thrust [N] for the
        commanded NED velocity [m/s] and heading [rad], given the 13-element state."""
        error = np.asarray(velocity_command, dtype=float) - state[3:6]
        limit = self.gains.integral_limit
        self.integral = np.clip(
            self.integral + self.gains.ki * error * self.dt, -limit, limit
        )
        acceleration = self.gains.kp * error + self.integral

        thrust_ned = self.params.mass * (acceleration - np.array([0.0, 0.0, GRAVITY]))
        thrust_ned = self._limit(thrust_ned)

        body_z = -thrust_ned / np.linalg.norm(thrust_ned)
        attitude_command = _attitude_from(body_z, yaw_command)

        current_body_z = quaternion.to_matrix(state[6:10])[:, 2]
        thrust = max(float(thrust_ned @ -current_body_z), 0.0)
        return attitude_command, thrust

    def _limit(self, thrust_ned: np.ndarray) -> np.ndarray:
        """Apply the thrust and tilt limits, keeping the vertical part first."""
        total = len(self.params.rotors) * self.params.max_rotor_thrust
        low, high = self.gains.thrust_range
        up = np.clip(-thrust_ned[2], low * total, high * total)
        horizontal = thrust_ned[0:2]
        max_horizontal = min(
            up * np.tan(self.gains.max_tilt), np.sqrt((high * total) ** 2 - up**2)
        )
        size = np.linalg.norm(horizontal)
        if size > max_horizontal:
            horizontal = horizontal * (max_horizontal / size)
        return np.array([horizontal[0], horizontal[1], -up])


def _attitude_from(body_z: np.ndarray, yaw: float) -> np.ndarray:
    """Quaternion whose body z axis is `body_z` (NED, unit) and whose nose points as
    close as possible to the heading `yaw` [rad, clockwise from North]."""
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    body_y = np.cross(body_z, heading)
    body_y = body_y / np.linalg.norm(body_y)
    body_x = np.cross(body_y, body_z)
    return quaternion.from_matrix(np.column_stack([body_x, body_y, body_z]))


@dataclass(frozen=True)
class MultirotorGains:
    """All the gains of the cascade for one vehicle profile."""

    rate: RateGains
    attitude: AttitudeGains
    velocity: VelocityGains


@dataclass
class Autopilot:
    """Velocity autopilot: the whole cascade in one pre_step hook.

    Set `velocity_command` (NED [m/s]) and `yaw_command` ([rad]); pass the autopilot
    itself as `pre_step` to `run`. Each call runs velocity -> attitude -> rate and
    writes the rotor speed commands into the vehicle.
    """

    vehicle: Multirotor
    gains: MultirotorGains
    dt: float  # control period [s]: the simulation step
    velocity_command: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_command: float = 0.0
    velocity_loop: VelocityController = field(init=False, repr=False)
    attitude_loop: AttitudeController = field(init=False, repr=False)
    rate_loop: RateController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        params = self.vehicle.params
        self.velocity_loop = VelocityController(params, self.gains.velocity, self.dt)
        self.attitude_loop = AttitudeController(self.gains.attitude)
        self.rate_loop = RateController(params, self.gains.rate, self.dt)

    def __call__(self, t: float, state: np.ndarray) -> None:
        """The pre_step hook: update the rotor speed commands of the vehicle."""
        attitude_command, thrust = self.velocity_loop.update(
            self.velocity_command, self.yaw_command, state
        )
        rate_command = self.attitude_loop.update(attitude_command, state)
        self.vehicle.rotor_speed_command = self.rate_loop.update(
            rate_command, thrust, state
        )