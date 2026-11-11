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

from jineta.core.environment import GRAVITY
from jineta.core.vector import cross, matvec
from jineta.escuadron import quaternion
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.jinete.control import (
    AttitudeGains,
    MultirotorGains,
    RateGains,
    VelocityGains,
)
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


@dataclass
class BatchAttitudeController:
    """Attitude loop for n drones: commanded attitudes (n, 4) -> body-rate
    commands (n, 3), as `jinete.control.AttitudeController`.

    q_err = conj(q_cmd) ⊗ q, the rotation from the command to the actual attitude
    in body axes; omega_cmd = -kp * (its rotation vector, the short way), clipped
    to the rate limits. No memory.
    """

    gains: AttitudeGains

    def update(self, attitude_command: np.ndarray, states: np.ndarray) -> np.ndarray:
        """Body-rate commands (n, 3) [rad/s] for the commanded attitudes (n, 4)
        (BODY -> NED), given the states (n, 13)."""
        error = quaternion.multiply(
            quaternion.conjugate(attitude_command), states[:, 6:10]
        )
        rate_command = -self.gains.kp * quaternion.to_rotation_vector(error)
        limit = self.gains.rate_limit
        return np.clip(rate_command, -limit, limit)


@dataclass
class BatchVelocityController:
    """Velocity loop for n drones: NED velocity commands (n, 3) + headings (n,) ->
    attitude commands (n, 4) + total thrusts (n,), as
    `jinete.control.VelocityController`.

    PI on the velocity error (integrals (n, 3)), thrust vector F = m (a - g), its
    limits with the vertical part first, the attitude with body z against F and the
    nose towards the heading, and the thrust F projected on the current thrust axis.
    """

    params: MultirotorParams
    gains: VelocityGains
    dt: float  # control period [s]
    n: int  # number of drones
    integral: np.ndarray = field(init=False)  # (n, 3) [m/s^2]

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.n < 1:
            raise ValueError("n (number of drones) must be >= 1")
        self.integral = np.zeros((self.n, 3))

    def reset(self, drones: Drones = None) -> None:
        """Forget the accumulated integral of these drones (all by default)."""
        self.integral[_rows(drones)] = 0.0

    @property
    def max_thrust(self) -> float:
        """Largest total thrust the loop will ask for [N]."""
        total = len(self.params.rotors) * self.params.max_rotor_thrust
        return self.gains.thrust_range[1] * total

    def update(
        self,
        velocity_command: np.ndarray,
        yaw_command: np.ndarray,
        states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Attitude commands (n, 4) (BODY -> NED) and total thrusts (n,) [N] for the
        commanded NED velocities (n, 3) [m/s] and headings (n,) [rad], given the
        states (n, 13)."""
        error = velocity_command - states[:, 3:6]
        limit = self.gains.integral_limit
        self.integral = np.clip(
            self.integral + self.gains.ki * error * self.dt, -limit, limit
        )
        acceleration = self.gains.kp * error + self.integral

        thrust_ned = self.params.mass * (acceleration - np.array([0.0, 0.0, GRAVITY]))
        thrust_ned = self._limit(thrust_ned)

        size = _norm(thrust_ned)
        body_z = -thrust_ned / size[:, None]
        attitude_command = _attitudes_from(body_z, yaw_command)

        current_body_z = quaternion.to_matrix(states[:, 6:10])[:, :, 2]
        projected = (
            thrust_ned[:, 0] * -current_body_z[:, 0]
            + thrust_ned[:, 1] * -current_body_z[:, 1]
            + thrust_ned[:, 2] * -current_body_z[:, 2]
        )
        return attitude_command, np.maximum(projected, 0.0)

    def _limit(self, thrust_ned: np.ndarray) -> np.ndarray:
        """The thrust and tilt limits, keeping the vertical part first."""
        total = len(self.params.rotors) * self.params.max_rotor_thrust
        low, high = self.gains.thrust_range
        up = np.clip(-thrust_ned[:, 2], low * total, high * total)
        horizontal = thrust_ned[:, 0:2]
        max_horizontal = np.minimum(
            up * np.tan(self.gains.max_tilt), np.sqrt((high * total) ** 2 - up**2)
        )
        size = np.sqrt(
            horizontal[:, 0] * horizontal[:, 0] + horizontal[:, 1] * horizontal[:, 1]
        )
        too_long = size > max_horizontal
        with np.errstate(divide="ignore", invalid="ignore"):
            shortened = horizontal * (max_horizontal / size)[:, None]
        horizontal = np.where(too_long[:, None], shortened, horizontal)
        return np.column_stack([horizontal, -up])


def _norm(vectors: np.ndarray) -> np.ndarray:
    """Lengths (n,) of 3-vectors (n, 3), the squares added in order."""
    return np.sqrt(
        vectors[:, 0] * vectors[:, 0]
        + vectors[:, 1] * vectors[:, 1]
        + vectors[:, 2] * vectors[:, 2]
    )


def _attitudes_from(body_z: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Quaternions (n, 4) whose body z axes are `body_z` (n, 3) (NED, unit) and
    whose noses point as close as possible to the headings `yaw` (n,) [rad]."""
    heading = np.column_stack([np.cos(yaw), np.sin(yaw), np.zeros(len(yaw))])
    body_y = cross(body_z, heading)
    body_y = body_y / _norm(body_y)[:, None]
    body_x = cross(body_y, body_z)
    return quaternion.from_matrix(np.stack([body_x, body_y, body_z], axis=-1))


@dataclass
class BatchAutopilot:
    """Velocity autopilot for n drones: the whole cascade in one pre_step hook, as
    `jinete.control.Autopilot`.

    Set `velocity_command` (n, 3) NED [m/s] and `yaw_command` (n,) [rad]; calling
    the autopilot with (t, states) runs velocity -> attitude -> rate for every drone
    and writes the rotor speed commands (n, rotors) of the vehicles.

    `landed` (n,) says which drones stand on the ground. A landed drone has its
    motors stopped and its integrators held at zero; asked to climb faster than
    the take-off speed it takes off, with the loops starting from zero. A flying
    drone lands after `hold_time` in a row slower than `max_speed` while asking for
    less than `thrust_fraction` of its weight. The reference decides this with an
    `if` per drone; here every loop runs for every drone and the drones that stay
    on the ground get their integrators put back to zero and their motors stopped,
    which leaves them exactly where the reference's early return leaves them.
    """

    vehicle: BatchMultirotor
    gains: MultirotorGains
    dt: float  # control period [s]: the simulation step
    landed: npt.ArrayLike = False  # (n,) or one value for all
    velocity_command: np.ndarray = field(init=False)
    yaw_command: np.ndarray = field(init=False)
    still_time: np.ndarray = field(init=False)  # how long landing has held [s]
    velocity_loop: BatchVelocityController = field(init=False, repr=False)
    attitude_loop: BatchAttitudeController = field(init=False, repr=False)
    rate_loop: BatchRateController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.vehicle.n
        params = self.vehicle.params
        self.landed = np.broadcast_to(np.asarray(self.landed, bool), (n,)).copy()
        self.velocity_command = np.zeros((n, 3))
        self.yaw_command = np.zeros(n)
        self.still_time = np.zeros(n)
        self.velocity_loop = BatchVelocityController(
            params, self.gains.velocity, self.dt, n
        )
        self.attitude_loop = BatchAttitudeController(self.gains.attitude)
        self.rate_loop = BatchRateController(params, self.gains.rate, self.dt, n)

    def reset(self, drones: Drones = None) -> None:
        """Forget the integrators and the landing timer of these drones (all by
        default). Whether they are landed is left as it is."""
        rows = _rows(drones)
        self.velocity_loop.reset(rows)
        self.rate_loop.reset(rows)
        self.still_time[rows] = 0.0

    def __call__(self, t: float | np.ndarray, states: np.ndarray) -> None:
        """The pre_step hook: update the rotor speed commands of the vehicles."""
        land = self.gains.land
        self.reset(self.landed)
        self.landed = self.landed & ~(self.velocity_command[:, 2] < -land.takeoff_speed)
        on_ground = self.landed.copy()

        attitude_command, thrust = self.velocity_loop.update(
            self.velocity_command, self.yaw_command, states
        )
        rate_command = self.attitude_loop.update(attitude_command, states)
        rotor_speeds = self.rate_loop.update(rate_command, thrust, states)

        # The drones that stay on the ground: as if the loops had not run.
        self.reset(on_ground)
        rotor_speeds[on_ground] = 0.0
        self.vehicle.rotor_speed_command = rotor_speeds

        weight = self.vehicle.params.mass * GRAVITY
        still = (_norm(states[:, 3:6]) < land.max_speed) & (
            thrust < land.thrust_fraction * weight
        )
        flying = ~on_ground
        self.still_time = np.where(
            flying, np.where(still, self.still_time + self.dt, 0.0), self.still_time
        )
        self.landed = self.landed | (flying & (self.still_time >= land.hold_time))


@dataclass
class BatchRatePilot:
    """Rate pilot for n drones: collective thrusts and body rates through the rate
    loop, as `jinete.control.RatePilot`.

    Set `thrust_command` (n,) [N] and `rate_command` (n, 3) [rad/s]; calling the
    pilot with (t, states) runs the rate loop for every drone and writes the rotor
    speed commands (n, rotors) of the vehicles. `landed` (n,): those drones do not
    integrate (their integrals are reset before and after the update) and take off
    when their commanded thrust exceeds the weight.
    """

    vehicle: BatchMultirotor
    gains: RateGains
    dt: float  # control period [s]: the simulation step
    landed: npt.ArrayLike = False  # (n,) or one value for all
    thrust_command: np.ndarray = field(init=False)
    rate_command: np.ndarray = field(init=False)
    rate_loop: BatchRateController = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.vehicle.n
        self.landed = np.broadcast_to(np.asarray(self.landed, bool), (n,)).copy()
        self.thrust_command = np.zeros(n)
        self.rate_command = np.zeros((n, 3))
        self.rate_loop = BatchRateController(
            self.vehicle.params, self.gains, self.dt, n
        )

    def reset(self, drones: Drones = None) -> None:
        """Forget the rate loop's integral of these drones (all by default).
        Whether they are landed is left as it is."""
        self.rate_loop.reset(drones)

    def __call__(self, t: float | np.ndarray, states: np.ndarray) -> None:
        """The pre_step hook: update the rotor speed commands of the vehicles."""
        self.reset(self.landed)
        weight = self.vehicle.params.mass * GRAVITY
        self.landed = self.landed & ~(self.thrust_command > weight)
        self.vehicle.rotor_speed_command = self.rate_loop.update(
            self.rate_command, self.thrust_command, states
        )
        self.reset(self.landed)
