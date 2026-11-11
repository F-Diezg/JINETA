"""What a policy sees and what it commands: the observation and the action.

Spaces. A space says what shape a vector has and between which limits each of its
numbers lies, as Gymnasium's `Box` does (`BoxSpace`; float32, the type learning
libraries use; the physics stays float64). The adapter of phase 11 turns these
into the library's own spaces.

Action. Four numbers in [-1, 1], the same range on every axis (learning
algorithms work best with commands of the same size): the first is the
collective thrust, mapped linearly onto [thrust_min, thrust_max] [N]; the other
three are the body rates [p, q, r] as fractions of `max_rate` [rad/s]. Out-of-
range actions are clipped. The rate pilot executes them (`BatchRatePilot`). The
default envelope is the autopilot's own (`autopilot_envelope`): the thrust range
of its velocity loop and the rate limits of its attitude loop, so a policy and
the classical baseline fly with the same limits.

Observation. In body axes, so that it does not depend on where the course lies
or which way it faces (only on where things are from the drone's point of view):

    velocity in body axes               R^T v                    3
    attitude (rotation matrix BODY->NED, row by row)             9
    body rates [p, q, r]                                         3
    the next `lookahead` gates: 4 points on the edge of each
      opening, relative to the drone, in body axes R^T (p - x)   12 per gate
    the previous action                                          4

Units are SI (m, m/s, rad/s): scaling the numbers for learning is the training's
business (phase 11). The four points of a rectangular opening are its corners;
those of a circular one are the corners of the square inscribed in it (at 45°),
in the same order: top left, top right, bottom right, bottom left, as seen by a
drone flying through. They give the gate's place, size and heading at once. After
the last gate (or when fewer than `lookahead` remain), the last gate is repeated.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jineta.campo.gates import CircularGate, Gate, RectangularGate
from jineta.core.vector import matvec
from jineta.escuadron import quaternion
from jineta.jinete.control import MultirotorGains
from jineta.jinete.multirotor import MultirotorParams

ACTION_SIZE = 4  # collective thrust, p, q, r
OPENING_POINTS = 4  # points per gate in the observation


def _read_only(value: npt.ArrayLike, dtype: type = float) -> np.ndarray:
    array = np.array(value, dtype=dtype)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class BoxSpace:
    """Vectors of a fixed shape whose numbers lie between `low` and `high` (each
    as long as the vector; infinite for no limit), float32."""

    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = _read_only(self.low, np.float32)
        high = _read_only(self.high, np.float32)
        if low.ndim != 1 or low.shape != high.shape or len(low) == 0:
            raise ValueError("low and high must be 1-d arrays of the same length")
        if np.any(np.isnan(low)) or np.any(np.isnan(high)) or np.any(low > high):
            raise ValueError("every low must be <= its high")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    @property
    def dtype(self) -> type:
        return np.float32

    def contains(self, x: npt.ArrayLike) -> bool:
        """True when x has this shape (or is a stack of such vectors, (n, size))
        and every number is inside its limits."""
        x = np.asarray(x)
        if x.shape[-1:] != self.shape or x.ndim > 2:
            return False
        return bool(np.all((x >= self.low) & (x <= self.high)))

    def sample(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
        """Random vectors inside the limits (uniform; the space must be bounded):
        one (size,) or n of them (n, size)."""
        if not (np.all(np.isfinite(self.low)) and np.all(np.isfinite(self.high))):
            raise ValueError("only a bounded space can be sampled")
        size = self.shape if n is None else (n, *self.shape)
        return rng.uniform(self.low, self.high, size=size).astype(np.float32)


# ---- action ------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionLimits:
    """The envelope of a policy's commands: total thrust in [thrust_min,
    thrust_max] [N], body rates within +-max_rate [p, q, r] [rad/s]."""

    thrust_min: float
    thrust_max: float
    max_rate: np.ndarray

    def __post_init__(self) -> None:
        if not 0.0 <= self.thrust_min < self.thrust_max:
            raise ValueError("the thrust range must satisfy 0 <= min < max")
        rates = _read_only(self.max_rate)
        if rates.shape != (3,) or not np.all(rates > 0.0):
            raise ValueError("max_rate must be 3 values > 0")
        object.__setattr__(self, "max_rate", rates)

    @property
    def space(self) -> BoxSpace:
        return BoxSpace(-np.ones(ACTION_SIZE), np.ones(ACTION_SIZE))

    def commands(self, actions: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """Total thrusts (n,) [N] and body rates (n, 3) [rad/s] for the actions
        (n, 4), each clipped to [-1, 1] first."""
        a = np.clip(np.asarray(actions, dtype=float), -1.0, 1.0)
        span = self.thrust_max - self.thrust_min
        thrust = self.thrust_min + (a[:, 0] + 1.0) / 2.0 * span
        return thrust, a[:, 1:4] * self.max_rate

    def actions(self, thrust: npt.ArrayLike, rates: npt.ArrayLike) -> np.ndarray:
        """The actions (n, 4) for total thrusts (n,) [N] and body rates (n, 3)
        [rad/s], clipped to [-1, 1]: the inverse of `commands` inside the
        envelope (a classical controller flies through it this way)."""
        span = self.thrust_max - self.thrust_min
        thrust_part = 2.0 * (np.asarray(thrust, dtype=float) - self.thrust_min) / span
        a = np.column_stack(
            [thrust_part - 1.0, np.asarray(rates, dtype=float) / self.max_rate]
        )
        return np.clip(a, -1.0, 1.0)


def autopilot_envelope(
    params: MultirotorParams, gains: MultirotorGains
) -> ActionLimits:
    """The envelope the vehicle's own autopilot flies in: the thrust range of its
    velocity loop (fractions of all rotors at full thrust) and the rate limits of
    its attitude loop. For the Iris: 10-90 % of 4 x 7.07 N, and 3.5 / 3.5 /
    1.5 rad/s (PX4's MC_ROLLRATE_MAX 220 deg/s for roll and pitch)."""
    total = len(params.rotors) * params.max_rotor_thrust
    low, high = gains.velocity.thrust_range
    return ActionLimits(low * total, high * total, gains.attitude.rate_limit)


# ---- observation -------------------------------------------------------------------


def opening_points(gate: Gate) -> np.ndarray:
    """The four points (4, 3) NED [m] on the edge of a gate's opening: top left, top
    right, bottom right, bottom left as seen flying through it (u to the right, v
    down). A rectangle's corners; for a circle, the corners of the inscribed
    square."""
    if isinstance(gate, RectangularGate):
        half_u, half_v = gate.width / 2.0, gate.height / 2.0
    elif isinstance(gate, CircularGate):
        half_u = half_v = gate.radius / np.sqrt(2.0)
    else:
        raise TypeError(f"unknown gate type {type(gate).__name__}")
    corners = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    u = corners[:, 0] * half_u
    v = corners[:, 1] * half_v
    c, s = np.cos(gate.yaw), np.sin(gate.yaw)
    # The gate's u axis in NED is (-s, c, 0); its v axis is down, (0, 0, 1).
    return gate.center + np.column_stack([-s * u, c * u, v])


@dataclass(frozen=True)
class Observer:
    """Builds the observation of n drones flying through `gates` (in order),
    looking `lookahead` gates ahead."""

    gates: Sequence[Gate]
    lookahead: int = 2

    def __post_init__(self) -> None:
        gates = tuple(self.gates)
        if len(gates) == 0:
            raise ValueError("a course needs at least one gate")
        if self.lookahead < 1:
            raise ValueError("lookahead must be >= 1")
        object.__setattr__(self, "gates", gates)
        object.__setattr__(
            self, "_points", _read_only(np.array([opening_points(g) for g in gates]))
        )

    @property
    def size(self) -> int:
        return 3 + 9 + 3 + 3 * OPENING_POINTS * self.lookahead + ACTION_SIZE

    @property
    def space(self) -> BoxSpace:
        return BoxSpace(np.full(self.size, -np.inf), np.full(self.size, np.inf))

    def targets(self, next_gate: npt.ArrayLike) -> np.ndarray:
        """The gates each drone looks at (n, lookahead): its next gate and the ones
        after it, the last gate repeated where the course runs out (and after the
        finish, next_gate = -1)."""
        last = len(self.gates) - 1
        first = np.where(np.asarray(next_gate) < 0, last, next_gate)
        ahead = first[:, None] + np.arange(self.lookahead)
        return np.minimum(ahead, last)

    def __call__(
        self,
        states: np.ndarray,
        next_gate: npt.ArrayLike,
        previous_action: npt.ArrayLike,
    ) -> np.ndarray:
        """Observations (n, size) float64 for the states (n, 13), each drone's next
        gate (n,) (-1 after the finish) and its previous action (n, 4)."""
        n = len(states)
        rotation = quaternion.to_matrix(states[:, 6:10])  # (n, 3, 3) BODY -> NED
        to_body = np.swapaxes(rotation, 1, 2)
        velocity = matvec(to_body, states[:, 3:6])
        points = self._points[self.targets(next_gate)]  # (n, lookahead, 4, 3)
        relative = (points - states[:, None, None, 0:3]).reshape(n, -1, 3)
        in_body = matvec(to_body[:, None], relative)  # (n, lookahead * 4, 3)
        return np.column_stack(
            [
                velocity,
                rotation.reshape(n, 9),
                states[:, 10:13],
                in_body.reshape(n, -1),
                np.asarray(previous_action, dtype=float),
            ]
        )
