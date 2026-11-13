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
    the nearest `nearest` surfaces (optional, phase 11): the
      vector to the closest point of each, in body axes, the
      closest first                                              3 per surface
    1 once every gate is crossed, 0 before (optional, phase 12) 1

The surroundings. On a course the policy has flown many times it can learn where
the walls and obstacles are; on a course it has never seen it has to see them.
The surfaces are the room's six faces (floor, ceiling, four walls), every
obstacle and every gate's frame; for each, the closest point to the drone's
centre of mass (on a face, the foot of the perpendicular; on a shape, the point
p - d grad(d) of its signed distance d, whose gradient is taken by central
differences), and the `nearest` closest are given, nearest first. Like the gate
points, they are relative to the drone, in body axes.

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
from jineta.campo.indoor import Room
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


GRADIENT_STEP = 1e-5  # [m], for the gradients of the signed distances


def closest_points(shape, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The closest points (n, 3) of a shape (anything with `signed_distance`) to
    points (n, 3) outside it, and their distances (n,): p - d grad(d), the
    gradient by central differences and scaled to length 1 (exactly 1 for a
    Euclidean signed distance, outside the shape)."""
    distance = np.asarray(shape.signed_distance(points), dtype=float)
    gradient = np.empty_like(points)
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = GRADIENT_STEP
        up = shape.signed_distance(points + step)
        down = shape.signed_distance(points - step)
        gradient[:, axis] = (up - down) / (2.0 * GRADIENT_STEP)
    length = np.sqrt(
        gradient[:, 0] * gradient[:, 0]
        + gradient[:, 1] * gradient[:, 1]
        + gradient[:, 2] * gradient[:, 2]
    )
    gradient /= np.maximum(length, 1e-12)[:, None]
    return points - distance[:, None] * gradient, distance


@dataclass(frozen=True)
class Surroundings:
    """The `nearest` surfaces around n drones in a room with obstacles and gate
    frames (`shapes`: anything with `signed_distance`)."""

    room: Room
    shapes: Sequence = ()
    nearest: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "shapes", tuple(self.shapes))
        if not 1 <= self.nearest <= 6 + len(self.shapes):
            raise ValueError("nearest must be between 1 and the number of surfaces")

    @property
    def size(self) -> int:
        return 3 * self.nearest

    def points(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The closest point of every surface (n, surfaces, 3) and its distance
        (n, surfaces): the room's faces (floor, ceiling, walls x = 0, x = length,
        y = 0, y = width), then the shapes in order."""
        length, width, height = self.room.size
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        n = len(positions)
        faces = np.repeat(positions[:, None, :], 6, axis=1)
        faces[:, 0, 2] = 0.0  # floor
        faces[:, 1, 2] = -height  # ceiling
        faces[:, 2, 0] = 0.0
        faces[:, 3, 0] = length
        faces[:, 4, 1] = 0.0
        faces[:, 5, 1] = width
        distances = [-z, z + height, x, length - x, y, width - y]
        points = [faces]
        for shape in self.shapes:
            point, distance = closest_points(shape, positions)
            points.append(point[:, None, :])
            distances.append(distance)
        return np.concatenate(points, axis=1), np.column_stack(distances).reshape(n, -1)

    def __call__(self, positions: np.ndarray, to_body: np.ndarray) -> np.ndarray:
        """The vectors to the nearest surfaces' closest points, nearest first, in
        body axes (n, 3 * nearest), for positions (n, 3) and the rotations NED
        -> BODY (n, 3, 3)."""
        points, distances = self.points(positions)
        order = np.argsort(distances, axis=1, kind="stable")[:, : self.nearest]
        chosen = np.take_along_axis(points, order[:, :, None], axis=1)
        relative = chosen - positions[:, None, :]
        return matvec(to_body[:, None], relative).reshape(len(positions), -1)


@dataclass(frozen=True)
class Observer:
    """Builds the observation of n drones flying through `gates` (in order),
    looking `lookahead` gates ahead, and seeing the nearest surfaces if given
    `surroundings`; with `finish_flag` one more number, 1 once every gate is
    crossed (phase 12: the drone must then stop) and 0 before."""

    gates: Sequence[Gate]
    lookahead: int = 2
    surroundings: Surroundings | None = None
    finish_flag: bool = False

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
        around = 0 if self.surroundings is None else self.surroundings.size
        flag = 1 if self.finish_flag else 0
        return (
            3 + 9 + 3 + 3 * OPENING_POINTS * self.lookahead + ACTION_SIZE + around
            + flag
        )  # fmt: skip

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
        parts = [
            velocity,
            rotation.reshape(n, 9),
            states[:, 10:13],
            in_body.reshape(n, -1),
            np.asarray(previous_action, dtype=float),
        ]
        if self.surroundings is not None:
            parts.append(self.surroundings(states[:, 0:3], to_body))
        if self.finish_flag:
            parts.append((np.asarray(next_gate) < 0).astype(float))
        return np.column_stack(parts)
