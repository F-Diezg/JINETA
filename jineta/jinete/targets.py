"""Scripted targets: threats that follow a fixed path, given in closed form.

A target track is any callable `track(t) -> (position, velocity)`, both NED vectors
[m], [m/s]. The pursuer only needs to know where the target is at each time, so a
scripted path, a recorded trajectory or (later) a simulated vehicle can all be used
behind the same interface.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# time [s] -> (position NED [m], velocity NED [m/s])
TargetTrack = Callable[[float], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class StraightLineTarget:
    """Constant velocity: position = start + velocity * t."""

    start: np.ndarray  # NED position at t = 0 [m]
    velocity: np.ndarray  # NED [m/s]

    def __post_init__(self) -> None:
        for name in ("start", "velocity"):
            value = np.array(getattr(self, name), dtype=float)
            if value.shape != (3,):
                raise ValueError(f"{name} must have 3 components (NED)")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return self.start + self.velocity * t, self.velocity.copy()


@dataclass(frozen=True)
class TurningTarget:
    """Level turn at constant speed and turn rate (a circle seen from above).

    Heading psi(t) = heading + turn_rate * t, clockwise from North (turn_rate > 0
    turns right). Velocity = speed * [cos psi, sin psi, 0]; integrating it gives
    N = N0 + (V / w)(sin psi - sin psi0), E = E0 - (V / w)(cos psi - cos psi0), a
    circle of radius V / |w|. Altitude stays constant.
    """

    start: np.ndarray  # NED position at t = 0 [m]
    speed: float  # [m/s]
    heading: float  # at t = 0 [rad], clockwise from North
    turn_rate: float  # [rad/s], > 0 turns right; 0 gives a straight line

    def __post_init__(self) -> None:
        start = np.array(self.start, dtype=float)
        if start.shape != (3,):
            raise ValueError("start must have 3 components (NED)")
        if self.speed < 0.0:
            raise ValueError("speed must be >= 0")
        start.setflags(write=False)
        object.__setattr__(self, "start", start)

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        psi0 = self.heading
        psi = psi0 + self.turn_rate * t
        velocity = self.speed * np.array([np.cos(psi), np.sin(psi), 0.0])
        if self.turn_rate == 0.0:
            return self.start + velocity * t, velocity
        radius = self.speed / self.turn_rate  # signed: the side of the centre
        offset = radius * np.array(
            [np.sin(psi) - np.sin(psi0), -(np.cos(psi) - np.cos(psi0)), 0.0]
        )
        return self.start + offset, velocity


@dataclass(frozen=True)
class RecordedTrack:
    """A trajectory recorded at sample times (e.g. a simulated vehicle), replayed.

    Between two samples the position is the cubic (Hermite) curve that passes
    through both positions with both velocities: exact for any path that is a cubic
    in time over the step, with an error of order dt^4 otherwise. The velocity is the
    derivative of that curve, so position and velocity stay consistent. Asking for a
    time outside the recording is an error, not an extrapolation.

    A vehicle that does not react to anyone (a threat flying its own route) can be
    simulated first and then given to `engage` as a target like any other.
    """

    times: np.ndarray  # (n,) strictly increasing [s]
    positions: np.ndarray  # (n, 3) NED [m]
    velocities: np.ndarray  # (n, 3) NED [m/s]

    def __post_init__(self) -> None:
        times = np.array(self.times, dtype=float)
        positions = np.array(self.positions, dtype=float)
        velocities = np.array(self.velocities, dtype=float)
        n = times.size
        if times.ndim != 1 or n < 2:
            raise ValueError("times must be a 1-D array with at least 2 samples")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        for name, value in (("positions", positions), ("velocities", velocities)):
            if value.shape != (n, 3):
                raise ValueError(f"{name} must have shape ({n}, 3)")
        for name, value in (
            ("times", times),
            ("positions", positions),
            ("velocities", velocities),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def __call__(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        times = self.times
        tolerance = 1e-9 * max(1.0, abs(times[-1]))
        if not times[0] - tolerance <= t <= times[-1] + tolerance:
            raise ValueError(
                f"t = {t} s is outside the recording [{times[0]}, {times[-1]}] s"
            )
        i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, times.size - 2))
        h = times[i + 1] - times[i]
        s = (t - times[i]) / h
        p0, p1 = self.positions[i], self.positions[i + 1]
        v0, v1 = self.velocities[i], self.velocities[i + 1]
        # Hermite basis functions and their derivatives with respect to s
        h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
        h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
        d00, d10 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1
        d01, d11 = -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
        position = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
        velocity = (d00 * p0 + d01 * p1) / h + d10 * v0 + d11 * v1
        return position, velocity


def track_positions(track: TargetTrack, times: npt.ArrayLike) -> np.ndarray:
    """Positions of a target track at each time, as an (n, 3) array."""
    return np.array([track(t)[0] for t in np.asarray(times, dtype=float)])
