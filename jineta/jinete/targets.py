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


def track_positions(track: TargetTrack, times: npt.ArrayLike) -> np.ndarray:
    """Positions of a target track at each time, as an (n, 3) array."""
    return np.array([track(t)[0] for t in np.asarray(times, dtype=float)])
