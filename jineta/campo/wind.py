"""Wind models: velocity of the air mass relative to the ground, in NED axes.

A wind model is a callable (t, position) -> wind velocity [m/s]: the time [s] and the
position in NED axes [m] go in, and a NED vector (north, east, down) comes out. Taking
time and position lets the interface grow into gusts and terrain-following profiles;
the models in this module only use the altitude, taken as -z like the density models.

Convention: the NED wind vector points where the air is going TO. Weather reports give
the direction the wind comes FROM; meteorological_wind() converts between the two.
"""

from dataclasses import dataclass

import numpy as np


def meteorological_wind(speed: float, direction_from_deg: float) -> np.ndarray:
    """NED wind vector [m/s] from a weather report (horizontal wind, no vertical part).

    direction_from_deg is where the wind comes FROM, clockwise from north:
    0 = from the north, 90 = from the east, 270 = from the west.
    """
    angle = np.radians(direction_from_deg)
    return np.array([-speed * np.cos(angle), -speed * np.sin(angle), 0.0])


@dataclass
class ConstantWind:
    """Uniform wind: the same NED velocity at every time and place."""

    velocity: np.ndarray  # [north, east, down] [m/s]; an updraft has negative down

    def __post_init__(self) -> None:
        self.velocity = np.asarray(self.velocity, dtype=float)
        if self.velocity.shape != (3,):
            raise ValueError("velocity must be a 3-vector [north, east, down]")

    def __call__(self, t: float, position: np.ndarray) -> np.ndarray:
        """Return the wind velocity [m/s]; it does not depend on time or position."""
        return self.velocity.copy()


@dataclass
class ProfileWind:
    """Wind that changes with altitude, linearly interpolated between table points.

    Outside the table the wind keeps the value of the nearest end point. The NED
    components are interpolated, not speed and direction: a wind that turns from 350
    to 010 degrees with height passes through north instead of through south.
    """

    altitudes: np.ndarray  # table altitudes, strictly increasing [m]
    velocities: np.ndarray  # wind at each altitude, shape (n, 3), NED [m/s]

    def __post_init__(self) -> None:
        self.altitudes = np.asarray(self.altitudes, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        if self.altitudes.ndim != 1 or self.altitudes.size == 0:
            raise ValueError("altitudes must be a non-empty 1-D array")
        if self.velocities.shape != (self.altitudes.size, 3):
            raise ValueError("velocities must have shape (n, 3), one row per altitude")
        if np.any(np.diff(self.altitudes) <= 0.0):
            raise ValueError("altitudes must be strictly increasing")

    def __call__(self, t: float, position: np.ndarray) -> np.ndarray:
        """Return the interpolated wind velocity [m/s] at the position's altitude."""
        altitude = -position[2]  # NED: z points down
        return np.array(
            [
                np.interp(altitude, self.altitudes, self.velocities[:, k])
                for k in range(3)
            ]
        )
