"""Wind models: velocity of the air mass relative to the ground, in NED axes.

A wind model is a callable (t, position) -> wind velocity [m/s]: the time [s] and the
position in NED axes [m] go in, and a NED vector (north, east, down) comes out. Taking
time and position lets the interface grow into gusts and terrain-following profiles;
the profile models in this module only use the altitude, taken as -z like the density
models.

Models: ConstantWind, ProfileWind (tabulated), PowerLawWind (boundary layer),
DiscreteGust (1-cosine bump in time) and CombinedWind (sum of models, for example a
mean wind plus a gust).

Convention: the NED wind vector points where the air is going TO. Weather reports give
the direction the wind comes FROM; meteorological_wind() converts between the two.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jineta.core.environment import WindModel


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


@dataclass
class PowerLawWind:
    """Boundary-layer profile: speed = v_ref * (height / h_ref) ** exponent.

    The direction is constant. The ground slows the air down, so the wind is zero at
    height 0 and grows with height up to the gradient height; above it the speed stays
    constant. Typical exponents: ~0.11 over open water, 1/7 over open land, 0.25 or
    more over cities. An exponent of 0 gives a uniform wind. Within the last metres
    above the ground it is only a rough fit: its slope is infinite at height 0.
    """

    reference_speed: float  # wind speed at the reference height [m/s]
    direction_from_deg: float  # where the wind comes FROM, clockwise from north [deg]
    reference_height: float = 10.0  # [m]
    exponent: float = 1.0 / 7.0  # [-]
    gradient_height: float = 500.0  # above this height the speed stays constant [m]

    def __post_init__(self) -> None:
        if self.reference_speed < 0.0:
            raise ValueError("reference_speed must be >= 0")
        if self.reference_height <= 0.0:
            raise ValueError("reference_height must be > 0")
        if self.exponent < 0.0:
            raise ValueError("exponent must be >= 0")
        if self.gradient_height <= 0.0:
            raise ValueError("gradient_height must be > 0")

    def __call__(self, t: float, position: np.ndarray) -> np.ndarray:
        """Return the wind velocity [m/s] at the position's altitude."""
        altitude = -position[2]  # NED: z points down
        height = min(max(altitude, 0.0), self.gradient_height)
        speed = self.reference_speed * (height / self.reference_height) ** self.exponent
        return meteorological_wind(speed, self.direction_from_deg)


@dataclass
class DiscreteGust:
    """Discrete "1-cosine" gust: a smooth bump of wind that hits everywhere at once.

    During [start_time, start_time + duration] the wind is
    peak * 0.5 * (1 - cos(2*pi*(t - start_time) / duration)), and zero outside. It
    starts and ends with zero value and zero slope and reaches the full peak in the
    middle. The gust is uniform in space; a gust that travels across the field would
    use the position as well.
    """

    peak: np.ndarray  # wind velocity at the middle of the gust, NED [m/s]
    start_time: float  # [s]
    duration: float  # [s]

    def __post_init__(self) -> None:
        self.peak = np.asarray(self.peak, dtype=float)
        if self.peak.shape != (3,):
            raise ValueError("peak must be a 3-vector [north, east, down]")
        if self.duration <= 0.0:
            raise ValueError("duration must be > 0")

    def __call__(self, t: float, position: np.ndarray) -> np.ndarray:
        """Return the gust velocity [m/s] at time t; it does not depend on position."""
        elapsed = t - self.start_time
        if elapsed < 0.0 or elapsed > self.duration:
            return np.zeros(3)
        shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * elapsed / self.duration))
        return self.peak * shape


@dataclass
class CombinedWind:
    """Sum of several wind models, for example a mean wind profile plus a gust."""

    models: Sequence[WindModel]

    def __post_init__(self) -> None:
        self.models = list(self.models)

    def __call__(self, t: float, position: np.ndarray) -> np.ndarray:
        """Return the sum of the wind velocities [m/s] of all the models."""
        total = np.zeros(3)
        for model in self.models:
            total = total + model(t, position)
        return total
