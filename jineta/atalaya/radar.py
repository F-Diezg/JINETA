"""A ground radar: it measures where a target is, as seen from its site, with noise.

A radar does not measure north, east and down. It measures in spherical
coordinates around its site:

  - range r: the straight-line distance to the target [m];
  - azimuth: the direction on the map, clockwise from North [rad], in (-pi, pi];
  - elevation: the angle above the horizontal [rad], in [-pi/2, pi/2].

Each one comes with its own Gaussian noise, independent of the others. The angle
noise is what matters far away: an angle error sigma_theta moves the measured point
sideways by about r * sigma_theta (0.5 deg is 0.9 m at 100 m and 8.7 m at 1 km),
while the range error stays the same at every distance. So the cloud of measured
points is not a ball but a thin, curved plate across the line of sight, and it
grows with the range.

The radar measures at its own rate (e.g. 10 Hz), slower than the simulation (100 Hz):
it is due at t = 0, 1/rate, 2/rate, ... and measures at the first simulation step at
or after each of those times, with the truth of that step. Only targets within its
maximum range are measured. It knows which target each measurement belongs to (no
data association problem), always detects what is in range and has no false alarms.

The noise comes from a seeded random generator of the radar's own, so a run can be
repeated exactly (`reset`), and a different seed gives a different run (Monte Carlo,
phase 13).
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


def wrap_angle(angle: float) -> float:
    """The same angle in (-pi, pi]."""
    wrapped = math.remainder(angle, 2 * math.pi)  # in [-pi, pi]
    return math.pi if wrapped == -math.pi else wrapped


def spherical(offset: npt.ArrayLike) -> tuple[float, float, float]:
    """(range, azimuth, elevation) of a NED offset from the radar site.

    Azimuth atan2(east, north), clockwise from North; elevation atan2(up,
    horizontal distance), positive above the horizon (up = -down).
    """
    north, east, down = _vector(offset, "offset")
    horizontal = math.hypot(north, east)
    return (
        math.sqrt(horizontal**2 + down**2),
        math.atan2(east, north),
        math.atan2(-down, horizontal),
    )


def cartesian(distance: float, azimuth: float, elevation: float) -> np.ndarray:
    """The NED offset at (range, azimuth, elevation): the inverse of `spherical`."""
    horizontal = distance * math.cos(elevation)
    return np.array(
        [
            horizontal * math.cos(azimuth),
            horizontal * math.sin(azimuth),
            -distance * math.sin(elevation),
        ]
    )


def _vector(value: npt.ArrayLike, what: str) -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{what} must have 3 components (NED)")
    return array


@dataclass(frozen=True)
class RadarParams:
    """What a radar can do: its noise, rate and range (a profile, as data)."""

    range_noise: float  # standard deviation of the range [m]
    azimuth_noise: float  # standard deviation of the azimuth [rad]
    elevation_noise: float  # standard deviation of the elevation [rad]
    rate: float  # measurements per second [Hz]
    max_range: float  # farthest target it measures [m]

    def __post_init__(self) -> None:
        for name in ("range_noise", "azimuth_noise", "elevation_noise"):
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("rate", "max_range"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0")

    @property
    def noise_covariance(self) -> np.ndarray:
        """R of one measurement [range, azimuth, elevation]: the variances on the
        diagonal (the three noises are independent)."""
        return np.diag(
            [self.range_noise**2, self.azimuth_noise**2, self.elevation_noise**2]
        )


@dataclass(frozen=True)
class Measurement:
    """One radar measurement of one target."""

    target: str  # which target it belongs to (the radar knows)
    time: float  # when the target was there [s]
    site: np.ndarray  # (3,) NED position of the radar [m]
    range: float  # [m]
    azimuth: float  # [rad], clockwise from North, in (-pi, pi]
    elevation: float  # [rad], above the horizon
    noise_covariance: np.ndarray  # (3, 3) R in [range, azimuth, elevation]

    @property
    def values(self) -> np.ndarray:
        """[range, azimuth, elevation], the measurement vector z."""
        return np.array([self.range, self.azimuth, self.elevation])

    @property
    def position(self) -> np.ndarray:
        """Where the measurement puts the target, NED [m] (noise included)."""
        return self.site + cartesian(self.range, self.azimuth, self.elevation)


@dataclass
class Radar:
    """A radar on a site, with its own seeded noise.

    Args:
        params: what it can do (`RadarParams`).
        site: (3,) NED position [m].
        seed: seed of its random generator.
    """

    params: RadarParams
    site: np.ndarray
    seed: int = 0
    scans: int = field(default=0, init=False)  # index of the next scan
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        site = _vector(self.site, "site")
        site.flags.writeable = False
        self.site = site
        self.reset()

    def reset(self) -> None:
        """Back to the start: due at t = 0, the noise sequence from its beginning."""
        self.scans = 0
        self.rng = np.random.default_rng(self.seed)

    @property
    def next_time(self) -> float:
        """When the next scan is due [s]: scans / rate, computed from the count
        (adding the period again and again would pile up rounding errors)."""
        return self.scans / self.params.rate

    def due(self, t: float) -> bool:
        """True when a scan is due at time t (sample times carry rounding, hence
        the small tolerance, as in the launch rules)."""
        return t >= self.next_time - 1e-9

    def measure(
        self, target: str, t: float, position: npt.ArrayLike
    ) -> Measurement | None:
        """Measure one target at time t from its true position (None if out of
        range). Every call draws new noise; `scan` is what keeps the rate."""
        distance, azimuth, elevation = spherical(
            _vector(position, "position") - self.site
        )
        if distance > self.params.max_range:
            return None
        p = self.params
        noise = self.rng.standard_normal(3)
        return Measurement(
            target=target,
            time=t,
            site=self.site,
            range=distance + p.range_noise * noise[0],
            azimuth=wrap_angle(azimuth + p.azimuth_noise * noise[1]),
            elevation=elevation + p.elevation_noise * noise[2],
            noise_covariance=p.noise_covariance,
        )

    def scan(self, t: float, targets: Mapping[str, npt.ArrayLike]) -> list[Measurement]:
        """If a scan is due at t, measure every target in range, in the order
        given; otherwise nothing.

        Args:
            t: simulation time [s].
            targets: true positions by name, NED [m]. The order fixes the order in
                which the noise is drawn, so it must not change between runs.
        """
        if not self.due(t):
            return []
        # The next scan is the first one due after t (if a step is longer than the
        # radar period, the scans in between are skipped).
        self.scans = math.floor((t + 1e-9) * self.params.rate) + 1
        found = (self.measure(name, t, p) for name, p in targets.items())
        return [m for m in found if m is not None]
