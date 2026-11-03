"""Tracks: a Kalman filter that follows one target from spherical measurements.

The state is a constant-velocity point, x = [N, E, D, vN, vE, vD] (NED), with the
target's acceleration as continuous white noise of density q (`kalman.py`). The
measurement is not a linear function of that state: a radar (or later a seeker)
measures z = h(x) = (range, azimuth, elevation) of the target as seen from its
site. The extended Kalman filter (EKF) linearises h around the predicted state:

    h(x) ~ h(x_pred) + H (x - x_pred),    H = dh/dx at x_pred  (3 x 6),

the same first-order step as the linearisation of the aircraft dynamics in
`jinete/stability.py`, here for the measurement. With it the ordinary update
applies, with the innovation y = z - h(x_pred) (its azimuth wrapped, so 179 deg
against -179 deg is 2 deg, not 358). The approximation is good while the
uncertainty is small compared with the range: the curvature of h across the
uncertainty is then negligible.

A track starts from the first measurement of its target (one-point
initialisation): the position is the measured point, with the covariance of that
point, and the velocity is zero with a standard deviation `speed_prior` on each
axis, what a target of that kind could be doing. From the second measurement on it
is a normal predict + update. (Starting from two measurements, with the velocity
from their difference, was the first idea: 0.1 s apart and 1 km away, that velocity
is uncertain by ~250 m/s, and the first updates were overconfident for ~2 s.)

Between measurements the track is predicted to whatever time is asked
(`estimate(t)`), from its last update and without changing it: with continuous
white-noise acceleration, predicting once or in many small steps gives the same.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.atalaya import kalman
from jineta.atalaya.radar import Measurement, spherical, wrap_angle


def measurement_jacobian(offset: npt.ArrayLike) -> np.ndarray:
    """d(range, azimuth, elevation) / d(N, E, D) of a NED offset from the site.

    With rho the horizontal distance and r the range:
        range:     [N, E, D] / r
        azimuth:   [-E, N, 0] / rho^2
        elevation: [D N / rho, D E / rho, -rho] / r^2
    Undefined straight above the site (rho = 0), where the azimuth has no meaning.
    """
    north, east, down = np.asarray(offset, dtype=float)
    rho2 = north**2 + east**2
    if rho2 == 0.0:
        raise ValueError("the azimuth is undefined straight above or below the site")
    rho = np.sqrt(rho2)
    r2 = rho2 + down**2
    r = np.sqrt(r2)
    return np.array(
        [
            [north / r, east / r, down / r],
            [-east / rho2, north / rho2, 0.0],
            [down * north / (rho * r2), down * east / (rho * r2), -rho / r2],
        ]
    )


def state_jacobian(site: npt.ArrayLike, x: npt.ArrayLike) -> np.ndarray:
    """H of the EKF: d h / d x for the 6-state (the velocities are not measured)."""
    offset = np.asarray(x, dtype=float)[:3] - np.asarray(site, dtype=float)
    return np.hstack([measurement_jacobian(offset), np.zeros((3, 3))])


def position_covariance(measurement: Measurement) -> np.ndarray:
    """The NED covariance of the point a measurement gives (first order):
    J R J^T, with J = d(N, E, D) / d(range, azimuth, elevation) at the measurement.

    Its long axes are across the line of sight: range * sigma_azimuth sideways and
    range * sigma_elevation up and down, against sigma_range along it.
    """
    r, az, el = measurement.values
    ce, se, ca, sa = np.cos(el), np.sin(el), np.cos(az), np.sin(az)
    J = np.array(
        [
            [ce * ca, -r * ce * sa, -r * se * ca],
            [ce * sa, r * ce * ca, -r * se * sa],
            [-se, 0.0, -r * ce],
        ]
    )
    return J @ measurement.noise_covariance @ J.T


@dataclass
class Track:
    """The filtered estimate of one target's position and velocity.

    Args:
        target: the name of the target it follows; it only takes its measurements.
        process_noise: q, spectral density of the target's acceleration [m^2/s^3].
            A rough guide: over one second the model allows velocity changes of
            about sqrt(q) m/s.
        speed_prior: standard deviation of each velocity component when the track
            starts [m/s]: how fast a target of this kind can plausibly fly.
    """

    target: str
    process_noise: float
    speed_prior: float
    x: np.ndarray | None = field(default=None, init=False)  # (6,) at `time`
    P: np.ndarray | None = field(default=None, init=False)  # (6, 6) at `time`
    time: float | None = field(default=None, init=False)  # of the last update [s]
    updates: int = field(default=0, init=False)  # measurements used so far

    def __post_init__(self) -> None:
        if not self.process_noise >= 0.0:
            raise ValueError("process_noise must be >= 0")
        if not self.speed_prior > 0.0:
            raise ValueError("speed_prior must be > 0")

    @property
    def ready(self) -> bool:
        """True once it has an estimate (after its first measurement)."""
        return self.x is not None

    def update(self, measurement: Measurement) -> None:
        """Use one measurement of this track's target."""
        if measurement.target != self.target:
            raise ValueError(
                f"a measurement of {measurement.target!r} given to the track of "
                f"{self.target!r}"
            )
        if self.time is not None and not measurement.time > self.time:
            raise ValueError("measurements must come in increasing time")
        if self.x is None:
            self._start(measurement)
        else:
            self._correct(measurement)
        self.updates += 1

    def _start(self, measurement: Measurement) -> None:
        """One-point initialisation: the measured point with its covariance, and a
        still target with the speed prior on each velocity component."""
        self.x = np.concatenate([measurement.position, np.zeros(3)])
        self.P = np.block(
            [
                [position_covariance(measurement), np.zeros((3, 3))],
                [np.zeros((3, 3)), self.speed_prior**2 * np.eye(3)],
            ]
        )
        self.time = measurement.time

    def _correct(self, measurement: Measurement) -> None:
        """EKF: predict to the measurement, linearise h there, update."""
        x, P = self.estimate(measurement.time)
        site = measurement.site
        H = state_jacobian(site, x)
        predicted = np.array(spherical(x[:3] - site))
        innovation = measurement.values - predicted
        innovation[1] = wrap_angle(innovation[1])
        # kalman.update takes a linear measurement z = H x; giving it
        # H x + innovation makes its own innovation (z - H x) the EKF one.
        x, P, _, _ = kalman.update(
            x, P, H @ x + innovation, H, measurement.noise_covariance
        )
        self.x, self.P, self.time = x, P, measurement.time

    def estimate(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """(x, P) predicted from the last update to time t (the track does not
        change). t may not be before the last update."""
        if self.x is None:
            raise ValueError(f"the track of {self.target!r} has not started yet")
        dt = t - self.time
        if dt < -1e-9:
            raise ValueError("cannot estimate before the last update")
        dt = max(dt, 0.0)
        F = kalman.constant_velocity_transition(dt)
        Q = kalman.white_noise_acceleration(dt, self.process_noise)
        return kalman.predict(self.x, self.P, F, Q)

    def position(self, t: float) -> np.ndarray:
        """Estimated position at time t, NED [m]."""
        return self.estimate(t)[0][:3]

    def velocity(self, t: float) -> np.ndarray:
        """Estimated velocity at time t, NED [m/s]."""
        return self.estimate(t)[0][3:]
