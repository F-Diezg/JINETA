"""The classical baseline: the velocity autopilot flown along a planned path
through the gate centres. Its best lap is the time a policy has to beat.

The path. A polyline from the point above the start through every gate: for each
gate an entry point `approach` metres before its centre along its heading, the
centre, and an exit point `approach` metres after it, so the drone crosses every
gate straight, along its heading, through the middle. (The last exit point may lie
past a wall: the lap ends at the last gate, before the drone gets there.)

The guidance (a pure-pursuit path follower, the classical way of following a
path):
  - progress: each drone keeps its distance along the path s, the closest point of
    the path to it, searched only a few segments ahead of where it was (so a path
    that comes back near itself cannot make it jump) and never going back;
  - direction: the velocity command points from the drone at the point of the path
    `lookahead` metres further on. A drone off the path aims back at it, and
    corners are cut smoothly: this is the position loop the velocity autopilot
    lacks (loose end 10), and it looks ahead to the next gate (loose end 28);
  - speed: `cruise` on straight stretches, limited before each corner so that the
    lateral acceleration in the turn stays below `lateral_accel` (a corner of
    angle theta cut by an arc tangent to both segments `lookahead` from the corner
    has radius R = lookahead / tan(theta / 2), and v^2 / R <= lateral_accel), and
    braking towards it at no more than `braking_accel`.
  - take-off: straight up until `takeoff_height` above the floor, then the path;
    the nose follows the heading of the next gate.

The settings are one value per drone, so the batch engine tries many at once;
`sweep` keeps the fastest that finishes.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import numpy.typing as npt

from jineta.campo.course import Course
from jineta.escuadron.contact import BatchGroundContact
from jineta.escuadron.control import BatchAutopilot, Drones, _rows
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.escuadron.referee import FINISHED, batch_referee
from jineta.escuadron.simulation import BatchFlight
from jineta.jinete.contact import resting_height
from jineta.jinete.profiles import MULTIROTOR_PROFILES, MultirotorProfile

WINDOW = 4  # segments searched ahead for the closest point of the path


@dataclass(frozen=True)
class BaselineSettings:
    """Settings of the path follower, each one value or one per drone (n,).

    cruise [m/s]: speed on straight stretches.
    lateral_accel [m/s^2]: largest lateral acceleration planned in a corner.
    braking_accel [m/s^2]: largest deceleration planned before a corner.
    lookahead [m]: how far ahead on the path the drone aims.
    approach [m]: entry and exit points before and after each gate centre.
    takeoff_height [m]: climb straight up this far before following the path.
    """

    cruise: npt.ArrayLike = 4.0
    lateral_accel: npt.ArrayLike = 4.0
    braking_accel: npt.ArrayLike = 4.0
    lookahead: npt.ArrayLike = 1.5
    approach: float = 1.0
    takeoff_height: float = 0.5

    def __post_init__(self) -> None:
        for name in ("cruise", "lateral_accel", "braking_accel", "lookahead"):
            value = np.array(getattr(self, name), dtype=float)
            if value.ndim > 1 or not np.all(value > 0.0):
                raise ValueError(f"{name} must be > 0 (one value or one per drone)")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if not self.approach > 0.0:
            raise ValueError("approach must be > 0")
        if not self.takeoff_height > 0.0:
            raise ValueError("takeoff_height must be > 0")

    def per_drone(self, n: int) -> dict[str, np.ndarray]:
        """The four per-drone settings as arrays (n,)."""
        names = ("cruise", "lateral_accel", "braking_accel", "lookahead")
        return {k: np.broadcast_to(getattr(self, k), (n,)).copy() for k in names}


@dataclass(frozen=True)
class GatePath:
    """The polyline through a course's gates: `points` (m, 3) NED [m], each
    segment's start, unit direction and length, the distance along the path at
    each point, and the turn angle at each inner point [rad]."""

    points: np.ndarray

    def __post_init__(self) -> None:
        points = np.array(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("a path needs at least two points (m, 3)")
        steps = np.diff(points, axis=0)
        lengths = np.linalg.norm(steps, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError("two consecutive points of a path coincide")
        directions = steps / lengths[:, None]
        cosines = np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1)
        for name, value in (
            ("points", points),
            ("lengths", lengths),
            ("directions", directions),
            ("distance", np.concatenate([[0.0], np.cumsum(lengths)])),
            ("turns", np.arccos(cosines)),
        ):
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @classmethod
    def through(cls, course: Course, approach: float, takeoff_height: float):
        """The path of a course: above the start, then entry, centre and exit of
        every gate."""
        start = np.array([course.start[0], course.start[1], -takeoff_height])
        points = [start]
        for gate in course.gates:
            heading = np.array([np.cos(gate.yaw), np.sin(gate.yaw), 0.0])
            points += [
                gate.center - approach * heading,
                gate.center,
                gate.center + approach * heading,
            ]
        return cls(np.array(points))

    @property
    def length(self) -> float:
        return float(self.distance[-1])

    def point_at(self, s: np.ndarray) -> np.ndarray:
        """Points (n, 3) at distances s (n,) along the path; past either end the
        path goes on straight along its first or last segment."""
        segment = np.clip(
            np.searchsorted(self.distance, s, side="right") - 1,
            0,
            len(self.lengths) - 1,
        )
        along = s - self.distance[segment]
        return self.points[segment] + along[:, None] * self.directions[segment]

    def closest(
        self, positions: np.ndarray, first: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """For each position (n, 3), the closest point of segments first ..
        first + WINDOW - 1: its distance along the path (n,) and its segment (n,)."""
        last = len(self.lengths) - 1
        segments = np.minimum(first[:, None] + np.arange(WINDOW), last)  # (n, W)
        start = self.points[segments]
        direction = self.directions[segments]
        offset = positions[:, None, :] - start
        along = np.clip(np.sum(offset * direction, axis=2), 0.0, self.lengths[segments])
        nearest = start + along[:, :, None] * direction
        gap = np.sum((positions[:, None, :] - nearest) ** 2, axis=2)
        best = np.argmin(gap, axis=1)
        rows = np.arange(len(positions))
        chosen = segments[rows, best]
        return self.distance[chosen] + along[rows, best], chosen


@dataclass
class PathGuidance:
    """The path follower for n drones: velocity and heading commands from the
    states (see the module). The headings are those of the course's gates."""

    course: Course
    settings: BaselineSettings
    n: int
    path: GatePath | None = None  # by default the course's (GatePath.through)
    progress: np.ndarray = field(init=False)  # (n,) distance along the path [m]
    segment: np.ndarray = field(init=False)  # (n,) segment of that point
    climbed: np.ndarray = field(init=False)  # (n,)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n (number of drones) must be >= 1")
        if self.path is None:
            self.path = GatePath.through(
                self.course, self.settings.approach, self.settings.takeoff_height
            )
        s = self.settings.per_drone(self.n)
        self._cruise, self._lateral = s["cruise"], s["lateral_accel"]
        self._braking, self._lookahead = s["braking_accel"], s["lookahead"]
        # Corners at the inner points; the speed allowed through each (n, corners).
        turns = self.path.turns
        with np.errstate(divide="ignore"):
            radius = self._lookahead[:, None] / np.tan(turns / 2.0)
        self._corner_speed = np.minimum(
            np.sqrt(self._lateral[:, None] * radius), self._cruise[:, None]
        )
        self._corner_at = self.path.distance[1:-1]
        # The gate whose heading each segment holds: segment 0 goes to gate 0's
        # entry; for gate g, entry -> centre (3g + 1), centre -> exit (3g + 2)
        # and exit -> the next gate's entry (3g + 3, the next gate's heading).
        lead = np.arange(len(self.path.lengths)) // 3
        self._yaw_of_segment = np.array([g.yaw for g in self.course.gates])[lead]
        self.progress = np.zeros(self.n)
        self.segment = np.zeros(self.n, dtype=int)
        self.climbed = np.zeros(self.n, dtype=bool)

    def reset(self, drones: Drones = None) -> None:
        """These drones (all by default) start a lap standing at the start."""
        rows = _rows(drones)
        self.progress[rows] = 0.0
        self.segment[rows] = 0
        self.climbed[rows] = False

    def speed_limit(self, s: np.ndarray) -> np.ndarray:
        """The planned speed (n,) [m/s] at distances s (n,) along the path."""
        ahead = self._corner_at[None, :] + self._lookahead[:, None] - s[:, None]
        to_brake = np.maximum(ahead - 2.0 * self._lookahead[:, None], 0.0)
        allowed = np.sqrt(
            self._corner_speed**2 + 2.0 * self._braking[:, None] * to_brake
        )
        allowed = np.where(ahead >= 0.0, allowed, np.inf)  # corners behind: none
        return np.minimum(self._cruise, np.min(allowed, axis=1, initial=np.inf))

    def commands(self, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Velocity commands (n, 3) NED [m/s] and headings (n,) [rad]."""
        positions = states[:, 0:3]
        self.climbed |= -positions[:, 2] > self.settings.takeoff_height
        s, segment = self.path.closest(positions, self.segment)
        forward = s > self.progress
        self.progress = np.where(forward, s, self.progress)
        self.segment = np.where(forward, segment, self.segment)
        target = self.path.point_at(self.progress + self._lookahead)
        to_target = target - positions
        distance = np.sqrt(np.sum(to_target * to_target, axis=1))
        speed = self.speed_limit(self.progress)
        follow = speed[:, None] * to_target / np.maximum(distance, 1e-9)[:, None]
        velocity = np.where(self.climbed[:, None], follow, [0.0, 0.0, -1.0])
        return velocity, self._yaw_of_segment[self.segment]


@dataclass
class BaselinePilot:
    """The baseline for n drones: the path guidance driving the batch velocity
    autopilot. Callable as the controller of a `BatchFlight`."""

    guidance: PathGuidance
    autopilot: BatchAutopilot

    def reset(self, drones: Drones = None) -> None:
        """These drones start a lap standing on the floor."""
        rows = _rows(drones)
        self.guidance.reset(rows)
        self.autopilot.reset(rows)
        self.autopilot.landed[rows] = True

    def __call__(self, clock: np.ndarray, states: np.ndarray) -> None:
        velocity, yaw = self.guidance.commands(states)
        self.autopilot.velocity_command = velocity
        self.autopilot.yaw_command = yaw
        self.autopilot(clock, states)


def baseline_flight(
    course: Course,
    settings: BaselineSettings,
    n: int,
    profile: MultirotorProfile = MULTIROTOR_PROFILES["iris"],
    dt: float = 0.01,
) -> tuple[BatchFlight, BaselinePilot]:
    """n drones standing at the course's start, landed, with the batch referee
    and the baseline pilot."""
    params, gains, gear, hitbox = profile
    vehicles = BatchMultirotor(params, n, contact=BatchGroundContact(gear))
    referee = batch_referee(course, hitbox.centers, hitbox.radii, gear.points, n)
    start = course.start_state(resting_height(gear, params.mass))
    flight = BatchFlight(vehicles, referee, dt, np.tile(start, (n, 1)))
    autopilot = BatchAutopilot(vehicles, gains, dt, landed=True)
    return flight, BaselinePilot(PathGuidance(course, settings, n), autopilot)


def fly_baseline(
    course: Course, settings: BaselineSettings, n: int = 1, **kwargs
) -> tuple[np.ndarray, np.ndarray, np.ndarray, BatchFlight]:
    """One lap for each of n drones (each with its own settings); returns their
    statuses (n,), lap times (n,) [s] (nan unless finished), the closest each came
    to a wall, the ceiling, an obstacle or a gate frame during its lap (n,) [m],
    at the samples, and the flight."""
    flight, pilot = baseline_flight(course, settings, n, **kwargs)
    referee = flight.referee
    clearance = np.full(n, np.inf)
    limit = round((course.time_limit + 1.0) / flight.dt)
    for _ in range(limit):
        flying = ~referee.ended
        flight.step(pilot)
        margins = referee.monitor.margins(flight.states)[:, :-1].min(axis=1)
        clearance = np.where(flying, np.minimum(clearance, margins), clearance)
        if np.all(referee.ended):
            break
    return referee.status.copy(), referee.lap_time, clearance, flight


@dataclass(frozen=True)
class SweepResult:
    """The best settings found and the lap they fly; and, for every setting tried
    (one row each), its values, whether it finished robustly and its lap time."""

    best: BaselineSettings
    lap_time: float
    tried: np.ndarray  # (k, 5): approach, cruise, lateral, braking, lookahead
    robust: np.ndarray  # (k,) finished, and so did every perturbed copy
    lap_times: np.ndarray  # (k,) [s] of the setting itself, nan unless finished


def sweep(
    course: Course,
    approach: npt.ArrayLike = (1.0, 1.5, 2.0),
    cruise: npt.ArrayLike = (4.0, 5.0, 6.0, 8.0),
    lateral_accel: npt.ArrayLike = (4.0, 6.0, 8.0, 10.0),
    braking_accel: npt.ArrayLike = (2.0, 3.0, 5.0),
    lookahead: npt.ArrayLike = (2.0, 2.5, 3.0),
    copies: int = 4,
    spread: float = 0.02,
    clearance: float = 0.05,
    seed: int = 0,
    **kwargs,
) -> SweepResult:
    """Fly every combination of these settings (one batch per approach) and keep
    the fastest lap that finishes robustly: never closer than `clearance` [m] to a
    wall, an obstacle or a frame, and the same with its four per-drone values each
    changed by up to +-spread (`copies` random combinations, seeded). A lap that
    only just clears a frame, which a 2 % change turns into a crash, is luck, not a
    time a controller can be trusted to fly."""
    rng = np.random.default_rng(seed)
    grid = np.array(list(product(cruise, lateral_accel, braking_accel, lookahead)))
    k = len(grid)
    rows, robust, times = [], [], []
    for distance in np.atleast_1d(approach):
        factors = np.ones((k, copies + 1, 4))
        factors[:, 1:] = rng.uniform(1.0 - spread, 1.0 + spread, (k, copies, 4))
        values = (grid[:, None, :] * factors).reshape(-1, 4)
        settings = BaselineSettings(*values.T, approach=float(distance), **kwargs)
        statuses, laps, gaps, _ = fly_baseline(course, settings, len(values))
        safe = (statuses == FINISHED) & (gaps >= clearance)
        rows.append(np.column_stack([np.full(k, distance), grid]))
        robust.append(np.all(safe.reshape(k, copies + 1), axis=1))
        times.append(laps.reshape(k, copies + 1)[:, 0])
    tried, robust, times = (
        np.vstack(rows),
        np.concatenate(robust),
        np.concatenate(times),
    )
    if not np.any(robust):
        raise RuntimeError("no setting of the baseline finished the course robustly")
    i = int(np.nanargmin(np.where(robust, times, np.nan)))
    best = BaselineSettings(*tried[i, 1:], approach=float(tried[i, 0]), **kwargs)
    return SweepResult(best, float(times[i]), tried, robust, times)
