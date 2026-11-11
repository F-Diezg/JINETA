"""The referee of many laps at once: `campo.referee.Referee` on a batch.

n drones fly the same course, each its own lap: its own next gate, passes, status
and end. The rules are the reference's: gates count only in order and forward, a
crash ends the lap, and when two events fall in the same step the earlier one wins
(a gate crossed before a crash counts; the last gate crossed is a finish).

The status of each drone is a code in `status` (n,): FLYING, FINISHED, CRASHED or
TIMEOUT (`STATUS_NAMES` gives the reference's words). A drone whose lap has ended
is not judged any more (nor checked by the monitor) until it is reset.

Each drone's time can be its own: the referee takes one time for all or one per
drone (n,), the time since that drone's lap started.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.campo.course import Course
from jineta.campo.gates import CircularGate, Gate, RectangularGate
from jineta.campo.referee import GatePass
from jineta.escuadron.collision import BatchCollisionMonitor, Drones, _mask

FLYING, FINISHED, CRASHED, TIMEOUT = 0, 1, 2, 3
STATUS_NAMES = ("flying", "finished", "crashed", "timeout")


def _in_opening(gate: Gate, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Whether points of the gate's plane (u, v), arrays, are inside its opening
    (edges included), as the gates' own `in_opening`."""
    if isinstance(gate, RectangularGate):
        return (np.abs(u) <= gate.width / 2.0) & (np.abs(v) <= gate.height / 2.0)
    if isinstance(gate, CircularGate):
        return u * u + v * v <= gate.radius**2
    raise TypeError(f"unknown gate type {type(gate).__name__}")


def forward_crossings(
    gate: Gate, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moves start -> end (k, 3) through the gate's opening along its heading, as
    the forward crossings of `gate.crossing`: (crossed (k,), fraction (k,), point
    (k, 3)). The plane n = 0 is cut when n goes from < 0 to >= 0, at the fraction
    n0 / (n0 - n1), and the point where it is cut must be in the opening."""
    a = gate.to_local(start)
    b = gate.to_local(end)
    forward = (a[:, 0] < 0.0) & (0.0 <= b[:, 0])
    # Moves that do not cut the plane get a meaningless fraction (or 0 / 0): it
    # is computed, and thrown away by `forward`.
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = a[:, 0] / (a[:, 0] - b[:, 0])
        local = a + fraction[:, None] * (b - a)
        point = start + fraction[:, None] * (end - start)
    crossed = forward & _in_opening(gate, local[:, 1], local[:, 2])
    return crossed, fraction, point


@dataclass
class BatchReferee:
    """Judges the laps of the monitor's n drones through `gates` (in order) with a
    `time_limit` [s]. Callable once per step with the time(s) and the states
    (n, 13); returns which laps have ended (n,)."""

    gates: Sequence[Gate]
    monitor: BatchCollisionMonitor
    time_limit: float
    status: np.ndarray = field(init=False)  # (n,) codes
    end_time: np.ndarray = field(init=False)  # (n,) [s], nan while flying
    passed: np.ndarray = field(init=False)  # (n,) gates crossed so far
    pass_time: np.ndarray = field(init=False)  # (n, gates) [s]
    pass_point: np.ndarray = field(init=False)  # (n, gates, 3) [m]

    def __post_init__(self) -> None:
        self.gates = tuple(self.gates)
        if len(self.gates) == 0:
            raise ValueError("a lap needs at least one gate")
        if not self.time_limit > 0.0:
            raise ValueError("time_limit must be > 0")
        n, gates = self.monitor.n, len(self.gates)
        self.status = np.full(n, FLYING)
        self.end_time = np.full(n, np.nan)
        self.passed = np.zeros(n, dtype=int)
        self.pass_time = np.full((n, gates), np.nan)
        self.pass_point = np.full((n, gates, 3), np.nan)
        self._has_previous = np.zeros(n, dtype=bool)
        self._previous_time = np.zeros(n)
        self._previous_position = np.zeros((n, 3))

    @property
    def n(self) -> int:
        return self.monitor.n

    @property
    def ended(self) -> np.ndarray:
        """Which laps have ended (n,)."""
        return self.status != FLYING

    @property
    def next_gate(self) -> np.ndarray:
        """Index of each drone's next gate (n,); -1 once every gate is crossed."""
        return np.where(self.passed < len(self.gates), self.passed, -1)

    @property
    def lap_time(self) -> np.ndarray:
        """The moment each drone crossed its last gate [s] (n,); nan unless it
        finished."""
        return np.where(self.status == FINISHED, self.end_time, np.nan)

    def status_name(self, i: int) -> str:
        return STATUS_NAMES[self.status[i]]

    def passes(self, i: int) -> list[GatePass]:
        """Drone i's passes, as the reference's `GatePass` list."""
        result = []
        for gate in range(self.passed[i]):
            point = self.pass_point[i, gate].copy()
            point.flags.writeable = False
            result.append(GatePass(gate, float(self.pass_time[i, gate]), point))
        return result

    def __call__(self, t: float | npt.ArrayLike, states: np.ndarray) -> np.ndarray:
        """Judge one sample of every drone still flying; returns `ended` (n,)."""
        times = np.broadcast_to(np.asarray(t, dtype=float), (self.n,))
        flying = self.status == FLYING
        crashed = self.monitor(times, states, flying) & flying
        crash_time = np.where(crashed, self.monitor.time, np.inf)
        positions = states[:, 0:3]

        for gate_index, gate in enumerate(self.gates):
            judged = np.flatnonzero(
                flying & self._has_previous & (self.passed == gate_index)
            )
            if len(judged) == 0:
                continue
            t0 = self._previous_time[judged]
            crossed, fraction, point = forward_crossings(
                gate, self._previous_position[judged], positions[judged]
            )
            when = t0 + fraction * (times[judged] - t0)
            counts = crossed & (when <= crash_time[judged])
            drones = judged[counts]
            self.pass_time[drones, gate_index] = when[counts]
            self.pass_point[drones, gate_index] = point[counts]
            self.passed[drones] += 1
            last = drones[self.passed[drones] == len(self.gates)]
            self._end(last, FINISHED, self.pass_time[last, gate_index])

        still = flying & (self.status == FLYING)
        crash = np.flatnonzero(still & crashed)
        self._end(crash, CRASHED, crash_time[crash])
        still = still & ~crashed
        late = still & (times >= self.time_limit - 1e-9)
        self._end(np.flatnonzero(late), TIMEOUT, times[late])
        going = still & ~late
        self._has_previous[going] = True
        self._previous_time[going] = times[going]
        self._previous_position[going] = positions[going]
        return self.ended

    def reset(self, drones: Drones = None) -> None:
        """Start new laps for these drones (all by default): no passes, flying,
        and their monitor memory reset too."""
        rows = _mask(drones, self.n)
        self.status[rows] = FLYING
        self.end_time[rows] = np.nan
        self.passed[rows] = 0
        self.pass_time[rows] = np.nan
        self.pass_point[rows] = np.nan
        self._has_previous[rows] = False
        self.monitor.reset(rows)

    def _end(self, drones: np.ndarray, status: int, times: np.ndarray) -> None:
        """End the laps of these drones (indices) at these moments."""
        self.status[drones] = status
        self.end_time[drones] = times


def batch_referee(
    course: Course,
    sphere_centers: npt.ArrayLike,
    sphere_radii: npt.ArrayLike,
    floor_points: npt.ArrayLike,
    n: int,
) -> BatchReferee:
    """A batch referee for n drones flying laps of `course` (a `campo.course.Course`)
    with this hitbox and gear, with its own batch collision monitor: the batch
    version of `Course.referee`."""
    monitor = BatchCollisionMonitor(
        course.room,
        course.obstacles,
        sphere_centers,
        sphere_radii,
        floor_points,
        n,
        gates=course.gates,
    )
    return BatchReferee(course.gates, monitor, course.time_limit)
