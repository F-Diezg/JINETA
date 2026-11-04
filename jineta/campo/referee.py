"""The referee of a lap: gates in order, lap time, and how the lap ended.

A lap starts at t = 0 (the drone standing on the floor) and ends in one of three
ways:
  - "finished": the last gate is crossed; the lap time is the moment it was
    crossed (interpolated inside the step);
  - "crashed": the collision monitor found a contact (wall, ceiling, obstacle, gate
    frame, or the floor after take-off);
  - "timeout": the time limit is reached first.
While none of them has happened the status is "flying".

Gates count only in order and forward: at each sample only the next gate is
checked, for a forward crossing between the previous sample and this one. A gate
crossed backwards, or a later gate crossed first, does not count and is not a crash:
the drone can still turn back and fly through the right gate.

Two events in the same step: the one that happened first wins, by their
interpolated moments. A drone that crosses the last gate and touches a wall a few
milliseconds later has finished; one that touches the frame before its centre goes
through has crashed. At most one gate is counted per step (gates are much farther
apart than a step: 0.2 m at 20 m/s).

The referee is the `stop` of simulation.run, so the run ends with the lap. Like the
monitor, it sees the samples after the first step (run never shows `stop` the
starting state), so nothing can be crossed during the very first step.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from jineta.campo.collision import CollisionMonitor
from jineta.campo.gates import Gate


@dataclass(frozen=True)
class GatePass:
    """A gate crossed in order: its index, the moment [s] and the point (NED) [m]."""

    gate: int
    time: float
    point: np.ndarray


@dataclass
class Referee:
    """Judges a lap through `gates` (in order), with `monitor` watching for
    collisions (its gates should be these, so their frames count) and a
    `time_limit` [s]. Callable as the `stop` of simulation.run."""

    gates: Sequence[Gate]
    monitor: CollisionMonitor
    time_limit: float
    status: str = field(default="flying", init=False)
    passes: list[GatePass] = field(default_factory=list, init=False)
    end_time: float | None = field(default=None, init=False)
    _previous: tuple[float, np.ndarray] | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.gates = tuple(self.gates)
        if len(self.gates) == 0:
            raise ValueError("a lap needs at least one gate")
        if not self.time_limit > 0.0:
            raise ValueError("time_limit must be > 0")

    @property
    def next_gate(self) -> int | None:
        """Index of the gate to cross next; None once every gate is crossed."""
        n = len(self.passes)
        return n if n < len(self.gates) else None

    @property
    def lap_time(self) -> float | None:
        """The moment the last gate was crossed [s]; None unless finished."""
        return self.end_time if self.status == "finished" else None

    def __call__(self, t: float, state: np.ndarray) -> bool:
        """Judge one sample; True once the lap has ended."""
        if self.status != "flying":
            return True
        position = state[0:3].copy()
        crashed = self.monitor(t, state)
        crash_time = self.monitor.collision.time if crashed else np.inf

        gate = self.next_gate
        if self._previous is not None and gate is not None:
            t0, p0 = self._previous
            crossing = self.gates[gate].crossing(p0, position)
            if crossing is not None and crossing.forward:
                pass_time = t0 + crossing.fraction * (t - t0)
                if pass_time <= crash_time:
                    self.passes.append(GatePass(gate, pass_time, crossing.point))
                    if self.next_gate is None:
                        return self._end("finished", pass_time)

        if crashed:
            return self._end("crashed", crash_time)
        if t >= self.time_limit - 1e-9:
            return self._end("timeout", t)
        self._previous = (t, position)
        return False

    def reset(self) -> None:
        """Start a new lap: no passes, flying, and the monitor reset too."""
        self.status = "flying"
        self.passes = []
        self.end_time = None
        self._previous = None
        self.monitor.reset()

    def _end(self, status: str, time: float) -> bool:
        self.status = status
        self.end_time = time
        return True
