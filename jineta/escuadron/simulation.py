"""The batch loop: n drones flying laps at once, each on its own lap clock.

The reference loop (`core.simulation.run`) flies one vehicle to the end of one
run. Training needs something else: many drones stepping together, each lap ending
on its own (a crash, the last gate, the time limit), and each drone starting its
next lap while the others fly on. `BatchFlight` keeps the states (n, 13), each
drone's lap clock (n,) and the referee, and moves everything one step at a time:

    controller(clock, states)   sets the vehicles' commands (an autopilot, or a
                                policy's rate loop), as the reference's pre_step
    one RK4 step for every drone, then the quaternions renormalised
    referee(clock, states)      judges the drones still flying

A drone whose lap has ended stays where it ended (its state and clock are not
advanced) until `reset` starts its next lap. The arithmetic is the reference's,
step for step, so a drone's lap in the batch is its lap in the reference engine
to rounding, and a drone's second lap after a reset is its first lap bit for bit.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from jineta.core.integrators import rk4_step
from jineta.escuadron.collision import Drones, _mask
from jineta.escuadron.multirotor import BatchMultirotor
from jineta.escuadron.referee import BatchReferee
from jineta.escuadron.rigid_body import normalize_attitudes

# (lap clocks (n,) [s], states (n, 13)) -> None: sets the vehicles' commands.
BatchController = Callable[[np.ndarray, np.ndarray], None]


@dataclass
class BatchFlight:
    """n drones of one profile flying laps of one course.

    vehicles: the drones (their forces and commands).
    referee: judges their laps (its monitor has the same n).
    dt: the step [s].
    start: the drones' states at the start of their first lap, (n, 13).
    """

    vehicles: BatchMultirotor
    referee: BatchReferee
    dt: float
    start: npt.ArrayLike
    states: np.ndarray = field(init=False)
    clock: np.ndarray = field(init=False)  # (n,) time since each lap started [s]

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.referee.n != self.vehicles.n:
            raise ValueError("the referee and the vehicles must have the same n")
        start = np.array(self.start, dtype=float)
        if start.shape != (self.vehicles.n, 13):
            raise ValueError("start must have shape (n, 13)")
        self.states = start
        self.clock = np.zeros(self.vehicles.n)

    @property
    def n(self) -> int:
        return self.vehicles.n

    def step(self, controller: BatchController) -> np.ndarray:
        """One step of every drone still flying; returns which laps ended in this
        step (n,) (`referee.ended` says which have ended at all)."""
        flying = ~self.referee.ended
        controller(self.clock, self.states)
        new_states = normalize_attitudes(
            rk4_step(self.vehicles.derivative, self.clock, self.states, self.dt)
        )
        self.states[flying] = new_states[flying]
        self.clock[flying] = self.clock[flying] + self.dt
        return self.referee(self.clock, self.states) & flying

    def reset(self, drones: Drones, start: npt.ArrayLike) -> None:
        """Start a new lap for these drones from `start` (one state (13,) for all of
        them, or one per drone chosen): state, lap clock and referee. Their
        controller's memory is the caller's to reset."""
        rows = _mask(drones, self.n)
        self.states[rows] = start
        self.clock[rows] = 0.0
        self.referee.reset(rows)
