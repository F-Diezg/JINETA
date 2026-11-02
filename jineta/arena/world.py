"""A world: every entity of a scenario flown together in one simulation loop.

The states of all the entities are stacked into one long vector,

    [ vehicle 1 (13) | vehicle 2 (13) | point (3) | ... ]

and that vector is flown by the same `run` as everything else: one RK4 step moves
every entity at once. This works because the entities do not push on each other:
each one's derivative depends only on its own slice of the state. They meet only
through their controls (guidance + autopilot), which run once per step in
`pre_step`, as for a single vehicle.

Every step goes:
  1. the rules (`World.rules`) look at the world and may change it, e.g. launch a
     waiting interceptor;
  2. the control of every active entity runs, and all of them read the same
     snapshot of the world at the start of the step: the order of the entities
     does not matter, and nobody sees where another one will be after the step;
  3. one RK4 step for the whole vector, with the commands held (zero-order hold);
  4. post_step (renormalise the quaternions) and the stop condition.

An entity that is not active (waiting on its pad, or later destroyed) has a zero
derivative, so it stays exactly where it is, and its control is not called.

Three kinds of entity:
  - `VehicleEntity`: a simulated vehicle (13-element rigid-body state).
  - `PointEntity`: an ideal point that flies exactly the velocity it is commanded
    (3-element state: its position). For tests of guidance laws.
  - `TrackEntity`: a scripted track `t -> (position, velocity)`. Closed form, so it
    has no state (0 elements) and is evaluated at the time it is asked about.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from jineta.core.integrators import Derivative
from jineta.core.rigid_body import normalize_attitude
from jineta.core.simulation import StopCondition, run
from jineta.jinete.targets import TargetTrack

# control(t, own state, snapshot of the world): runs once per step, acts by side
# effect (sets the commands of its vehicle). The own state is the entity's slice of
# the snapshot. Both are read-only.
Control = Callable[[float, np.ndarray, "Snapshot"], None]
# rule(snapshot): runs once per step before the controls; may change the world.
Rule = Callable[["Snapshot"], None]


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("an entity needs a non-empty name")


def _read_only(value: npt.ArrayLike, shape: tuple[int, ...], what: str) -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{what} must have shape {shape}, not {array.shape}")
    array.flags.writeable = False
    return array


@dataclass
class VehicleEntity:
    """A simulated vehicle in the world.

    `dynamics` is the vehicle's `derivative(t, state)` (of a `Multirotor`, a
    `FixedWing` or a bare `RigidBody`); the vehicle object keeps its own commands,
    which `control` sets every step (usually through an autopilot).
    """

    size: ClassVar[int] = 13

    name: str
    dynamics: Derivative
    initial_state: np.ndarray  # (13,) at t = 0
    control: Control | None = None
    waiting: bool = False  # True: starts on standby until a rule launches it
    active: bool = field(init=False)

    def __post_init__(self) -> None:
        _check_name(self.name)
        self.initial_state = _read_only(self.initial_state, (13,), "initial_state")
        if not np.isclose(np.linalg.norm(self.initial_state[6:10]), 1.0):
            raise ValueError("the initial quaternion must have unit norm")
        self.active = not self.waiting

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        return self.dynamics(t, state)

    def post_step(self, state: np.ndarray) -> np.ndarray:
        return normalize_attitude(state)

    def position(self, t: float, state: np.ndarray) -> np.ndarray:
        return state[0:3]

    def velocity(self, t: float, state: np.ndarray) -> np.ndarray:
        return state[3:6]


@dataclass
class PointEntity:
    """An ideal point: it flies exactly `velocity_command`, with no lag.

    Its state is its position. Its velocity is the command it is flying, i.e. the
    one set in the previous step (in a snapshot, the controls of this step have not
    run yet).
    """

    size: ClassVar[int] = 3

    name: str
    initial_position: np.ndarray  # (3,) NED [m]
    control: Control | None = None
    waiting: bool = False
    velocity_command: np.ndarray = field(default_factory=lambda: np.zeros(3))
    active: bool = field(init=False)

    def __post_init__(self) -> None:
        _check_name(self.name)
        self.initial_position = _read_only(
            self.initial_position, (3,), "initial_position"
        )
        self.active = not self.waiting

    @property
    def initial_state(self) -> np.ndarray:
        return self.initial_position

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.array(self.velocity_command, dtype=float)

    def post_step(self, state: np.ndarray) -> np.ndarray:
        return state

    def position(self, t: float, state: np.ndarray) -> np.ndarray:
        return state

    def velocity(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.array(self.velocity_command, dtype=float)


@dataclass
class TrackEntity:
    """A scripted track in the world: no state, no control, always active."""

    size: ClassVar[int] = 0

    name: str
    track: TargetTrack
    control: None = field(default=None, init=False)
    waiting: bool = field(default=False, init=False)
    active: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        _check_name(self.name)

    @property
    def initial_state(self) -> np.ndarray:
        return np.empty(0)

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.empty(0)

    def post_step(self, state: np.ndarray) -> np.ndarray:
        return state

    def position(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.asarray(self.track(t)[0], dtype=float)

    def velocity(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.asarray(self.track(t)[1], dtype=float)


Entity = VehicleEntity | PointEntity | TrackEntity


@dataclass(frozen=True)
class Snapshot:
    """The world at one instant, as the rules and the controls see it."""

    world: "World"
    time: float  # [s]
    state: np.ndarray  # the stacked state, read-only

    def of(self, name: str) -> np.ndarray:
        """The state of one entity (its slice; empty for a track)."""
        return self.state[self.world.slice_of(name)]

    def position(self, name: str) -> np.ndarray:
        """Position of one entity, NED [m]."""
        return self.world.entity(name).position(self.time, self.of(name))

    def velocity(self, name: str) -> np.ndarray:
        """Velocity of one entity, NED [m/s]."""
        return self.world.entity(name).velocity(self.time, self.of(name))


@dataclass
class World:
    """Every entity of a scenario, flown together by one `run`.

    Args:
        entities: the entities, each with a unique name. Their order fixes where
            each state sits in the stacked vector, nothing else.
        rules: optional `rule(snapshot)`, run every step before the controls.
    """

    entities: Sequence[Entity]
    rules: Sequence[Rule] = ()
    size: int = field(init=False)  # length of the stacked state
    _slices: dict[str, slice] = field(init=False, repr=False)
    _by_name: dict[str, Entity] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.entities = tuple(self.entities)
        self.rules = tuple(self.rules)
        if not self.entities:
            raise ValueError("a world needs at least one entity")
        names = [entity.name for entity in self.entities]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"entity names must be unique: {repeated} repeated")
        self._by_name = dict(zip(names, self.entities, strict=True))
        self._slices = {}
        start = 0
        for entity in self.entities:
            self._slices[entity.name] = slice(start, start + entity.size)
            start += entity.size
        self.size = start

    def entity(self, name: str) -> Entity:
        if name not in self._by_name:
            raise KeyError(f"no entity named {name!r}")
        return self._by_name[name]

    def slice_of(self, name: str) -> slice:
        """Where the state of one entity sits in the stacked vector."""
        self.entity(name)  # a clear error for an unknown name
        return self._slices[name]

    def initial_state(self) -> np.ndarray:
        """The stacked state at t = 0."""
        return np.concatenate([e.initial_state for e in self.entities])

    def reset(self) -> None:
        """Put every entity back to its starting status (waiting or active)."""
        for entity in self.entities:
            if not isinstance(entity, TrackEntity):
                entity.active = not entity.waiting

    def snapshot(self, t: float, state: np.ndarray) -> Snapshot:
        """A read-only view of the world at time t (the caller's array is not
        changed)."""
        view = np.asarray(state).view()
        view.flags.writeable = False
        return Snapshot(self, t, view)

    def derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """d(state)/dt of the whole world: each entity on its own slice; zero for
        the entities that are not active, so they stay exactly where they are."""
        rate = np.zeros_like(state)
        for entity in self.entities:
            if entity.size and entity.active:
                part = self._slices[entity.name]
                rate[part] = entity.derivative(t, state[part])
        return rate

    def pre_step(self, t: float, state: np.ndarray) -> None:
        """Rules first, then the control of every active entity, all on the same
        snapshot (the state at the start of the step)."""
        snapshot = self.snapshot(t, state)
        for rule in self.rules:
            rule(snapshot)
        for entity in self.entities:
            if entity.active and entity.control is not None:
                entity.control(t, snapshot.of(entity.name), snapshot)

    def post_step(self, state: np.ndarray) -> np.ndarray:
        """Each active entity's correction on its slice (e.g. the quaternions)."""
        state = state.copy()
        for entity in self.entities:
            if entity.size and entity.active:
                part = self._slices[entity.name]
                state[part] = entity.post_step(state[part])
        return state

    def fly(
        self, dt: float, t_end: float, stop: StopCondition | None = None
    ) -> "Flight":
        """Fly every entity from t = 0, starting from `reset()`.

        Args:
            dt: time step [s] (also the control period of every entity).
            t_end: maximum duration [s].
            stop: optional `stop(t, stacked state) -> bool`, as in `run`.

        Controllers with memory (autopilot integrators) are not reset here: whoever
        builds the controls resets them, as `engage` does.
        """
        self.reset()
        times, states = run(
            self.derivative,
            self.initial_state(),
            dt=dt,
            t_end=t_end,
            stop=stop,
            post_step=self.post_step,
            pre_step=self.pre_step,
        )
        return Flight(self, times, states)


@dataclass(frozen=True)
class Flight:
    """The history of one `World.fly`."""

    world: World
    times: np.ndarray  # (n,) [s]
    states: np.ndarray  # (n, world.size) stacked states

    def state(self, name: str) -> np.ndarray:
        """(n, size) history of one entity's state ((n, 0) for a track)."""
        return self.states[:, self.world.slice_of(name)]

    def positions(self, name: str) -> np.ndarray:
        """(n, 3) positions of one entity, NED [m]."""
        entity = self.world.entity(name)
        return np.array(
            [
                entity.position(t, y)
                for t, y in zip(self.times, self.state(name), strict=True)
            ]
        )
