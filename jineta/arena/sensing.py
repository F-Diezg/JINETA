"""Sensors in the world: who knows what about whom.

The snapshot of the world is the truth. Until phase 9 every control read it
directly: each interceptor knew exactly where its threat was. Now there are two
ways of knowing, behind one interface, a `Knowledge`:

    knowledge(target, snapshot) -> (position, velocity) NED, or None

  - `perfect_knowledge`: the truth from the snapshot (the baseline of phases 1-8).
  - `RadarStation.knowledge`: the estimate of a radar's track of that target,
    predicted to the time of the snapshot; None while the radar has no track of
    it yet.

A `RadarStation` is a rule of the world: once per step, before the controls, it
reads the true positions of the targets it watches (it is the only participant
that reads the truth), and when its radar is due it measures them and feeds each
measurement to that target's track. Rules run in their order, so a station placed
before the launch rules and the controls lets them use this step's measurements.

The referee (`engagement.py`) still judges with the truth: it is the umpire, not a
participant.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from jineta.arena.world import Snapshot
from jineta.atalaya.radar import Measurement, Radar
from jineta.atalaya.tracker import Track

# knowledge(target name, snapshot) -> (position NED [m], velocity NED [m/s]) or None
Knowledge = Callable[[str, Snapshot], tuple[np.ndarray, np.ndarray] | None]


def perfect_knowledge(target: str, snapshot: Snapshot) -> tuple[np.ndarray, np.ndarray]:
    """The truth: where the target is in the snapshot, and its velocity."""
    return snapshot.position(target), snapshot.velocity(target)


@dataclass(frozen=True)
class TrackSettings:
    """How a station tracks one kind of target (see `Track`)."""

    process_noise: float  # q [m^2/s^3]
    speed_prior: float  # [m/s]


@dataclass(frozen=True)
class TrackRecord:
    """A track right after one of its updates (for plots and the viewer)."""

    time: float  # [s]
    x: np.ndarray  # (6,) NED position and velocity
    P: np.ndarray  # (6, 6)


@dataclass
class RadarStation:
    """A radar with one track per target it watches, as a rule of the world.

    Args:
        radar: the radar (with its site and seed).
        targets: the targets it watches and how to track each, by entity name.
            The order fixes the order of the measurements in a scan (and so of the
            noise draws).

    The station has memory (its tracks and its radar's noise sequence); the world
    calls `reset` at the start of every flight, so flying again repeats the run.
    """

    radar: Radar
    targets: Mapping[str, TrackSettings]
    tracks: dict[str, Track] = field(init=False)
    measurements: list[Measurement] = field(init=False)  # every one, in order
    history: dict[str, list[TrackRecord]] = field(init=False)  # per target

    def __post_init__(self) -> None:
        self.targets = dict(self.targets)
        if not self.targets:
            raise ValueError("a radar station needs at least one target")
        self.reset()

    def reset(self) -> None:
        """Back to the start: no tracks, no measurements, the radar's noise from
        its beginning."""
        self.radar.reset()
        self.tracks = {
            name: Track(name, s.process_noise, s.speed_prior)
            for name, s in self.targets.items()
        }
        self.measurements = []
        self.history = {name: [] for name in self.targets}

    def __call__(self, snapshot: Snapshot) -> None:
        """Measure the targets when the radar is due and update their tracks. A
        destroyed target is no longer measured (its track is left as it was)."""
        if not self.radar.due(snapshot.time):
            return
        world = snapshot.world
        truths = {
            name: snapshot.position(name)
            for name in self.targets
            if not world.entity(name).destroyed
        }
        for measurement in self.radar.scan(snapshot.time, truths):
            track = self.tracks[measurement.target]
            track.update(measurement)
            self.measurements.append(measurement)
            self.history[measurement.target].append(
                TrackRecord(measurement.time, track.x.copy(), track.P.copy())
            )

    def knowledge(
        self, target: str, snapshot: Snapshot
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """A `Knowledge`: the track's estimate of `target` at the snapshot's time,
        or None while there is no track of it yet."""
        if target not in self.tracks:
            raise KeyError(f"this radar station does not watch {target!r}")
        track = self.tracks[target]
        if not track.ready:
            return None
        x, _ = track.estimate(snapshot.time)
        return x[:3], x[3:]
