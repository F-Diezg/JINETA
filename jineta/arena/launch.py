"""Launch rules: when a waiting interceptor takes off.

An interceptor that waits on its pad is a waiting entity of the world (zero
derivative: it sits still with its motors off). A launch rule is a `Rule` of the
world: once per step, before the controls, it looks at the snapshot and may launch
it (`World.launch`). The control then runs in that same step.

  - `LaunchAt`: at a fixed time.
  - `LaunchOnAlert`: at the first step at which one of the threats it watches is
    inside its alert dome (the straight-line distance to the site, as a radar
    measures it, is at most the dome radius).

  - `LaunchOnTrack`: at the first step at which a radar station's track of one of
    the threats it watches puts that threat inside the dome: the alert comes from
    what the radar believes, not from the truth.

All are instant: no reaction time, no motor spin-up. `LaunchOnAlert` reads the
truth (perfect detection); `LaunchOnTrack` reads the estimate. Put the station
before the launch rule in the world's rules, so the launch uses the scan of the
same step.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jineta.arena.sensing import RadarStation
from jineta.arena.world import Snapshot


@dataclass(frozen=True)
class Dome:
    """An alert dome: a sphere of `radius` around a site (NED)."""

    site: np.ndarray  # (3,) NED [m]
    radius: float  # [m]

    def __post_init__(self) -> None:
        site = np.array(self.site, dtype=float)
        if site.shape != (3,):
            raise ValueError("site must have 3 components")
        if not self.radius > 0.0:
            raise ValueError("radius must be > 0")
        site.flags.writeable = False
        object.__setattr__(self, "site", site)  # frozen: set once, here

    def contains(self, position: npt.ArrayLike) -> bool:
        """True when the position is inside the dome or on its surface."""
        offset = np.asarray(position, dtype=float) - self.site
        return bool(np.linalg.norm(offset) <= self.radius)


@dataclass(frozen=True)
class LaunchAt:
    """Launch `interceptor` at the first step at or after `time`."""

    interceptor: str
    time: float  # [s]

    def __call__(self, snapshot: Snapshot) -> None:
        entity = snapshot.world.entity(self.interceptor)
        if entity.launched_at is not None or entity.destroyed:
            return
        # Sample times accumulate rounding error (0.01 + 0.01 + ...), hence 1e-9.
        if snapshot.time >= self.time - 1e-9:
            snapshot.world.launch(self.interceptor, snapshot.time)


@dataclass(frozen=True)
class LaunchOnAlert:
    """Launch `interceptor` when any of `threats` is inside `dome`."""

    interceptor: str
    dome: Dome
    threats: Sequence[str]

    def __call__(self, snapshot: Snapshot) -> None:
        world = snapshot.world
        entity = world.entity(self.interceptor)
        if entity.launched_at is not None or entity.destroyed:
            return
        for threat in self.threats:
            if not world.entity(threat).destroyed and self.dome.contains(
                snapshot.position(threat)
            ):
                world.launch(self.interceptor, snapshot.time)
                return


@dataclass(frozen=True)
class LaunchOnTrack:
    """Launch `interceptor` when `station` believes any of `threats` is inside
    `dome` (its track's estimate, predicted to the current time)."""

    interceptor: str
    dome: Dome
    station: RadarStation
    threats: Sequence[str]

    def __call__(self, snapshot: Snapshot) -> None:
        world = snapshot.world
        entity = world.entity(self.interceptor)
        if entity.launched_at is not None or entity.destroyed:
            return
        for threat in self.threats:
            known = self.station.knowledge(threat, snapshot)
            if known is not None and self.dome.contains(known[0]):
                world.launch(self.interceptor, snapshot.time)
                return
