"""After the last gate: the drone must come to a stop under control.

A lap ends when the last gate is crossed, so a policy trained only on laps does
not care what happens after it: phase 11's first policy crossed the last gate
rolled 144° and a real drone would have crashed a moment later (loose end 39).
With `StopAfterFinish` the episode goes on after the finish until the drone is
stopped and level (a hover that could be handed over to a position hold), and
only then counts as finished; touching anything on the way is a crash, and not
stopping within the `window` is a time-out. The lap time is still the moment
the last gate was crossed.

The drone counts as stopped when its speed is below `speed`, its tilt (the
angle between its thrust axis and the vertical) below `tilt`, and its body rates
below `rate`, all at one sample.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StopAfterFinish:
    """When a drone has stopped after the finish (see the module): speed [m/s],
    tilt [rad] and rate [rad/s] below these, within `window` [s] of the
    finish."""

    window: float = 3.0
    speed: float = 0.5
    tilt: float = float(np.radians(15.0))
    rate: float = 0.5

    def __post_init__(self) -> None:
        for name in ("window", "speed", "tilt", "rate"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be > 0")

    def stopped(self, states: np.ndarray) -> np.ndarray:
        """Which drones (n,) are stopped and level, from their states (n, 13
        or more)."""
        v = states[:, 3:6]
        w = states[:, 10:13]
        qx, qy = states[:, 7], states[:, 8]
        # cos(tilt) is the vertical component of the body z axis: R[2, 2].
        cos_tilt = 1.0 - 2.0 * (qx * qx + qy * qy)
        return (
            (np.sqrt(np.sum(v * v, axis=1)) < self.speed)
            & (cos_tilt > np.cos(self.tilt))
            & (np.sqrt(np.sum(w * w, axis=1)) < self.rate)
        )
