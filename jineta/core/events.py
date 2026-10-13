"""Events between samples: find when something happened inside a time step.

The simulation only knows the state at the sample times. An event such as the
closest approach between two vehicles usually happens between two samples; taking
the nearest sample can be off by up to v_rel * dt / 2. Inside one step the relative
motion is taken as a straight line, which is exact for constant relative velocity
and good to O(dt^2) otherwise.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ClosestApproach:
    """Smallest separation between two tracks, interpolated inside a step."""

    time: float  # [s]
    distance: float  # miss distance [m]
    step: int  # index i of the step [times[i], times[i+1]] that contains it


def closest_approach(
    times: npt.ArrayLike, relative_positions: npt.ArrayLike
) -> ClosestApproach:
    """Closest approach from sampled relative positions r_i = r_target - r_pursuer.

    Inside each step r(s) = r_i + s (r_(i+1) - r_i), s in [0, 1]. Its length is
    smallest at s* = -r_i . d / (d . d), with d = r_(i+1) - r_i, clipped to [0, 1].
    The smallest over all the steps is returned.
    """
    times = np.asarray(times, dtype=float)
    r = np.asarray(relative_positions, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3 or len(r) != len(times) or len(times) < 2:
        raise ValueError("need n >= 2 times and an (n, 3) array of relative positions")

    start, change = r[:-1], r[1:] - r[:-1]
    change_squared = (change * change).sum(axis=1)
    s = np.zeros(len(start))  # steps with no relative motion keep s = 0
    moving = change_squared > 0.0
    s[moving] = -(start[moving] * change[moving]).sum(axis=1) / change_squared[moving]
    s = np.clip(s, 0.0, 1.0)
    distances = np.linalg.norm(start + s[:, None] * change, axis=1)

    step = int(np.argmin(distances))
    time = times[step] + s[step] * (times[step + 1] - times[step])
    return ClosestApproach(time=float(time), distance=float(distances[step]), step=step)
