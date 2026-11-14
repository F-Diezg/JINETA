"""Where a recorded drone is at any moment between its samples.

A recording keeps a lap every environment step (0.02 s); the viewer draws many
more frames than that, and plays laps slower or faster. Between two samples the
position is interpolated along a straight line and the attitude by spherical
linear interpolation (slerp): the rotation from one attitude to the next is cut
in the same fraction, about the same axis. Interpolating the four numbers of
the quaternions linearly would not stay a rotation (its length shrinks
half-way); slerp moves along the unit sphere of quaternions at a steady rate.

q and -q are the same attitude, so the shorter way round is taken (the other
way turns the drone more than half a turn backwards between two samples).
"""

import numpy as np
import numpy.typing as npt

from jineta.picadero.recording import Lap


def slerp(q0: npt.ArrayLike, q1: npt.ArrayLike, fraction: float) -> np.ndarray:
    """The attitude a fraction of the way from q0 to q1 (unit quaternions
    [w, x, y, z]), the shorter way round; normalised."""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # -q1 is the same attitude, nearer
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly the same: a straight line is as good, and safe
        q = q0 + fraction * (q1 - q0)
        return q / np.linalg.norm(q)
    angle = np.arccos(dot)  # half the rotation between them
    q = (
        np.sin((1.0 - fraction) * angle) * q0 + np.sin(fraction * angle) * q1
    ) / np.sin(angle)
    return q / np.linalg.norm(q)


def pose_at(lap: Lap, t: float) -> tuple[np.ndarray, np.ndarray]:
    """The drone's position (NED [m]) and attitude (quaternion) at lap clock t
    [s], between the samples around it; before the start the first sample,
    after the end the last (where the lap ended)."""
    times = lap.times
    if t <= times[0]:
        return lap.positions[0].copy(), lap.attitudes[0] / np.linalg.norm(
            lap.attitudes[0]
        )
    if t >= times[-1]:
        return lap.positions[-1].copy(), lap.attitudes[-1] / np.linalg.norm(
            lap.attitudes[-1]
        )
    j = int(np.searchsorted(times, t, side="right"))  # times[j - 1] <= t < times[j]
    span = times[j] - times[j - 1]
    fraction = (t - times[j - 1]) / span if span > 0.0 else 1.0
    position = lap.positions[j - 1] + fraction * (
        lap.positions[j] - lap.positions[j - 1]
    )
    return position, slerp(lap.attitudes[j - 1], lap.attitudes[j], fraction)


def path_until(lap: Lap, t: float) -> np.ndarray:
    """The path flown up to lap clock t: the samples before t and the position
    at t (k, 3); the whole path once the lap has ended."""
    times = lap.times
    if t >= times[-1]:
        return lap.positions.copy()
    before = lap.positions[times < t]
    position, _ = pose_at(lap, t)
    return np.vstack([before, position[None, :]])
