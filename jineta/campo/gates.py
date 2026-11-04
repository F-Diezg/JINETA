"""Gates: frames a drone must fly through, in the right direction.

A gate stands upright. Its heading `yaw` is the direction to fly through it
(clockwise from North, like a heading; yaw 0 is crossed flying North). In its own
axes a point has three coordinates:
  - n, along the heading (negative before the gate, positive after it);
  - u, sideways (positive to the right of a drone flying through);
  - v, vertical (positive down, as NED).

The opening is a rectangle (width along u, height along v) or a circle, centred on
`center`; the frame around it is solid:
  - rectangular gate: four bars of square section thickness x thickness around the
    opening;
  - circular gate: a torus (a ring-shaped tube of diameter `thickness`) around the
    opening.

Every gate is also an obstacle: `signed_distance(points)` is the distance to its
frame (the collision monitor checks it like any obstacle). `crossing(start, end)`
tells whether a straight move from `start` to `end` (two samples of the centre of
mass) goes through the opening: the plane n = 0 is cut between them, at the
fraction n0 / (n0 - n1) of the step (exact for a straight move), and the point
where it is cut must be inside the opening (or on its edge).
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from jineta.campo.indoor import (
    _distance_from_overhangs,
    _points,
    _read_only,
    _result,
)


@dataclass(frozen=True)
class Crossing:
    """A move through a gate's opening.

    fraction: where along the move the gate's plane was cut, from 0 (start) to 1.
    point: the point where it was cut (NED) [m].
    forward: True when crossed along the gate's heading, False when backwards.
    """

    fraction: float
    point: np.ndarray
    forward: bool


def _gate_axes(points: npt.ArrayLike, center: np.ndarray, yaw: float) -> np.ndarray:
    """Points in a gate's axes (n, u, v): n along the heading, u to its right, v
    down. The same turn as a box's yaw: n = c N + s E, u = -s N + c E."""
    d = _points(points) - center
    c, s = np.cos(yaw), np.sin(yaw)
    north, east = d[..., 0], d[..., 1]
    return np.stack([c * north + s * east, -s * north + c * east, d[..., 2]], -1)


def _check_common(gate: "RectangularGate | CircularGate") -> None:
    object.__setattr__(gate, "center", _read_only(gate.center, (3,), "center"))
    if not np.isfinite(gate.yaw):
        raise ValueError("yaw must be finite")
    if not gate.thickness > 0.0:
        raise ValueError("thickness must be > 0")


def _crossing(
    gate: "RectangularGate | CircularGate", start: npt.ArrayLike, end: npt.ArrayLike
) -> Crossing | None:
    """The move start -> end through the gate's opening, or None.

    Forward when the start is before the plane (n < 0) and the end on it or after
    it (n >= 0); backwards the other way round (n >= 0, then n < 0). A start exactly
    on the plane counts as after it, so a move that only touches the plane, or
    stays on it, crosses nothing.
    """
    a = gate.to_local(start)
    b = gate.to_local(end)
    if a[0] < 0.0 <= b[0]:
        forward = True
    elif b[0] < 0.0 <= a[0]:
        forward = False
    else:
        return None
    fraction = float(a[0] / (a[0] - b[0]))
    local = a + fraction * (b - a)
    if not gate.in_opening(local[1], local[2]):
        return None
    start = np.asarray(start, dtype=float)
    point = start + fraction * (np.asarray(end, dtype=float) - start)
    point.flags.writeable = False
    return Crossing(fraction, point, forward)


@dataclass(frozen=True)
class RectangularGate:
    """A rectangular opening (width x height [m]) centred on `center` (NED), in a
    frame of four square bars `thickness` [m] thick; crossed along `yaw` [rad]."""

    center: np.ndarray
    yaw: float
    width: float
    height: float
    thickness: float

    def __post_init__(self) -> None:
        _check_common(self)
        if not (self.width > 0.0 and self.height > 0.0):
            raise ValueError("width and height must be > 0")

    def to_local(self, points: npt.ArrayLike) -> np.ndarray:
        """Points in the gate's axes (n, u, v)."""
        return _gate_axes(points, self.center, self.yaw)

    def in_opening(self, u: float, v: float) -> bool:
        """Whether a point of the gate's plane (u, v) is inside the opening."""
        return abs(u) <= self.width / 2.0 and abs(v) <= self.height / 2.0

    def signed_distance(self, points: npt.ArrayLike) -> float | np.ndarray:
        """Distance [m] to the frame: > 0 outside it (in the opening too), < 0 inside
        a bar.

        In the gate's plane the frame is the ring between two rectangles: inside
        the outer one (opening + thickness all round) and outside the opening, so
        its distance is max(d_outer, -d_opening) (exact: inside the opening the
        nearest bar is across the nearest edge of the opening, outside the outer
        rectangle it is the outer rectangle). Then the ring is given its depth,
        thickness along n, as a box is: overhangs (in-plane distance, |n| - t/2).
        """
        p = self.to_local(points)
        uv = np.abs(p[..., 1:3])
        half = np.array([self.width, self.height]) / 2.0
        outer = _distance_from_overhangs(uv - (half + self.thickness))
        opening = _distance_from_overhangs(uv - half)
        ring = np.maximum(outer, -opening)
        depth = np.abs(p[..., 0]) - self.thickness / 2.0
        return _result(_distance_from_overhangs(np.stack([ring, depth], axis=-1)))

    def crossing(self, start: npt.ArrayLike, end: npt.ArrayLike) -> Crossing | None:
        """The move start -> end through the opening, or None (see module doc)."""
        return _crossing(self, start, end)


@dataclass(frozen=True)
class CircularGate:
    """A round opening of `radius` [m] centred on `center` (NED), in a ring-shaped
    tube `thickness` [m] across (a torus); crossed along `yaw` [rad]."""

    center: np.ndarray
    yaw: float
    radius: float
    thickness: float

    def __post_init__(self) -> None:
        _check_common(self)
        if not self.radius > 0.0:
            raise ValueError("radius must be > 0")

    def to_local(self, points: npt.ArrayLike) -> np.ndarray:
        """Points in the gate's axes (n, u, v)."""
        return _gate_axes(points, self.center, self.yaw)

    def in_opening(self, u: float, v: float) -> bool:
        """Whether a point of the gate's plane (u, v) is inside the opening."""
        return u * u + v * v <= self.radius**2

    def signed_distance(self, points: npt.ArrayLike) -> float | np.ndarray:
        """Distance [m] to the frame, a torus: > 0 outside it, < 0 inside the tube.

        The tube's centre line is a circle of radius R = radius + thickness/2 in
        the gate's plane. The nearest point of that circle to p is in the
        direction of p in the plane, at (sqrt(u^2 + v^2) - R) sideways and n
        across the plane: the distance to the circle is the hypotenuse, and to the
        tube's surface that minus the tube's radius, thickness/2.
        """
        p = self.to_local(points)
        ring = np.hypot(p[..., 1], p[..., 2]) - (self.radius + self.thickness / 2.0)
        return _result(np.hypot(ring, p[..., 0]) - self.thickness / 2.0)

    def crossing(self, start: npt.ArrayLike, end: npt.ArrayLike) -> Crossing | None:
        """The move start -> end through the opening, or None (see module doc)."""
        return _crossing(self, start, end)


Gate = RectangularGate | CircularGate
