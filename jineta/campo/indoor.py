"""Indoor geometry: the room and its obstacles, described by their signed distance.

The signed distance of a shape at a point is how far the point is from the shape's
surface: positive outside, zero on the surface, negative inside (then its size is
how deep the point is). It is the true Euclidean distance, so one number answers
both "is it touching?" (distance < 0) and "how much room is left?" (distance > 0),
and a sphere of radius r touches the shape exactly when the distance of its centre
is less than r.

Every shape here has `signed_distance(points)`: one NED point (shape (3,)) gives a
float, an array of points (shape (n, 3)) gives an array of n distances, so the same
code serves one drone or many.

Shapes:
  - Box: a rectangular block with a centre, a size and a yaw (turned about the
    vertical; yaw 0 has its length along North, positive yaw turns it towards
    East, as a heading).
  - Cylinder: a vertical column standing on a base point, with a radius and a
    height.
  - Room: the inside of a rectangular room, with one corner of its floor at the
    origin. Its `clearance` is the distance to the nearest wall, the floor or the
    ceiling: positive inside the room, negative outside. Its `wall_clearance`
    leaves the floor out (a drone standing on the floor is not hitting it; its
    landing gear says when it touches the floor).

Conventions: NED axes, so "up" is -z and the floor of the room is z = 0.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


def _points(points: npt.ArrayLike) -> np.ndarray:
    """One point (3,) or several (n, 3) as a float array; anything else is an error."""
    p = np.asarray(points, dtype=float)
    if p.shape[-1:] != (3,) or p.ndim > 2:
        raise ValueError("points must have shape (3,) or (n, 3)")
    return p


def _result(distance: np.ndarray) -> float | np.ndarray:
    """A float for one point, an array for several."""
    return float(distance) if distance.ndim == 0 else distance


def _read_only(value: npt.ArrayLike, shape: tuple[int, ...], what: str) -> np.ndarray:
    array = np.array(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{what} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{what} must be finite")
    array.flags.writeable = False
    return array


def box_distance(local_points: np.ndarray, half_size: np.ndarray) -> np.ndarray:
    """Signed distance to a box centred at the origin and aligned with the axes.

    For each axis, q = |p| - half size is how far the point sticks out past that
    face (negative: how far inside it). Outside, the distance is the length of the
    parts that stick out (past one face it is that overhang; past an edge or a
    corner, Pythagoras over two or three of them). Inside, every q is negative and
    the nearest face is the one with the largest q.
    """
    return _distance_from_overhangs(np.abs(local_points) - half_size)


def _distance_from_overhangs(q: np.ndarray) -> np.ndarray:
    """Signed distance from how far a point sticks out past each face (q, last axis).

    Valid for any shape that is the intersection of half-spaces at right angles to
    each other (a box, a box open on some sides): outside, Pythagoras over the
    overhangs that are positive; inside, the nearest face is the largest q.
    """
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside


@dataclass(frozen=True)
class Box:
    """A rectangular block: centre (NED) [m], size [length, width, height] [m], yaw.

    The length lies along the box's own x axis, which points North at yaw 0 and
    turns towards East with positive yaw [rad]. Height is always vertical.
    """

    center: np.ndarray
    size: np.ndarray
    yaw: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _read_only(self.center, (3,), "center"))
        object.__setattr__(self, "size", _read_only(self.size, (3,), "size"))
        if not np.all(self.size > 0.0):
            raise ValueError("every side of a box must be > 0")
        if not np.isfinite(self.yaw):
            raise ValueError("yaw must be finite")

    def to_local(self, points: npt.ArrayLike) -> np.ndarray:
        """Points in the box's own axes: origin at its centre, x along its length.

        A yaw turns the box's axes into NED as R = [[cos, -sin, 0], [sin, cos, 0],
        [0, 0, 1]]; going back is the transpose, R^T (p - centre).
        """
        d = _points(points) - self.center
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        north, east = d[..., 0], d[..., 1]
        return np.stack([c * north + s * east, -s * north + c * east, d[..., 2]], -1)

    def signed_distance(self, points: npt.ArrayLike) -> float | np.ndarray:
        """Distance [m] from the box's surface: > 0 outside, < 0 inside."""
        return _result(box_distance(self.to_local(points), self.size / 2.0))


@dataclass(frozen=True)
class Cylinder:
    """A vertical column: base point (NED, centre of its bottom face) [m], radius [m]
    and height [m]. It rises from the base towards -z."""

    base: np.ndarray
    radius: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "base", _read_only(self.base, (3,), "base"))
        if not self.radius > 0.0:
            raise ValueError("radius must be > 0")
        if not self.height > 0.0:
            raise ValueError("height must be > 0")

    def signed_distance(self, points: npt.ArrayLike) -> float | np.ndarray:
        """Distance [m] from the column's surface: > 0 outside, < 0 inside.

        The same idea as the box, in two directions instead of three: how far the
        point sticks out past the curved side (horizontal distance to the axis
        minus the radius) and past the top or bottom face (vertical distance to
        the middle height minus half the height).
        """
        d = _points(points) - self.base
        radial = np.hypot(d[..., 0], d[..., 1]) - self.radius
        vertical = np.abs(d[..., 2] + self.height / 2.0) - self.height / 2.0
        q = np.stack([radial, vertical], axis=-1)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return _result(outside + inside)


@dataclass(frozen=True)
class Room:
    """The inside of a rectangular room: size [length North, width East, height] [m].

    One corner of the floor is at the origin: the room spans 0 <= x <= length,
    0 <= y <= width, -height <= z <= 0 (the floor is z = 0, the ceiling is above it).
    """

    size: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "size", _read_only(self.size, (3,), "size"))
        if not np.all(self.size > 0.0):
            raise ValueError("every side of a room must be > 0")

    @property
    def center(self) -> np.ndarray:
        """The middle of the room (NED) [m]."""
        return np.array([self.size[0] / 2.0, self.size[1] / 2.0, -self.size[2] / 2.0])

    def clearance(self, points: npt.ArrayLike) -> float | np.ndarray:
        """Distance [m] to the nearest wall, floor or ceiling: > 0 inside the room,
        < 0 outside. The room is the inside of a box, so this is the box's signed
        distance with the sign turned round."""
        local = _points(points) - self.center
        return _result(-box_distance(local, self.size / 2.0))

    def wall_clearance(self, points: npt.ArrayLike) -> float | np.ndarray:
        """Distance [m] to the nearest wall or the ceiling, the floor left out:
        > 0 inside, < 0 past a wall or above the ceiling.

        Without its floor the room is a box open at the bottom that reaches down
        forever. Its overhangs are the box's for the walls (|x - length/2| -
        length/2, the same for y) and, for the ceiling at z = -height, how far the
        point is above it: -height - z. Below the floor nothing changes: only the
        walls and the ceiling count.
        """
        p = _points(points)
        length, width, height = self.size
        q = np.stack(
            [
                np.abs(p[..., 0] - length / 2.0) - length / 2.0,
                np.abs(p[..., 1] - width / 2.0) - width / 2.0,
                -height - p[..., 2],
            ],
            axis=-1,
        )
        return _result(-_distance_from_overhangs(q))

    def contains(self, points: npt.ArrayLike) -> bool | np.ndarray:
        """True where a point is inside the room or on its walls."""
        inside = np.asarray(self.clearance(points)) >= 0.0
        return bool(inside) if inside.ndim == 0 else inside
