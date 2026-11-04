"""The shape a vehicle collides with: spheres fixed to its body.

A sphere of radius r touches a shape exactly when the signed distance of its
centre to the shape is less than r (see campo/indoor.py), so spheres are the
cheapest shape to check against any wall or obstacle, and a set of them can follow
a vehicle's outline as closely as needed. A vehicle touches something when any of
its spheres does.

The floor is not judged with the spheres: a vehicle resting on its landing gear has
its rotors closer to the floor than their reach, so its spheres go through the
floor while it stands still. Its landing gear points (jinete/contact.py) say when it
touches the floor.

Like the rotors and the landing gear, a hitbox is profile data: centres in body axes
(FRD) relative to the centre of mass, and radii.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Hitbox:
    """Spheres fixed to a vehicle: centres in body axes (FRD, relative to the centre
    of mass) [m], one row per sphere, and their radii [m]."""

    centers: np.ndarray
    radii: np.ndarray

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=float)
        radii = np.array(self.radii, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 3 or len(centers) == 0:
            raise ValueError("centers must have shape (n, 3), n >= 1")
        if radii.shape != (len(centers),):
            raise ValueError("radii must have one value per centre")
        if not np.all(np.isfinite(centers)):
            raise ValueError("centers must be finite")
        if not np.all(np.isfinite(radii) & (radii > 0.0)):
            raise ValueError("every radius must be finite and > 0")
        centers.flags.writeable = False
        radii.flags.writeable = False
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radii", radii)


def disc_reach(center: npt.ArrayLike, radius: float) -> float:
    """Farthest distance [m] from the centre of mass to a rotor disc.

    The disc has `radius` and its centre at `center` (body axes), and lies in the
    plane of the body x and y axes (the rotor axis is body z). A point of its rim is
    c + radius u, with u a unit vector in that plane, and
    |c + radius u|^2 = |c|^2 + 2 radius (c . u) + radius^2 is largest when u points
    along the horizontal part of c, which gives
    sqrt((|c_xy| + radius)^2 + c_z^2). (|c| + radius, from the triangle
    inequality, is a little more than that whenever c_z is not zero.)
    """
    c = np.asarray(center, dtype=float)
    return float(np.hypot(np.hypot(c[0], c[1]) + radius, c[2]))


def box_corners(size: Sequence[float]) -> np.ndarray:
    """The eight corners of a box centred on the centre of mass and aligned with the
    body axes: (+-length/2, +-width/2, +-height/2)."""
    half = np.asarray(size, dtype=float) / 2.0
    signs = np.array(
        [[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)]
    )
    return signs * half


def bounding_sphere(
    rotor_hubs: npt.ArrayLike, propeller_radius: float, body_points: npt.ArrayLike
) -> Hitbox:
    """One sphere at the centre of mass that holds every rotor disc (hub positions in
    body axes, all with the same propeller radius) and every body point."""
    hubs = np.atleast_2d(np.asarray(rotor_hubs, dtype=float))
    points = np.atleast_2d(np.asarray(body_points, dtype=float))
    reach = max(
        max(disc_reach(hub, propeller_radius) for hub in hubs),
        float(np.max(np.linalg.norm(points, axis=1))),
    )
    return Hitbox(centers=[[0.0, 0.0, 0.0]], radii=[reach])
